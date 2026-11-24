"""Temporal stabilization helpers.

Single-frame AI outputs flicker: a car is labelled "truck" for two frames, a
traffic light reads "UNKNOWN" for one frame due to LED flicker / compression, an
OCR pass misreads one character. These helpers vote over time so downstream
logic (counting, violation rules, evidence) sees stable values.
"""
from collections import Counter, defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple


class TrackClassVoter:
    """Confidence-weighted majority vote of the class label per track ID."""

    def __init__(self, window: int = 15):
        self.window = max(1, int(window))
        self._history: Dict[int, Deque[Tuple[int, float]]] = defaultdict(lambda: deque(maxlen=self.window))

    def update(self, track_id: int, class_id: int, conf: float = 1.0) -> int:
        if track_id < 0:
            return class_id
        hist = self._history[track_id]
        hist.append((class_id, float(conf)))
        scores: Counter = Counter()
        for cid, c in hist:
            scores[cid] += c
        return scores.most_common(1)[0][0]

    def forget(self, track_id: int) -> None:
        self._history.pop(track_id, None)


class SignalSmoother:
    """Majority vote over the last N frames for each signal key.

    UNKNOWN/OFF readings never outvote a real colour; a colour must win the
    window to replace the current stable state, which suppresses 1-2 frame
    glitches while still reacting within ``window`` frames.
    """

    _IGNORED = {"UNKNOWN", "OFF"}

    def __init__(self, window: int = 5):
        self.window = max(1, int(window))
        self._history: Dict[str, Deque[str]] = defaultdict(lambda: deque(maxlen=self.window))

    def update(self, states: Dict[str, str]) -> Dict[str, str]:
        if self.window <= 1:
            return dict(states)
        smoothed: Dict[str, str] = {}
        for key, value in states.items():
            if key == "mode":
                smoothed[key] = value
                continue
            hist = self._history[key]
            hist.append(value)
            votes = Counter(v for v in hist if v not in self._IGNORED)
            smoothed[key] = votes.most_common(1)[0][0] if votes else value
        return smoothed


class PlateVoter:
    """Accumulates OCR reads for a track and returns the consensus plate.

    Each read votes with its OCR confidence; identical strings accumulate.
    When there is no exact agreement, a per-character vote across reads of the
    same length/format is used.
    """

    def __init__(self):
        self._reads: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
        self._ocr_calls: Dict[int, int] = defaultdict(int)
        self._det_calls: Dict[int, int] = defaultdict(int)
        self._last_frame: Dict[int, int] = {}
        self._best_area: Dict[int, int] = defaultdict(int)

    def should_sample(self, track_id: int, frame_number: int, interval: int, max_reads: int) -> bool:
        """Whether to run the plate detector for this track on this frame.

        Detector calls are capped at ``3 * max_reads`` and OCR calls at ``max_reads``.
        """
        if track_id < 0 or self._ocr_calls[track_id] >= max_reads or self._det_calls[track_id] >= 3 * max_reads:
            return False
        last = self._last_frame.get(track_id)
        return last is None or frame_number - last >= max(1, interval)

    def mark_sampled(self, track_id: int, frame_number: int) -> None:
        self._last_frame[track_id] = frame_number
        self._det_calls[track_id] += 1

    def is_better_crop(self, track_id: int, area: int) -> bool:
        """Only spend OCR on plate crops at least ~as large as the best seen so far."""
        return area >= 0.8 * self._best_area[track_id]

    def add(self, track_id: int, text: str, conf: float, crop_area: int = 0) -> None:
        self._ocr_calls[track_id] += 1
        self._best_area[track_id] = max(self._best_area[track_id], crop_area)
        if text:
            self._reads[track_id].append((text, max(0.01, float(conf))))

    def result(self, track_id: int) -> Tuple[str, float]:
        reads = self._reads.get(track_id)
        if not reads:
            return "", 0.0
        totals: Counter = Counter()
        for text, conf in reads:
            totals[text] += conf
        best, score = totals.most_common(1)[0]
        if len(totals) == 1 or score >= 0.5 * sum(totals.values()):
            return best, score / sum(1 for t, _ in reads if t == best)

        # Character-level vote among reads with the same layout as the best one
        same_len = [(t, c) for t, c in reads if len(t) == len(best)]
        if len(same_len) >= 2:
            chars = []
            for i in range(len(best)):
                col: Counter = Counter()
                for t, c in same_len:
                    col[t[i]] += c
                chars.append(col.most_common(1)[0][0])
            fused = "".join(chars)
            return fused, sum(c for _, c in same_len) / len(same_len)
        return best, score

    def has_result(self, track_id: int) -> bool:
        return bool(self._reads.get(track_id))

    def forget(self, track_id: int) -> None:
        for d in (self._reads, self._ocr_calls, self._det_calls, self._last_frame, self._best_area):
            d.pop(track_id, None)

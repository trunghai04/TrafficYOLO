import csv
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from ultralytics import YOLO

from .config import load_config
from .evidence import create_combined_evidence, detect_license_plate_crop, patch_plate_model_compat
from .ocr import recognize_license_plate_text
from .geometry import clamp_box, get_line_side, is_red_light_crossing
from .tracker_utils import (
    is_track_continuous,
    is_vehicle_box_plausible,
    recover_oversized_vehicle_boxes,
)
from .traffic_light import (
    classify_approach_signal,
    classify_multiple_signals,
    evaluate_directional_violation,
)
from .trajectory import compute_movement_vector, classify_vehicle_direction

SHORT_VEHICLE_TYPES = {
    "motorcycle": "MOTO",
    "car": "CAR",
    "bus": "BUS",
    "truck": "TRUCK",
}
GREEN = (50, 220, 50)
RED = (0, 0, 255)


def _patch_yolov5_compat():
    """Apply compatibility shims between legacy YOLOv5 models and Ultralytics YOLOv8 engine."""
    try:
        import yolov5.models.yolo
        if hasattr(yolov5.models.yolo, "BaseModel"):
            orig_fuse = yolov5.models.yolo.BaseModel.fuse
            def _patched_fuse(self, *args, **kwargs):
                return orig_fuse(self)
            yolov5.models.yolo.BaseModel.fuse = _patched_fuse

        if hasattr(yolov5.models.yolo, "DetectionModel"):
            orig_forward = yolov5.models.yolo.DetectionModel.forward
            def _patched_forward(self, *args, **kwargs):
                kwargs.pop("embed", None)
                return orig_forward(self, *args, **kwargs)
            yolov5.models.yolo.DetectionModel.forward = _patched_forward
    except Exception:
        pass


class TrafficMonitorPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        _patch_yolov5_compat()
        
        # Normalize compute device for YOLO
        import torch
        raw_device = str(config["models"].get("device", "auto")).lower()
        if raw_device in ("auto", "cuda"):
            self.device = "0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = raw_device
        self.config["models"]["device"] = self.device

        self.vehicle_classes = {int(k): v for k, v in config["vehicle_classes"].items()}

        # Load main vehicle detector
        model_path = config["models"]["vehicle_detector"]
        print(f"Loading YOLO vehicle model from: {model_path} (Device: {self.device})")
        self.model = YOLO(model_path)
        self.recovery_model = None

        # Load license plate detector
        plate_model_path = Path(config["models"]["plate_detector"])
        self.plate_model = None
        if plate_model_path.exists():
            print(f"Loading license plate detector from: {plate_model_path}")
            try:
                self.plate_model = YOLO(str(plate_model_path))
                patch_plate_model_compat(self.plate_model)
            except Exception as e:
                print(f"[WARN] Failed to load plate detector ({e}). Proceeding without plate crops.")
        else:
            print(f"[INFO] Plate detector not found at {plate_model_path}. Proceeding without plate crops.")

    def run(
        self,
        video_source: Optional[str] = None,
        output_dir: Optional[str] = None,
        save_video: Optional[bool] = None,
        save_csv: Optional[bool] = None,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute traffic analysis pipeline on input video."""
        io_cfg = self.config["io"]
        video_path = Path(video_source or io_cfg["video_path"])
        out_dir = Path(output_dir or io_cfg["output_dir"])
        should_save_video = save_video if save_video is not None else io_cfg["save_video"]
        should_save_csv = save_csv if save_csv is not None else io_cfg["save_csv"]

        if not video_path.exists():
            raise FileNotFoundError(f"Video input file not found: {video_path.resolve()}")

        evidence_dir = out_dir / "evidence"
        out_dir.mkdir(parents=True, exist_ok=True)
        evidence_dir.mkdir(parents=True, exist_ok=True)

        output_video_path = out_dir / io_cfg["video_filename"]
        output_csv_path = out_dir / io_cfg["csv_filename"]

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        writer = None
        if should_save_video:
            writer = cv2.VideoWriter(
                str(output_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

        # Config parameters
        line_start = tuple(self.config["counting_line"]["start"])
        line_end = tuple(self.config["counting_line"]["end"])
        line_margin = self.config["counting_line"]["margin"]
        max_crossing_frame_gap = self.config["counting_line"]["max_crossing_frame_gap"]

        light_roi = tuple(self.config["traffic_light"]["roi"])
        light_min_pixels = self.config["traffic_light"]["min_pixels"]
        sat_min = self.config["traffic_light"]["hsv"]["saturation_min"]
        val_min = self.config["traffic_light"]["hsv"]["value_min"]

        max_track_speed = self.config["tracking"]["max_track_speed"]
        max_missed_frames = self.config["tracking"]["max_missed_frames"]
        max_violation_missed_frames = self.config["tracking"]["max_violation_missed_frames"]

        tracker_config = Path(self.config["models"]["tracker_config"])

        # Signals Configuration
        signals_dict = self.config.get("traffic_signals", {})
        if not signals_dict or not any(k.endswith("_light") for k in signals_dict):
            signals_dict = {
                "straight_light": {
                    "name": "Đèn Đi Thẳng",
                    "roi": light_roi,
                    "min_pixels": light_min_pixels,
                    "saturation_min": sat_min,
                    "value_min": val_min,
                }
            }

        is_single_signal = (
            signals_dict.get("mode") == "single"
            or not signals_dict.get("has_right_turn_light", True)
            or "right_turn_light" not in signals_dict
            or (
                signals_dict.get("right_turn_light", {}).get("roi") is not None
                and signals_dict.get("straight_light", {}).get("roi") is not None
                and list(signals_dict.get("right_turn_light", {}).get("roi")) == list(signals_dict.get("straight_light", {}).get("roi"))
            )
        )
        allow_right_on_red = bool(signals_dict.get("allow_right_turn_on_red", False))

        # State tracking
        previous_side = {}
        tracked_vehicles = {}
        track_history = {}  # track_id -> [(cx, cy), ...]
        track_plates = {}   # track_id -> license_plate_str
        pending_crossings = {}  # track_id -> {crossing_frame, signal_states, ...}
        counted_ids = set()
        violation_ids = set()
        counts = Counter()
        events = []
        frame_number = 0

        print(f"Starting traffic analysis on: {video_path.name} ({width}x{height}, ~{total_frames} frames)...")

        map_riders = bool(self.config.get("tracking", {}).get("map_road_riders_to_motorcycle", False))
        track_classes = list(self.vehicle_classes.keys())
        if map_riders and 0 not in track_classes:
            track_classes.append(0)

        results = self.model.track(
            source=str(video_path.resolve()),
            tracker=str(tracker_config.resolve()) if tracker_config.exists() else "bytetrack.yaml",
            conf=self.config["inference"]["confidence"],
            imgsz=self.config["inference"]["image_size"],
            device=self.device,
            classes=track_classes,
            persist=True,
            stream=True,
            verbose=False,
        )

        for result in results:
            frame_number += 1
            seen_track_ids = set()
            raw_frame = result.orig_img.copy()
            frame = raw_frame.copy()

            # Classify all directional signals
            signal_states = classify_multiple_signals(raw_frame, signals_dict)
            straight_state = signal_states.get("straight_light", "UNKNOWN")
            right_turn_state = signal_states.get("right_turn_light", "UNKNOWN")
            signal_state = straight_state  # primary signal state for fallback

            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().tolist()
                raw_class_ids = boxes.cls.int().cpu().tolist()
                track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [-1] * len(raw_class_ids)

                # Nighttime / Low-light rider mapping: only map person (class 0) to motorcycle if explicitly enabled
                # AND person is not an upright walking pedestrian (ratio >= 1.95)
                class_ids = []
                for b, cid in zip(xyxy, raw_class_ids):
                    if map_riders and cid == 0:
                        bw = b[2] - b[0]
                        bh = b[3] - b[1]
                        ratio = bh / max(1.0, bw)
                        cy = (b[1] + b[3]) / 2
                        is_walking_pedestrian = (ratio >= 1.95 and bw < 65)
                        if not is_walking_pedestrian and cy >= 450:
                            class_ids.append(3)  # motorcycle
                        else:
                            class_ids.append(cid)  # person (ignored)
                    else:
                        class_ids.append(cid)

                # Recover oversized boxes if detected
                needs_recovery = any(
                    cid in self.vehicle_classes
                    and not is_vehicle_box_plausible(
                        clamp_box(b, width, height),
                        self.vehicle_classes[cid],
                        width,
                        height,
                    )
                    for b, cid in zip(xyxy, class_ids)
                )
                if needs_recovery:
                    if self.recovery_model is None:
                        self.recovery_model = YOLO(self.config["models"]["vehicle_detector"])
                    xyxy = recover_oversized_vehicle_boxes(
                        self.recovery_model,
                        raw_frame,
                        xyxy,
                        class_ids,
                        width,
                        height,
                        self.vehicle_classes,
                        recovery_confidence=self.config["inference"]["recovery_confidence"],
                        image_size=self.config["inference"]["image_size"],
                        device=self.device,
                    )

                for box, class_id, track_id in zip(xyxy, class_ids, track_ids):
                    if class_id not in self.vehicle_classes:
                        continue

                    vehicle_type = self.vehicle_classes[class_id]
                    vehicle_box = clamp_box(box, width, height)
                    x1, y1, x2, y2 = vehicle_box
                    if x2 <= x1 or y2 <= y1:
                        continue
                    if not is_vehicle_box_plausible(vehicle_box, vehicle_type, width, height):
                        continue

                    previous_track = tracked_vehicles.get(track_id) if track_id >= 0 else None
                    if previous_track is not None and not is_track_continuous(
                        previous_track, vehicle_box, vehicle_type, frame_number
                    ):
                        tracked_vehicles.pop(track_id, None)
                        previous_side.pop(track_id, None)
                        if previous_track["violation"]:
                            violation_ids.discard(track_id)
                        previous_track = None

                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    current_side = get_line_side(
                        center_x, center_y, line_start, line_end, margin=line_margin
                    ) if track_id >= 0 else None

                    prev_obs = previous_side.get(track_id) if track_id >= 0 else None
                    prev_is_recent = prev_obs is not None and (frame_number - prev_obs[1] <= max_crossing_frame_gap)
                    old_side = prev_obs[0] if prev_is_recent else None

                    min_line_x = min(line_start[0], line_end[0])
                    max_line_x = max(line_start[0], line_end[0])

                    if current_side is not None:
                        previous_side[track_id] = (current_side, frame_number)
                    elif min_line_x <= center_x <= max_line_x and prev_is_recent:
                        previous_side[track_id] = (prev_obs[0], frame_number)
                    elif not (min_line_x <= center_x <= max_line_x):
                        previous_side.pop(track_id, None)

                    # Update track motion history
                    if track_id >= 0:
                        if track_id not in track_history:
                            track_history[track_id] = []
                        track_history[track_id].append((center_x, center_y))
                        if len(track_history[track_id]) > 25:
                            track_history[track_id].pop(0)

                    # Register vehicle crossing the line into pending evaluations
                    standard_cross = (old_side == "below" and current_side == "above")
                    
                    # Detect fast motorcycles emerging from blind spots (e.g. from behind the large stopped bus)
                    blindspot_cross = False
                    if vehicle_type == "motorcycle" and old_side is None and current_side == "above" and min_line_x <= center_x <= max_line_x:
                        line_y_at_x = line_start[1] + (center_x - line_start[0]) * (line_end[1] - line_start[1]) / max(1, line_end[0] - line_start[0])
                        # If motorcycle suddenly appears in the crossing corridor ahead of the bus
                        if line_y_at_x - 200 <= center_y <= line_y_at_x:
                            blindspot_cross = True

                    just_crossed_line = (
                        track_id >= 0
                        and (standard_cross or blindspot_cross)
                        and track_id not in counted_ids
                        and track_id not in pending_crossings
                    )

                    if just_crossed_line:
                        counted_ids.add(track_id)
                        counts[vehicle_type] += 1
                        bx1, by1, bx2, by2 = clamp_box(vehicle_box, width, height)
                        pad = 15
                        cx1 = max(0, bx1 - pad)
                        cy1 = max(0, by1 - pad)
                        cx2 = min(width, bx2 + pad)
                        cy2 = min(height, by2 + pad)
                        crop_img = raw_frame[cy1:cy2, cx1:cx2].copy()
                        pending_crossings[track_id] = {
                            "crossing_frame": frame_number,
                            "entry_point": (center_x, center_y),
                            "vehicle_type": vehicle_type,
                            "vehicle_box": vehicle_box,
                            "evidence_crop": crop_img,
                            "raw_crossing_frame": raw_frame.copy(),
                            "signal_states_at_cross": signal_states.copy(),
                        }
                        # Pre-read plate for vehicle crossing if not already captured
                        if track_id not in track_plates and self.plate_model is not None and crop_img.size > 0:
                            p_zoom = detect_license_plate_crop(crop_img, self.plate_model, device=self.device)
                            if p_zoom is not None:
                                p_num, _ = recognize_license_plate_text(p_zoom, use_gpu=(self.device != "cpu"))
                                if p_num:
                                    track_plates[track_id] = p_num

                    # Evaluate pending crossings
                    is_red_light_violation = False
                    violation_details = None

                    if track_id in pending_crossings:
                        p_info = pending_crossings[track_id]
                        elapsed_cross = frame_number - p_info["crossing_frame"]
                        entry_x, entry_y = p_info["entry_point"]
                        shift_x = center_x - entry_x
                        shift_y = center_y - entry_y
                        dist_moved = np.hypot(shift_x, shift_y)

                        # Interpolate stop line y at current vehicle x
                        if line_end[0] != line_start[0]:
                            line_y_at_x = line_start[1] + (center_x - line_start[0]) * (line_end[1] - line_start[1]) / (line_end[0] - line_start[0])
                        else:
                            line_y_at_x = line_start[1]

                        # Genuine forward progression into intersection check:
                        # Moving into intersection requires moving forward (shift_y <= -20)
                        # and having center_y past the stop line (center_y <= line_y_at_x - 10).
                        is_forward_progression = (shift_y <= -20) or (shift_y <= -12 and abs(shift_x) >= 25)
                        is_past_stop_line = (center_y <= line_y_at_x - 10)

                        if not (is_forward_progression and is_past_stop_line):
                            # Vehicle is stationary, creeping, or waiting at the red light stop line
                            if elapsed_cross >= 6:
                                pending_crossings.pop(track_id, None)
                        else:
                            min_wait = 2 if p_info["vehicle_type"] == "motorcycle" else 4
                            if elapsed_cross >= min_wait:
                                actual_dir = classify_vehicle_direction(
                                    shift_x, shift_y,
                                    right_turn_threshold=float(self.config["tracking"].get("turn_right_threshold", 25.0)),
                                    left_turn_threshold=float(self.config["tracking"].get("turn_left_threshold", -25.0)),
                                )
                                # Strictly evaluate directional violation with current signal status and lane position
                                is_viol, sig_resp, resp_col = evaluate_directional_violation(
                                    actual_dir,
                                    p_info["signal_states_at_cross"],
                                    current_signal_states=signal_states,
                                    vehicle_x=entry_x,
                                    dx=shift_x,
                                    has_separate_right_light=not is_single_signal,
                                    allow_right_turn_on_red=allow_right_on_red,
                                )

                                if is_viol:
                                    is_red_light_violation = True
                                    violation_ids.add(track_id)
                                    violation_details = {
                                        "actual_dir": actual_dir,
                                        "sig_resp": sig_resp,
                                        "resp_col": resp_col,
                                        "info": p_info,
                                    }
                                # Remove from pending once evaluated
                                pending_crossings.pop(track_id, None)

                    display_box = vehicle_box
                    if track_id >= 0:
                        velocity_x = 0.0
                        velocity_y = 0.0
                        smoothed_box = vehicle_box
                        if previous_track is not None:
                            elapsed = max(1, frame_number - previous_track["last_seen"])
                            old_box = previous_track["box"]
                            smoothed_box = tuple(
                                round(0.65 * ob + 0.35 * nb)
                                for ob, nb in zip(old_box, vehicle_box)
                            )
                            old_cx = (old_box[0] + old_box[2]) / 2
                            old_cy = (old_box[1] + old_box[3]) / 2
                            velocity_x = float(np.clip((center_x - old_cx) / elapsed, -max_track_speed, max_track_speed))
                            velocity_y = float(np.clip((center_y - old_cy) / elapsed, -max_track_speed, max_track_speed))

                        tracked_vehicles[track_id] = {
                            "box": smoothed_box,
                            "vehicle_type": vehicle_type,
                            "last_seen": frame_number,
                            "velocity_x": velocity_x,
                            "velocity_y": velocity_y,
                            "violation": (track_id in violation_ids) or is_red_light_violation,
                        }
                        seen_track_ids.add(track_id)
                        display_box = smoothed_box

                    dx1, dy1, dx2, dy2 = clamp_box(display_box, width, height)
                    box_color = RED if (track_id in violation_ids or is_red_light_violation) else GREEN
                    cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), box_color, 2, cv2.LINE_AA)

                    vehicle_label = SHORT_VEHICLE_TYPES.get(vehicle_type, vehicle_type.upper())
                    plate_num = track_plates.get(track_id)
                    if plate_num:
                        label = f"{vehicle_label} | {plate_num}"
                    elif track_id >= 0:
                        label = f"{vehicle_label} | ID: {track_id}"
                    else:
                        label = f"{vehicle_label}"

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.43
                    thickness = 1
                    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                    if dy1 >= text_h + baseline + 8:
                        label_top = dy1 - text_h - baseline - 5
                    else:
                        label_top = min(height - text_h - baseline - 5, dy1 + 3)

                    label_bottom = label_top + text_h + baseline + 5
                    label_right = min(width - 1, dx1 + text_w + 8)
                    cv2.rectangle(frame, (dx1, label_top), (label_right, label_bottom), box_color, -1)
                    cv2.putText(frame, label, (dx1 + 4, label_bottom - baseline - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                    if is_red_light_violation and violation_details is not None:
                        event_number = len(events) + 1
                        event_dir = evidence_dir / f"event_{event_number:04d}"
                        event_dir.mkdir(parents=True, exist_ok=True)

                        actual_dir = violation_details["actual_dir"]
                        sig_resp = violation_details["sig_resp"]
                        resp_col = violation_details["resp_col"]
                        p_info = violation_details["info"]

                        evidence_crop = p_info.get("evidence_crop", raw_frame[dy1:dy2, dx1:dx2])
                        vehicle_crop_path = event_dir / "vehicle_crop.jpg"
                        cv2.imwrite(str(vehicle_crop_path), evidence_crop)

                        evidence_frame = frame.copy()
                        cv2.putText(
                            evidence_frame,
                            f"VIOLATION: {actual_dir.upper()} | SIGNAL: {resp_col} ({sig_resp}) | FRAME: {frame_number}",
                            (10, height - 18),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            RED,
                            1,
                            cv2.LINE_AA,
                        )
                        evidence_path = event_dir / "evidence.jpg"

                        plate_zoom = detect_license_plate_crop(evidence_crop, self.plate_model, device=self.device)
                        plate_str = track_plates.get(track_id, "")
                        if not plate_str and plate_zoom is not None:
                            plate_str, _ = recognize_license_plate_text(plate_zoom, use_gpu=(self.device != "cpu"))
                            if plate_str:
                                track_plates[track_id] = plate_str

                        combined_evidence = create_combined_evidence(
                            evidence_frame,
                            plate_zoom,
                            plate_text=plate_str,
                            vehicle_crop=evidence_crop,
                        )
                        cv2.imwrite(str(evidence_path), combined_evidence)

                        events.append({
                            "event_id": event_number,
                            "frame": frame_number,
                            "track_id": track_id,
                            "license_plate": plate_str if plate_str else f"ID_{track_id} (Chưa rõ)",
                            "vehicle_type": vehicle_type,
                            "violation_type": f"red_light_{actual_dir}",
                            "movement_direction": actual_dir,
                            "signal_responsible": sig_resp,
                            "signal_state": resp_col,
                            "center_x": round(center_x, 1),
                            "center_y": round(center_y, 1),
                            "evidence_image": str(evidence_path),
                            "vehicle_crop": str(vehicle_crop_path),
                            "plate_crop": "embedded_in_evidence" if plate_zoom is not None else "",
                        })
                        print(f"Violation #{event_number}: Plate={plate_str or track_id}, Type={vehicle_type}, Dir={actual_dir}, Signal={resp_col} ({sig_resp}), Frame={frame_number}")

            # Prune obsolete tracks
            for track_id, state in list(tracked_vehicles.items()):
                if track_id in seen_track_ids:
                    continue
                missed = frame_number - state["last_seen"]
                max_miss = max_violation_missed_frames if state["violation"] else max_missed_frames
                if missed > max_miss:
                    tracked_vehicles.pop(track_id)
                    previous_side.pop(track_id, None)
                    track_history.pop(track_id, None)
                    if state["violation"]:
                        violation_ids.discard(track_id)

            # Draw HUD
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (360, 145), (15, 20, 25), -1)
            frame = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)
            cv2.putText(frame, "AI TRAFFIC MONITOR", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Car: {counts['car']}  Moto: {counts['motorcycle']}", (18, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1, cv2.LINE_AA)
            cv2.putText(frame, f"Bus: {counts['bus']}  Truck: {counts['truck']}", (18, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1, cv2.LINE_AA)
            cv2.putText(frame, f"Violations: {len(events)}", (18, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Show signal states on HUD
            if is_single_signal:
                sig_col = RED if straight_state == "RED" else (GREEN if straight_state == "GREEN" else YELLOW)
                cv2.putText(frame, f"Traffic Signal: {straight_state}", (18, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.42, sig_col, 1, cv2.LINE_AA)
            else:
                straight_col = RED if straight_state == "RED" else GREEN
                right_col = RED if right_turn_state == "RED" else GREEN
                cv2.putText(frame, f"Straight Signal: {straight_state}", (18, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.4, straight_col, 1, cv2.LINE_AA)
                cv2.putText(frame, f"Right-Turn Signal: {right_turn_state}", (18, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.4, right_col, 1, cv2.LINE_AA)

            # Draw virtual stop line with state color
            stop_line_col = RED if straight_state == "RED" else GREEN
            cv2.line(frame, line_start, line_end, stop_line_col, 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                "STOP LINE",
                (line_start[0] + 15, max(20, line_start[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                stop_line_col,
                1,
                cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(frame)

            if progress_callback is not None:
                progress_callback(frame_number, total_frames, signal_state, dict(counts), len(events))

            if frame_number % 100 == 0:
                print(f"Processed {frame_number}/{total_frames} frames...")

        if writer is not None:
            writer.release()

        # Export CSV
        if should_save_csv:
            fieldnames = [
                "event_id", "frame", "track_id", "license_plate", "vehicle_type", "violation_type",
                "movement_direction", "signal_responsible", "signal_state",
                "center_x", "center_y", "evidence_image", "vehicle_crop", "plate_crop"
            ]
            with output_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
                csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
                csv_writer.writeheader()
                csv_writer.writerows(events)

        return {
            "total_frames": frame_number,
            "counts": dict(counts),
            "violations_count": len(events),
            "events": events,
            "output_video": str(output_video_path) if should_save_video else None,
            "output_csv": str(output_csv_path) if should_save_csv else None,
            "evidence_dir": str(evidence_dir),
        }

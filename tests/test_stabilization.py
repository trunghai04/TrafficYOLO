from traffic_yolo.stabilization import PlateVoter, SignalSmoother, TrackClassVoter


def test_class_voter_suppresses_flicker():
    voter = TrackClassVoter(window=10)
    for _ in range(6):
        assert voter.update(1, 2, 0.8) == 2  # car
    # A single high-ish confidence "truck" frame must not flip the label
    assert voter.update(1, 7, 0.9) == 2
    # Untracked detections pass through unchanged
    assert voter.update(-1, 7, 0.9) == 7


def test_class_voter_forget_resets_history():
    voter = TrackClassVoter(window=5)
    voter.update(3, 2, 0.9)
    voter.forget(3)
    assert voter.update(3, 7, 0.5) == 7


def test_signal_smoother_ignores_single_frame_glitch():
    sm = SignalSmoother(window=5)
    for _ in range(4):
        out = sm.update({"straight_light": "RED", "mode": "single"})
    assert out["straight_light"] == "RED"
    out = sm.update({"straight_light": "GREEN", "mode": "single"})
    assert out["straight_light"] == "RED"
    assert out["mode"] == "single"


def test_signal_smoother_unknown_never_wins_over_colour():
    sm = SignalSmoother(window=5)
    sm.update({"straight_light": "GREEN"})
    for _ in range(3):
        out = sm.update({"straight_light": "UNKNOWN"})
    assert out["straight_light"] == "GREEN"


def test_signal_smoother_switches_after_majority():
    sm = SignalSmoother(window=5)
    for _ in range(5):
        sm.update({"straight_light": "GREEN"})
    for _ in range(3):
        out = sm.update({"straight_light": "RED"})
    assert out["straight_light"] == "RED"


def test_plate_voter_majority_and_char_fusion():
    pv = PlateVoter()
    pv.add(5, "30E-514.31", 0.6)
    pv.add(5, "30E-514.31", 0.5)
    pv.add(5, "30E-574.31", 0.4)
    assert pv.result(5)[0] == "30E-514.31"

    pv2 = PlateVoter()
    pv2.add(9, "29A-123.45", 0.5)
    pv2.add(9, "29A-128.45", 0.4)
    pv2.add(9, "20A-123.45", 0.45)
    # Per-character vote recovers the consistent string
    assert pv2.result(9)[0] == "29A-123.45"


def test_plate_voter_sampling_limits():
    pv = PlateVoter()
    assert pv.should_sample(1, 1, interval=3, max_reads=2)
    pv.mark_sampled(1, 1)
    assert not pv.should_sample(1, 2, interval=3, max_reads=2)
    assert pv.should_sample(1, 4, interval=3, max_reads=2)
    pv.add(1, "", 0.0)
    pv.add(1, "", 0.0)
    assert not pv.should_sample(1, 10, interval=3, max_reads=2)
    assert not pv.has_result(1)

import cv2
import numpy as np
from traffic_yolo.traffic_light import (
    classify_approach_signal,
    classify_multiple_signals,
    evaluate_directional_violation,
)


def test_classify_traffic_light_colors():
    height, width = 200, 200
    roi = (20, 20, 60, 60)

    # 1. Red signal test
    frame_red = np.zeros((height, width, 3), dtype=np.uint8)
    # Fill ROI with saturated bright red (BGR: 0, 0, 255)
    frame_red[20:60, 20:60] = (0, 0, 255)
    color = classify_approach_signal(frame_red, roi=roi, min_pixels=8)
    assert color == "RED"

    # 2. Green signal test
    frame_green = np.zeros((height, width, 3), dtype=np.uint8)
    # Fill ROI with saturated bright green (BGR: 0, 255, 0)
    frame_green[20:60, 20:60] = (0, 255, 0)
    color = classify_approach_signal(frame_green, roi=roi, min_pixels=8)
    assert color == "GREEN"

    # 3. Yellow signal test
    frame_yellow = np.zeros((height, width, 3), dtype=np.uint8)
    # Fill ROI with saturated bright yellow (BGR: 0, 255, 255)
    frame_yellow[20:60, 20:60] = (0, 255, 255)
    color = classify_approach_signal(frame_yellow, roi=roi, min_pixels=8)
    assert color == "YELLOW"

    # 4. Dark / Off signal test
    frame_dark = np.zeros((height, width, 3), dtype=np.uint8)
    color = classify_approach_signal(frame_dark, roi=roi, min_pixels=8)
    assert color == "UNKNOWN"


def test_single_signal_mode_and_coinciding_rois():
    height, width = 200, 200
    frame_red = np.zeros((height, width, 3), dtype=np.uint8)
    frame_red[20:60, 20:60] = (0, 0, 255)  # RED

    frame_green = np.zeros((height, width, 3), dtype=np.uint8)
    frame_green[20:60, 20:60] = (0, 255, 0)  # GREEN

    # 1. User configures coinciding ROIs: straight and right-turn are identical
    coinciding_signals = {
        "straight_light": {"roi": [20, 20, 60, 60]},
        "right_turn_light": {"roi": [20, 20, 60, 60]},
    }
    states_red = classify_multiple_signals(frame_red, coinciding_signals)
    assert states_red["straight_light"] == "RED"
    assert states_red["right_turn_light"] == "RED"
    assert states_red.get("mode") == "single"

    states_green = classify_multiple_signals(frame_green, coinciding_signals)
    assert states_green["straight_light"] == "GREEN"
    assert states_green["right_turn_light"] == "GREEN"

    # 2. Violation evaluation in single signal mode:
    # On GREEN: All movements are 100% lawful
    viol, _, _ = evaluate_directional_violation(
        "turn_right", states_green, has_separate_right_light=False
    )
    assert viol is False

    viol_st, _, _ = evaluate_directional_violation(
        "straight", states_green, has_separate_right_light=False
    )
    assert viol_st is False

    # On RED without permitted right-turn sign:
    # All movements crossing line are violations
    viol_red_st, _, _ = evaluate_directional_violation(
        "straight", states_red, has_separate_right_light=False, allow_right_turn_on_red=False
    )
    assert viol_red_st is True

    viol_red_rt, _, _ = evaluate_directional_violation(
        "turn_right", states_red, has_separate_right_light=False, allow_right_turn_on_red=False
    )
    assert viol_red_rt is True

    # On RED with permitted right-turn sign:
    # Right-turn is allowed, straight is violation
    viol_red_rt_allowed, _, resp_col = evaluate_directional_violation(
        "turn_right", states_red, has_separate_right_light=False, allow_right_turn_on_red=True
    )
    assert viol_red_rt_allowed is False
    assert "RẼ PHẢI HỢP LỆ" in resp_col

    viol_red_st_disallowed, _, _ = evaluate_directional_violation(
        "straight", states_red, has_separate_right_light=False, allow_right_turn_on_red=True
    )
    assert viol_red_st_disallowed is True

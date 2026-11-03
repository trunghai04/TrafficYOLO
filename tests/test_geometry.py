import pytest
from traffic_yolo.geometry import clamp_box, get_line_side, is_red_light_crossing


def test_clamp_box():
    # Outside negative bounds
    assert clamp_box((-10, -5, 100, 200), 640, 480) == (0, 0, 100, 200)
    # Outside positive bounds
    assert clamp_box((100, 200, 700, 500), 640, 480) == (100, 200, 640, 480)
    # Inside normal bounds
    assert clamp_box((10, 20, 100, 150), 640, 480) == (10, 20, 100, 150)


def test_get_line_side():
    line_start = (0, 100)
    line_end = (200, 100)  # Horizontal line at y=100

    # Outside x range
    assert get_line_side(250, 50, line_start, line_end) is None
    assert get_line_side(-10, 50, line_start, line_end) is None

    # Within margin zone (margin=5: y in [95, 105])
    assert get_line_side(100, 100, line_start, line_end, margin=5) is None
    assert get_line_side(100, 104, line_start, line_end, margin=5) is None

    # Clearly above (smaller y in image coordinates)
    assert get_line_side(100, 80, line_start, line_end, margin=5) == "above"

    # Clearly below (greater y in image coordinates)
    assert get_line_side(100, 120, line_start, line_end, margin=5) == "below"


def test_is_red_light_crossing():
    # Legitimate red light violation: below -> above during RED
    assert is_red_light_crossing("below", "above", "RED") is True

    # Same side: not a crossing
    assert is_red_light_crossing("below", "below", "RED") is False
    assert is_red_light_crossing("above", "above", "RED") is False

    # Reverse direction (above -> below)
    assert is_red_light_crossing("above", "below", "RED") is False

    # Crossing during GREEN or YELLOW: not a red light violation
    assert is_red_light_crossing("below", "above", "GREEN") is False
    assert is_red_light_crossing("below", "above", "YELLOW") is False

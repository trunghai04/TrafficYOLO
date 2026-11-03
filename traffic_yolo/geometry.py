from typing import Tuple, Optional
import numpy as np


def clamp_box(box: Tuple[float, float, float, float] | list, width: int, height: int) -> Tuple[int, int, int, int]:
    """Clamp bounding box coordinates to image boundaries."""
    x1, y1, x2, y2 = map(int, box)
    return (
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )


def get_line_side(
    center_x: float,
    center_y: float,
    line_start: Tuple[int, int],
    line_end: Tuple[int, int],
    margin: int = 5,
) -> Optional[str]:
    """
    Determine if a point (center_x, center_y) is 'above' or 'below' the virtual line.
    
    Returns:
        'above': if the point is above the line + margin.
        'below': if the point is below the line - margin.
        None: if outside line x-range or within the margin zone.
    """
    x1, y1 = line_start
    x2, y2 = line_end

    min_x, max_x = min(x1, x2), max(x1, x2)
    if not (min_x <= center_x <= max_x):
        return None

    # Interpolate y coordinate of the line at center_x
    if x2 == x1:
        return None
    line_y = y1 + (center_x - x1) * (y2 - y1) / (x2 - x1)

    if center_y >= line_y + margin:
        return "below"
    if center_y <= line_y - margin:
        return "above"
    return None


def is_red_light_crossing(
    previous_side: Optional[str],
    current_side: Optional[str],
    signal_state: str,
) -> bool:
    """Check if the vehicle crossed the virtual line during red light."""
    return previous_side == "below" and current_side == "above" and signal_state == "RED"

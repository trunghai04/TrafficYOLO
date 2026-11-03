from typing import List, Tuple, Optional, Dict
import numpy as np


def compute_movement_vector(
    history: List[Tuple[int, int]],
    min_points: int = 3,
) -> Tuple[float, float, float]:
    """
    Compute displacement dx, dy and angle (degrees) from track coordinate history.
    
    Returns:
        (dx, dy, angle_degrees)
        dx > 0 indicates rightward movement.
        dy < 0 indicates upward movement (into intersection).
        angle_degrees: 0° is right, -90° is straight up, 180°/-180° is left.
    """
    if len(history) < min_points:
        return 0.0, 0.0, 0.0

    # Compare oldest in window vs latest
    start_x, start_y = history[0]
    end_x, end_y = history[-1]

    dx = float(end_x - start_x)
    dy = float(end_y - start_y)
    angle = float(np.degrees(np.arctan2(dy, dx)))

    return dx, dy, angle


def classify_vehicle_direction(
    dx: float,
    dy: float,
    right_turn_threshold: float = 35.0,
    left_turn_threshold: float = -35.0,
) -> str:
    """
    Classify movement into 'turn_right', 'turn_left', or 'straight'.
    
    In standard camera perspective looking towards an intersection:
    - Moving into the intersection means dy is negative (y decreases as vehicles move away).
    - Substantial positive dx means veering right (turning right).
    - Substantial negative dx means veering left (turning left).
    """
    # If dx is strongly positive, vehicle is turning right
    if dx >= right_turn_threshold:
        return "turn_right"
    # If dx is strongly negative, vehicle is turning left
    if dx <= left_turn_threshold:
        return "turn_left"
    return "straight"

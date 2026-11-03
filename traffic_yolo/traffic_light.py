from typing import Tuple, Dict, Any, List, Optional
import cv2
import numpy as np


def classify_lamp_hsv(
    crop: np.ndarray,
    min_pixels: int = 15,
    saturation_min: int = 90,
    value_min: int = 90,
) -> Tuple[str, int]:
    """Classify dominant color of a lamp crop using calibrated HSV thresholds."""
    if crop is None or crop.size == 0:
        return "UNKNOWN", 0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    bright = (s >= saturation_min) & (v >= value_min)

    red_mask = bright & ((h <= 10) | (h >= 170))
    yellow_mask = bright & (h >= 11) & (h <= 34)
    green_mask = bright & (h >= 35) & (h <= 95)

    r_cnt = int(np.count_nonzero(red_mask))
    y_cnt = int(np.count_nonzero(yellow_mask))
    g_cnt = int(np.count_nonzero(green_mask))

    counts = {"RED": r_cnt, "YELLOW": y_cnt, "GREEN": g_cnt}
    color, max_cnt = max(counts.items(), key=lambda item: item[1])

    if max_cnt < min_pixels:
        return "OFF", max_cnt
    return color, max_cnt


def auto_detect_signals(frame: np.ndarray) -> Tuple[str, List[int], List[int]]:
    """
    Automatically detect the camera perspective and return calibrated ROIs:
    - 12h wide-angle camera: right pole is at far right [1525, 345, 1585, 530]
    - 16h/18h camera: right pole is at [1375, 280, 1445, 415]
    
    Returns:
        (preset_name, straight_roi, right_turn_roi)
    """
    h, w = frame.shape[:2]
    sx = w / 1600.0
    sy = h / 1200.0

    # Candidate A: 12h wide angle (far right pole x ~ 1520..1590, y ~ 290..530)
    c_a = frame[int(290 * sy):int(530 * sy), int(1520 * sx):int(1590 * sx)]
    # Candidate B: 16h/18h angle (mid-right pole x ~ 1370..1450, y ~ 270..420)
    c_b = frame[int(270 * sy):int(420 * sy), int(1370 * sx):int(1450 * sx)]

    def score_active_lamps(crop):
        if crop is None or crop.size == 0:
            return 0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        s, v = hsv[:, :, 1], hsv[:, :, 2]
        return int(np.count_nonzero((s >= 70) & (v >= 100)))

    score_a = score_active_lamps(c_a)
    score_b = score_active_lamps(c_b)

    if score_a > score_b * 1.3:
        return "12h_wide", [1525, 345, 1585, 530], [1525, 345, 1585, 530]
    else:
        return "16h_standard", [545, 15, 610, 85], [1375, 280, 1445, 415]


def classify_right_pole_assembly(frame: np.ndarray, roi: Optional[List[int]] = None) -> str:
    """
    Intelligently classify the right-pole traffic light assembly.
    Supports both user-specified ROI and dynamic auto-detection between:
    - 12h wide-angle camera: [1525, 345, 1585, 530]
    - 16h standard camera: [1375, 280, 1445, 415]
    """
    h, w = frame.shape[:2]
    sx = w / 1600.0
    sy = h / 1200.0

    # Auto-resolve if ROI is not provided or if pointing at legacy default coordinates
    if roi is None or (roi[0] <= 1400 and roi[2] <= 1460):
        _, _, auto_rp = auto_detect_signals(frame)
        if roi is None:
            roi = auto_rp
        elif auto_rp[0] > 1500:
            # Video is actually 12h wide-angle, switch dynamically to 1525..1585
            roi = auto_rp

    rx1 = max(0, min(w - 1, int(roi[0] * sx)))
    ry1 = max(0, min(h - 1, int(roi[1] * sy)))
    rx2 = max(1, min(w, int(roi[2] * sx)))
    ry2 = max(1, min(h, int(roi[3] * sy)))

    if rx2 <= rx1 or ry2 <= ry1:
        return "UNKNOWN"

    pole_crop = frame[ry1:ry2, rx1:rx2]
    if pole_crop.size == 0:
        return "UNKNOWN"

    ph = ry2 - ry1
    # 3 vertically stacked lamps inside the assembly
    red_crop = pole_crop[0 : int(ph * 0.38), :]
    yellow_crop = pole_crop[int(ph * 0.33) : int(ph * 0.67), :]
    green_crop = pole_crop[int(ph * 0.62) : ph, :]

    r_color, r_cnt = classify_lamp_hsv(red_crop, min_pixels=25)
    y_color, y_cnt = classify_lamp_hsv(yellow_crop, min_pixels=25)
    g_color, g_cnt = classify_lamp_hsv(green_crop, min_pixels=25)

    if g_color == "GREEN" and g_cnt >= 30 and g_cnt > r_cnt:
        return "GREEN"
    if r_color == "RED" and r_cnt >= 25 and r_cnt > g_cnt:
        return "RED"
    if y_color == "YELLOW" and y_cnt >= 25:
        return "YELLOW"

    return "RED" if r_cnt > g_cnt else "UNKNOWN"


def classify_approach_signal(
    frame: np.ndarray,
    roi: Tuple[int, int, int, int] | List[int],
    min_pixels: int = 8,
    saturation_min: int = 80,
    value_min: int = 85,
) -> str:
    """Classify the traffic light color in a designated Region of Interest (ROI)."""
    x1, y1, x2, y2 = roi
    # If ROI points to the tall vertical right-pole assembly, delegate to 3-lamp pole classifier
    if x1 >= 1480 and (y2 - y1) >= 80:
        return classify_right_pole_assembly(frame, roi=roi)

    h, w = frame.shape[:2]
    x1, y1 = max(0, min(w - 1, int(x1))), max(0, min(h - 1, int(y1)))
    x2, y2 = max(1, min(w, int(x2))), max(1, min(h, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return "UNKNOWN"

    lamp_roi = frame[y1:y2, x1:x2]
    if lamp_roi.size == 0:
        return "UNKNOWN"

    lh, lw = lamp_roi.shape[:2]
    # For vertical multi-aspect traffic lights (height >= 35):
    # Enforce vertical lamp positioning to prevent background yellow walls or sun glare from overpowering red/green bulbs
    if lh >= 35:
        top_slice = lamp_roi[0 : int(lh * 0.40), :]
        mid_slice = lamp_roi[int(lh * 0.30) : int(lh * 0.70), :]
        bot_slice = lamp_roi[int(lh * 0.60) :, :]

        def count_color(crop, h_min, h_max):
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            s_c, v_c, hue = hsv[:, :, 1], hsv[:, :, 2], hsv[:, :, 0]
            bright = (s_c >= saturation_min) & (v_c >= value_min)
            if h_min <= 0:
                return int(np.count_nonzero(bright & ((hue <= h_max) | (hue >= 170))))
            return int(np.count_nonzero(bright & (hue >= h_min) & (hue <= h_max)))

        r_top = count_color(top_slice, -1, 10)
        y_mid = count_color(mid_slice, 12, 32)
        g_bot = count_color(bot_slice, 35, 95)

        threshold = max(min_pixels, 15)
        if g_bot >= threshold and g_bot > r_top:
            return "GREEN"
        if r_top >= threshold and r_top > g_bot:
            return "RED"
        if y_mid >= threshold:
            return "YELLOW"

    color, cnt = classify_lamp_hsv(lamp_roi, min_pixels=min_pixels, saturation_min=saturation_min, value_min=value_min)
    return color if color != "OFF" else "UNKNOWN"


def classify_multiple_signals(
    frame: np.ndarray,
    signals_dict: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    """
    Classify all defined traffic lights simultaneously.
    Supports both:
    1. Single Signal Mode (Chỉ 1 cụm đèn chung cho toàn bộ nút giao)
    2. Multi-Signal Directional Mode (Đèn đi thẳng riêng + Đèn rẽ phải riêng)
    """
    states = {}
    is_single_mode = (
        signals_dict.get("mode") == "single"
        or not signals_dict.get("has_right_turn_light", True)
        or "right_turn_light" not in signals_dict
    )

    straight_cfg = signals_dict.get("straight_light", {})
    roi_straight = straight_cfg.get("roi")

    right_cfg = signals_dict.get("right_turn_light", {})
    roi_right = right_cfg.get("roi")

    # If user set both ROIs to the same coordinates, automatically treat as single traffic light!
    if roi_right is not None and roi_straight is not None and list(roi_right) == list(roi_straight):
        is_single_mode = True

    if is_single_mode:
        # SINGLE TRAFFIC LIGHT: Only classify the designated approach signal
        if roi_straight is None:
            _, auto_straight, _ = auto_detect_signals(frame)
            roi_straight = auto_straight
        single_color = classify_approach_signal(frame, roi=roi_straight)
        states["straight_light"] = single_color
        states["right_turn_light"] = single_color
        states["traffic_light"] = single_color
        states["mode"] = "single"
        return states

    # MULTI-SIGNAL DIRECTIONAL MODE
    mode, auto_straight, auto_right = auto_detect_signals(frame)
    if roi_right is None:
        roi_right = auto_right
    elif mode == "12h_wide" and roi_right[0] <= 1400:
        roi_right = auto_right
    elif mode == "16h_standard" and roi_right[0] >= 1500:
        roi_right = auto_right

    if roi_straight is None:
        roi_straight = auto_straight
    elif mode == "12h_wide" and roi_straight[1] <= 100:
        roi_straight = auto_straight
    elif mode == "16h_standard" and roi_straight[1] > 200:
        roi_straight = auto_straight

    states["right_turn_light"] = classify_right_pole_assembly(frame, roi=roi_right)
    states["straight_light"] = classify_approach_signal(frame, roi=roi_straight)
    states["mode"] = "directional"
    return states


def evaluate_directional_violation(
    vehicle_direction: str,
    signal_states: Dict[str, str],
    current_signal_states: Optional[Dict[str, str]] = None,
    vehicle_x: Optional[float] = None,
    dx: Optional[float] = None,
    right_turn_lane_x: float = 750.0,
    has_separate_right_light: bool = True,
    allow_right_turn_on_red: bool = False,
) -> Tuple[bool, str, str]:
    """
    Evaluate violation with strict adherence to directional signals and traffic rules:
    
    1. LAWFUL GREEN SIGNAL RULE (Đèn xanh tuyệt đối được phép đi):
       - If the responsible signal is GREEN either at crossing line OR currently when clearing,
         the vehicle is obeying the green light. NEVER flag violation on green light!
       
    2. RIGHT TURN DIRECTION (Đèn rẽ phải):
       - Right-turn light controls the right turn corridor (right lane x >= right_turn_lane_x or rightward motion dx > 10).
       - If right_turn_light is GREEN: All vehicles turning right or in the right corridor are 100% lawful!
       - If right_turn_light is RED or YELLOW and vehicle is turning right: VIOLATION!
       
    3. STRAIGHT & LEFT DIRECTION (Đèn đi thẳng):
       - Straight light controls straight movement and left turns.
       - If straight_light is GREEN: STRICTLY PERMITTED! NEVER flag straight/left vehicles on green!
       - If straight_light is RED: VIOLATION!
       
    4. FAIL-SAFE / BENEFIT OF THE DOUBT:
       - If direction is ambiguous and either signal is green matching vehicle position, do not penalize.
    """
    curr = current_signal_states or {}

    def get_effective_state(key: str) -> str:
        s_cross = signal_states.get(key, "UNKNOWN")
        s_curr = curr.get(key, "UNKNOWN")
        # If either is GREEN, the movement is green-authorized
        if s_cross == "GREEN" or s_curr == "GREEN":
            return "GREEN"
        if s_curr in ("RED", "YELLOW"):
            return s_curr
        return s_cross

    # Auto-detect if single mode from signal_states or matching states
    if signal_states.get("mode") == "single":
        has_separate_right_light = False

    # Check right turn corridor (right lane x >= right_turn_lane_x or rightward motion dx > 10)
    is_in_right_corridor = (vehicle_x is not None and vehicle_x >= right_turn_lane_x) or (dx is not None and dx > 10)

    # ========================================================
    # CASE 1: SINGLE TRAFFIC LIGHT (Không có đèn rẽ phải riêng)
    # ========================================================
    if not has_separate_right_light:
        single_color = get_effective_state("straight_light")
        if single_color == "UNKNOWN":
            single_color = get_effective_state("traffic_light")

        # Lawful Green: Any movement is 100% permitted
        if single_color == "GREEN":
            return False, "traffic_light", "GREEN"

        # Red / Yellow Signal
        if single_color in ("RED", "YELLOW"):
            # Check if right-turn on red is permitted by sign
            if allow_right_turn_on_red and (vehicle_direction == "turn_right" or is_in_right_corridor):
                return False, "traffic_light", f"{single_color} (RẼ PHẢI HỢP LỆ)"
            # All other crossings on red are violations
            return True, "traffic_light", single_color

        return False, "traffic_light", "UNKNOWN"

    # ========================================================
    # CASE 2: MULTI-SIGNAL DIRECTIONAL (2 Cụm đèn riêng biệt)
    # ========================================================
    right_color = get_effective_state("right_turn_light")
    straight_color = get_effective_state("straight_light")

    # 1. Right turn movement or vehicle in right corridor
    if vehicle_direction == "turn_right" or (is_in_right_corridor and right_color == "GREEN"):
        if right_color == "GREEN":
            return False, "right_turn_light", "GREEN"
        if right_color in ("RED", "YELLOW"):
            return True, "right_turn_light", right_color
        # Fallback if right light is unknown: check straight light
        if straight_color == "GREEN":
            return False, "straight_light", "GREEN"
        return (straight_color == "RED"), "straight_light", straight_color

    # 2. Straight movement: ALWAYS evaluate against straight_light
    if vehicle_direction == "straight":
        if straight_color == "GREEN":
            return False, "straight_light", "GREEN"
        # If straight light is red, but vehicle is actually in right lane and right light is green
        if is_in_right_corridor and right_color == "GREEN":
            return False, "right_turn_light", "GREEN"
        return (straight_color == "RED"), "straight_light", straight_color

    # 3. Left turn movement
    if vehicle_direction == "turn_left":
        if straight_color == "GREEN":
            return False, "straight_light", "GREEN"
        # If vehicle was on right lane and right light is green (slight left wiggle during start)
        if is_in_right_corridor and right_color == "GREEN" and (dx is not None and dx > -40):
            return False, "right_turn_light", "GREEN"
        return (straight_color == "RED"), "straight_light", straight_color

    # Fail-safe default
    if straight_color == "GREEN" or right_color == "GREEN":
        return False, "signal", "GREEN"
    return (straight_color == "RED"), "straight_light", straight_color

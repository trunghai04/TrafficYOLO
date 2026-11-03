import pytest
from traffic_yolo.trajectory import compute_movement_vector, classify_vehicle_direction
from traffic_yolo.traffic_light import evaluate_directional_violation


def test_compute_movement_vector():
    history_straight = [(100, 500), (100, 450), (102, 400), (101, 350)]
    dx, dy, angle = compute_movement_vector(history_straight)
    assert abs(dx) <= 5
    assert dy < -100

    history_right = [(100, 500), (120, 480), (160, 460), (220, 450)]
    dx_r, dy_r, angle_r = compute_movement_vector(history_right)
    assert dx_r > 50


def test_classify_vehicle_direction():
    assert classify_vehicle_direction(dx=5, dy=-120) == "straight"
    assert classify_vehicle_direction(dx=60, dy=-50) == "turn_right"
    assert classify_vehicle_direction(dx=-50, dy=-50) == "turn_left"


def test_evaluate_directional_violation():
    # Case 1: Straight lane vehicle (x=200 < 350), straight light is RED -> Violation
    signals = {"straight_light": "RED", "right_turn_light": "GREEN"}
    viol, resp_sig, color = evaluate_directional_violation("straight", signals, vehicle_x=200, right_turn_lane_x=350)
    assert viol is True
    assert resp_sig == "straight_light"
    assert color == "RED"

    # Case 2: Vehicle in right turn corridor (x=500 >= 350), right turn light is GREEN -> Permitted!
    signals = {"straight_light": "RED", "right_turn_light": "GREEN"}
    viol, resp_sig, color = evaluate_directional_violation("straight", signals, vehicle_x=500, right_turn_lane_x=350)
    assert viol is False
    assert resp_sig == "right_turn_light"
    assert color == "GREEN"

    # Case 3: Vehicle turning right (x=600 >= 350), right turn light is GREEN -> Permitted!
    viol, resp_sig, color = evaluate_directional_violation("turn_right", signals, vehicle_x=600, right_turn_lane_x=350)
    assert viol is False

    # Case 4: Vehicle turning right, but right turn light is RED -> Violation!
    signals_red = {"straight_light": "GREEN", "right_turn_light": "RED"}
    viol, resp_sig, color = evaluate_directional_violation("turn_right", signals_red, vehicle_x=600, right_turn_lane_x=350)
    assert viol is True
    assert resp_sig == "right_turn_light"
    assert color == "RED"

    # Case 5: Waiting at red light, then moving on green (Frame 173 scenario)
    # At cross it was RED, but current signal is GREEN when moving -> Strict Permitted!
    signals_at_cross = {"straight_light": "RED", "right_turn_light": "RED"}
    curr_signals = {"straight_light": "GREEN", "right_turn_light": "RED"}
    viol, resp_sig, color = evaluate_directional_violation(
        "straight",
        signals_at_cross,
        current_signal_states=curr_signals,
        vehicle_x=200,
    )
    assert viol is False
    assert color == "GREEN"

    # Case 6: Right turn green arrow while car has minor initial curve (Frame 86 scenario)
    signals_f86 = {"straight_light": "RED", "right_turn_light": "GREEN"}
    viol, resp_sig, color = evaluate_directional_violation(
        "turn_left",  # initial misleading jitter
        signals_f86,
        vehicle_x=830,  # clearly in right turn lane
        dx=-15,
        right_turn_lane_x=750,
    )
    assert viol is False
    assert color == "GREEN"

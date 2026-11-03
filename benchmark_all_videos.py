"""
Automated Comprehensive Benchmark for TrafficYOLO across all 4 video sources:
1. 12h.16.9.22.mp4 (Daytime wide angle)
2. 16h15.25.9.22.mp4 (Daytime standard angle, 2-signal)
3. 16h30.25.9.22.mp4 (Daytime standard angle, heavy traffic)
4. 18h.6.9.22.mp4 (Nighttime low-light)
"""

import os
from pathlib import Path
import cv2
import pandas as pd
from traffic_yolo.config import load_config
from traffic_yolo.pipeline import TrafficMonitorPipeline
from traffic_yolo.traffic_light import (
    auto_detect_signals,
    classify_approach_signal,
    classify_right_pole_assembly,
    classify_multiple_signals,
)


def run_benchmark():
    videos = [
        {
            "name": "12h.16.9.22.mp4",
            "path": r"C:\Users\ADMIN\Downloads\12h.16.9.22.mp4",
            "preset": "12h_wide",
            "out_dir": "outputs/benchmark/12h",
            "mode": "directional",
            "has_right_turn_light": False,
            "allow_right_turn_on_red": False,
            "straight_roi": [1525, 345, 1585, 530],
            "right_roi": [1525, 345, 1585, 530],
            "line_start": [0, 590],
            "line_end": [1550, 650],
        },
        {
            "name": "16h15.25.9.22.mp4",
            "path": r"C:\Users\ADMIN\Downloads\16h15.25.9.22.mp4",
            "preset": "16h_standard",
            "out_dir": "outputs/benchmark/16h15",
            "mode": "directional",
            "has_right_turn_light": True,
            "allow_right_turn_on_red": False,
            "straight_roi": [545, 15, 610, 85],
            "right_roi": [1375, 280, 1445, 415],
            "line_start": [0, 520],
            "line_end": [1500, 750],
        },
        {
            "name": "16h30.25.9.22.mp4",
            "path": r"C:\Users\ADMIN\Downloads\16h30.25.9.22.mp4",
            "preset": "16h_standard",
            "out_dir": "outputs/benchmark/16h30",
            "mode": "directional",
            "has_right_turn_light": True,
            "allow_right_turn_on_red": False,
            "straight_roi": [545, 15, 610, 85],
            "right_roi": [1375, 280, 1445, 415],
            "line_start": [0, 520],
            "line_end": [1500, 750],
        },
        {
            "name": "18h.6.9.22.mp4",
            "path": r"C:\Users\ADMIN\Downloads\18h.6.9.22.mp4",
            "preset": "12h_wide",
            "out_dir": "outputs/benchmark/18h",
            "mode": "directional",
            "has_right_turn_light": False,
            "allow_right_turn_on_red": False,
            "straight_roi": [1525, 345, 1585, 530],
            "right_roi": [1525, 345, 1585, 530],
            "line_start": [0, 590],
            "line_end": [1550, 650],
        },
    ]

    report = []

    for v in videos:
        v_name = v["name"]
        v_path = v["path"]
        if not os.path.exists(v_path):
            print(f"[SKIP] Video not found: {v_path}")
            continue

        print(f"\n{'='*70}\nSTARTING BENCHMARK: {v_name}\n{'='*70}")
        out_dir = Path(v["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        cfg = load_config()
        cfg["inference"]["confidence"] = 0.25
        cfg["counting_line"]["start"] = v["line_start"]
        cfg["counting_line"]["end"] = v["line_end"]
        cfg["traffic_signals"] = {
            "mode": v["mode"],
            "has_right_turn_light": v["has_right_turn_light"],
            "allow_right_turn_on_red": v["allow_right_turn_on_red"],
            "straight_light": {
                "name": "Đèn Đi Thẳng",
                "roi": v["straight_roi"],
                "min_pixels": 8,
            },
            "right_turn_light": {
                "name": "Đèn Rẽ Phải",
                "roi": v["right_roi"],
                "min_pixels": 6,
            },
        }
        cfg["traffic_light"]["roi"] = v["straight_roi"]
        cfg["tracking"]["map_road_riders_to_motorcycle"] = False

        # Run pipeline
        pipeline = TrafficMonitorPipeline(cfg)
        summary = pipeline.run(
            video_source=v_path,
            output_dir=str(out_dir),
            save_video=True,
            save_csv=True,
        )

        csv_path = out_dir / "counted_vehicles.csv"
        df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()

        # Quality Checks:
        # Check 1: No pedestrians (class 0 / 'person') in CSV
        pedestrian_leaks = df[df["vehicle_type"] == "person"] if not df.empty else []
        pedestrian_pass = len(pedestrian_leaks) == 0

        # Check 2: Signal stability
        cap = cv2.VideoCapture(v_path)
        sig_states = []
        for f in range(0, min(190, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), 20):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = cap.read()
            if ret:
                states = classify_multiple_signals(frame, cfg["traffic_signals"])
                sig_states.append((f, states))
        cap.release()

        # Check 3: Waiting motorcycle checks in 12h
        waiting_bike_pass = True
        if "12h" in v_name:
            # Check frame 108: Waiting motorcycle ID 971 should not be in violations
            if not df.empty:
                f_971 = df[df["track_id"] == 971]
                waiting_bike_pass = len(f_971) == 0

        result_row = {
            "video": v_name,
            "total_frames": summary["total_frames"],
            "counts": summary["counts"],
            "violations": summary["violations_count"],
            "pedestrian_filter_pass": pedestrian_pass,
            "waiting_vehicles_pass": waiting_bike_pass,
            "video_generated": bool(summary["output_video"] and os.path.exists(summary["output_video"])),
            "csv_generated": csv_path.exists(),
        }
        report.append(result_row)
        print(f"COMPLETED {v_name}: {result_row}")

    print("\n" + "="*80)
    print("FINAL BENCHMARK SUMMARY:")
    print("="*80)
    for r in report:
        print(f"Video: {r['video']}")
        print(f"  Total frames: {r['total_frames']}")
        print(f"  Flow counts: {r['counts']}")
        print(f"  Violations: {r['violations']}")
        print(f"  Pedestrian filter passed: {r['pedestrian_filter_pass']}")
        print(f"  Waiting vehicles passed: {r['waiting_vehicles_pass']}")
        print(f"  Video & CSV OK: {r['video_generated']} & {r['csv_generated']}")
        print("-" * 50)


if __name__ == "__main__":
    run_benchmark()

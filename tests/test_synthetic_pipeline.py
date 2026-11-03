from pathlib import Path
import cv2
import numpy as np
import pytest
from traffic_yolo.config import load_config
from traffic_yolo.pipeline import TrafficMonitorPipeline


def create_synthetic_video(file_path: Path, num_frames: int = 25, width: int = 640, height: int = 480):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(file_path), fourcc, 10.0, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        # Draw a simulated red traffic light at (50, 50)
        cv2.circle(frame, (50, 50), 15, (0, 0, 255), -1)
        # Draw some background scenery
        cv2.line(frame, (0, 300), (640, 300), (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


def test_traffic_monitor_pipeline_synthetic(tmp_path):
    test_video_path = tmp_path / "synthetic_traffic.mp4"
    create_synthetic_video(test_video_path, num_frames=15, width=640, height=480)

    cfg = load_config()
    cfg["models"]["device"] = "cpu"
    cfg["inference"]["image_size"] = 640
    cfg["traffic_light"]["roi"] = [35, 35, 65, 65]
    cfg["counting_line"]["start"] = [0, 300]
    cfg["counting_line"]["end"] = [640, 300]
    cfg["io"]["video_path"] = str(test_video_path)
    cfg["io"]["output_dir"] = str(tmp_path / "outputs")

    pipeline = TrafficMonitorPipeline(cfg)
    summary = pipeline.run(
        video_source=str(test_video_path),
        output_dir=str(tmp_path / "outputs"),
        save_video=True,
        save_csv=True,
    )

    assert summary["total_frames"] == 15
    assert Path(summary["output_video"]).exists()
    assert Path(summary["output_csv"]).exists()
    assert Path(summary["evidence_dir"]).exists()

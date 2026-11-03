import os
from pathlib import Path
from typing import Any, Dict
import torch
import yaml

DEFAULT_CONFIG_PATH = Path("configs/default_config.yaml")


def get_default_device(preferred_device: str = "auto") -> str:
    """Detect appropriate PyTorch compute device for YOLO."""
    pref = str(preferred_device).lower()
    if pref in ("auto", "cuda"):
        return "0" if torch.cuda.is_available() else "cpu"
    return pref


def deep_merge(base: dict, update: dict) -> dict:
    """Recursively merge two dictionaries."""
    merged = base.copy()
    for key, value in update.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    """Load configuration from YAML file and apply defaults."""
    cfg_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
    else:
        user_config = {}

    # Default fallback values
    default_cfg = {
        "models": {
            "vehicle_detector": "yolov8n.pt",
            "plate_detector": "models/license_plate_detector.pt",
            "tracker_config": "vehicle_bytetrack.yaml",
            "device": "auto",
        },
        "inference": {
            "confidence": 0.30,
            "recovery_confidence": 0.25,
            "image_size": 1280,
        },
        "io": {
            "video_path": "videos/3.mp4",
            "output_dir": "outputs",
            "save_video": True,
            "save_csv": True,
            "csv_filename": "counted_vehicles.csv",
            "video_filename": "counted_video.mp4",
        },
        "counting_line": {
            "start": [0, 925],
            "end": [1400, 750],
            "margin": 5,
            "max_crossing_frame_gap": 5,
        },
        "traffic_light": {
            "roi": [545, 16, 570, 80],
            "min_pixels": 8,
            "hsv": {
                "saturation_min": 100,
                "value_min": 150,
            },
        },
        "tracking": {
            "max_missed_frames": 15,
            "max_violation_missed_frames": 15,
            "max_track_speed": 40,
        },
        "vehicle_classes": {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
        },
    }

    final_config = deep_merge(default_cfg, user_config)
    final_config["models"]["device"] = get_default_device(
        final_config["models"].get("device", "auto")
    )
    return final_config

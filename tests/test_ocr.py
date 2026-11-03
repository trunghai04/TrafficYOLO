import cv2
import numpy as np
import pytest
from traffic_yolo.ocr import format_vietnam_plate, recognize_license_plate_text
from traffic_yolo.evidence import patch_plate_model_compat


def test_format_vietnam_plate_motorcycle():
    # Standard motorcycle 2-line: 90-B2 580.92
    parts = [("90-02}", 0.45), ("580,92", 0.60)]
    plate, conf = format_vietnam_plate(parts, min_conf=0.25)
    assert "90-B2" in plate or "90-D2" in plate
    assert "580.92" in plate
    assert conf > 0.4


def test_format_vietnam_plate_car():
    # Standard car 1-line: 30E-514.31
    parts = [("30E51431", 0.85)]
    plate, conf = format_vietnam_plate(parts, min_conf=0.25)
    assert plate == "30E-514.31"
    assert conf >= 0.8


def test_format_vietnam_plate_reject_noise():
    # Broken noise fragments should be strictly rejected
    assert format_vietnam_plate([("87", 0.2)]) == ("", 0.0)
    assert format_vietnam_plate([("MY", 0.3)]) == ("", 0.0)
    assert format_vietnam_plate([("Eulb", 0.4)]) == ("", 0.0)
    assert format_vietnam_plate([]) == ("", 0.0)


def test_recognize_license_plate_synthetic():
    synth = np.full((120, 160, 3), 255, dtype=np.uint8)
    cv2.putText(synth, "90-B2", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(synth, "520.87", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    plate_str, conf = recognize_license_plate_text(synth, use_gpu=False)
    assert "90-B2" in plate_str
    assert "520.87" in plate_str
    assert conf > 0.5


def test_patch_plate_model_compat_safe_noop():
    # Should safely handle None or mocked objects without raising
    patch_plate_model_compat(None)


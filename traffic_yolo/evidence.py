from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO


def patch_plate_model_compat(plate_model: Optional[YOLO]) -> None:
    """Patch older YOLOv5 PyTorch model inside Ultralytics YOLO to avoid kwarg incompatibilities."""
    if plate_model is None or not hasattr(plate_model, "model") or plate_model.model is None:
        return
    inner = plate_model.model
    if not getattr(inner, "_agy_compat_patched", False):
        import inspect
        if hasattr(inner, "fuse"):
            orig_fuse = inner.fuse
            inner.fuse = lambda *a, **kw: orig_fuse() if not a and not kw else inner
        if hasattr(inner, "forward"):
            orig_forward = inner.forward
            sig = inspect.signature(orig_forward)
            inner.forward = lambda x, *a, **kw: orig_forward(x, *a, **{k: v for k, v in kw.items() if k in sig.parameters})
        inner._agy_compat_patched = True


def enhance_license_plate(plate_crop: np.ndarray, zoom_factor: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Super-resolve and enhance a low-resolution license plate crop.
    
    Returns:
        (enhanced_color_plate, high_contrast_bw_plate)
    """
    h, w = plate_crop.shape[:2]
    # Ensure minimum target size for visibility and OCR detection
    target_w = max(180, w * zoom_factor)
    scale = target_w / max(1, w)
    target_h = max(110, int(h * scale))

    # Step 1: High-fidelity interpolation upscale (Lanczos-4)
    upscaled = cv2.resize(plate_crop, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    # Step 2: Edge-preserving bilateral filter to eliminate compression artifacts
    denoised = cv2.bilateralFilter(upscaled, d=7, sigmaColor=75, sigmaSpace=75)

    # Step 3: Local adaptive contrast enhancement (CLAHE on L-channel of LAB)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
    l_enhanced = clahe.apply(l)
    enhanced_color = cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)

    # Step 4: Unsharp masking to accentuate sharp text edges
    gaussian = cv2.GaussianBlur(enhanced_color, (0, 0), sigmaX=2.0)
    sharpened_color = cv2.addWeighted(enhanced_color, 1.7, gaussian, -0.7, 0)

    # Step 5: High-contrast adaptive binary threshold for crystal-clear readability
    gray = cv2.cvtColor(sharpened_color, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8)
    sharpened_bw = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

    return sharpened_color, sharpened_bw


def detect_license_plate_crop(
    vehicle_crop: np.ndarray,
    plate_model: Optional[YOLO],
    conf: float = 0.20,
    device: str = "cpu",
    zoom_factor: int = 5,
) -> Optional[np.ndarray]:
    """Detect, upscale, and enhance a license plate crop from a vehicle image."""
    if plate_model is None or vehicle_crop is None or vehicle_crop.size == 0:
        return None

    patch_plate_model_compat(plate_model)

    try:
        results = plate_model.predict(
            source=vehicle_crop,
            conf=conf,
            imgsz=640,
            device=device,
            verbose=False,
        )
    except Exception as exc:
        print(f"[WARN] Error during license plate detection: {exc}")
        return None

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None

    boxes = results[0].boxes
    confidences = boxes.conf.cpu().tolist()
    best_index = max(range(len(confidences)), key=lambda i: confidences[i])

    raw_coords = boxes.xyxy[best_index].cpu().tolist()
    b0, b1, b2, b3 = raw_coords[:4]

    h, w = vehicle_crop.shape[:2]

    # For models/license_plate_detector.pt, raw output is [center_x, center_y, width, height]
    # where b2 < b0 or b3 < b1 indicates center/dimensions rather than standard [x1, y1, x2, y2]
    if b2 < b0 or b3 < b1:
        x1 = int(b0 - b2 / 2)
        x2 = int(b0 + b2 / 2)
        y1 = int(b1 - b3 / 2)
        y2 = int(b1 + b3 / 2)
    else:
        x1, y1, x2, y2 = int(b0), int(b1), int(b2), int(b3)

    # Add a slight margin (pad 6%) to prevent cutting outer plate borders
    pad_x = max(1, int((x2 - x1) * 0.06))
    pad_y = max(1, int((y2 - y1) * 0.06))
    x1, y1 = max(0, min(w - 1, x1 - pad_x)), max(0, min(h - 1, y1 - pad_y))
    x2, y2 = max(1, min(w, x2 + pad_x)), max(1, min(h, y2 + pad_y))

    if x2 <= x1 or y2 <= y1:
        return None

    plate_crop = vehicle_crop[y1:y2, x1:x2]
    if plate_crop.size == 0:
        return None

    # Apply enhancement pipeline
    enhanced_color, _ = enhance_license_plate(plate_crop, zoom_factor=zoom_factor)
    return enhanced_color


def create_combined_evidence(
    evidence_frame: np.ndarray,
    plate_zoom: Optional[np.ndarray],
    panel_title: str = "ENHANCED LICENSE PLATE",
    plate_text: Optional[str] = None,
    vehicle_crop: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Combine full violation frame with an enlarged, dual-view enhanced license plate panel.
    Shows the recognized plate text, enhanced color plate, and high-contrast scanned view.
    If tight plate detector missed, falls back to the zoomed vehicle crop so officer can inspect.
    """
    display_target = plate_zoom
    if display_target is None and vehicle_crop is not None and vehicle_crop.size > 0:
        display_target, _ = enhance_license_plate(vehicle_crop, zoom_factor=2)

    if display_target is None:
        return evidence_frame

    height, width = evidence_frame.shape[:2]
    panel_width = max(1, min(520, int(width * 0.42)))
    panel_height = height
    panel = np.full((panel_height, panel_width, 3), (25, 25, 25), dtype=np.uint8)

    # Header Title
    cv2.putText(
        panel,
        panel_title,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # OCR License Plate Banner
    cv2.rectangle(panel, (15, 42), (panel_width - 15, 82), (45, 55, 65), -1)
    if plate_text:
        cv2.rectangle(panel, (15, 42), (panel_width - 15, 82), (0, 255, 255), 1)
        cv2.putText(
            panel,
            f"BKS: {plate_text}",
            (25, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.rectangle(panel, (15, 42), (panel_width - 15, 82), (70, 70, 70), 1)
        cv2.putText(
            panel,
            "BKS: CHUA RO (ANH MO)",
            (25, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )

    # Produce dual-view: Enhanced Color and High-Contrast Scan
    gray = cv2.cvtColor(display_target, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 7)
    bw_bgr = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

    # Fit width
    max_w = max(1, panel_width - 30)
    top_offset = 100
    max_slot_h = max(1, (panel_height - top_offset - 90) // 2)

    def scale_to_slot(image: np.ndarray) -> np.ndarray:
        scale = min(max_w / image.shape[1], max_slot_h / image.shape[0], 1.2)
        nw = max(1, int(image.shape[1] * scale))
        nh = max(1, int(image.shape[0] * scale))
        return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA)

    display_color = scale_to_slot(display_target)
    display_bw = scale_to_slot(bw_bgr)

    # Slot 1: Color Enhanced View
    py1 = top_offset + 25
    cv2.putText(panel, "1. COLOR ENHANCED (HDR)", (18, py1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (180, 220, 255), 1, cv2.LINE_AA)
    px1 = (panel_width - display_color.shape[1]) // 2
    panel[py1:py1 + display_color.shape[0], px1:px1 + display_color.shape[1]] = display_color
    cv2.rectangle(panel, (px1 - 1, py1 - 1), (px1 + display_color.shape[1], py1 + display_color.shape[0]), (80, 80, 80), 1)

    # Slot 2: High-Contrast Scan View
    slot2_top = py1 + display_color.shape[0] + 30
    cv2.putText(panel, "2. HIGH-CONTRAST SCAN (B&W)", (18, slot2_top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (180, 255, 180), 1, cv2.LINE_AA)
    px2 = (panel_width - display_bw.shape[1]) // 2
    if slot2_top + display_bw.shape[0] < panel_height:
        panel[slot2_top:slot2_top + display_bw.shape[0], px2:px2 + display_bw.shape[1]] = display_bw
        cv2.rectangle(panel, (px2 - 1, slot2_top - 1), (px2 + display_bw.shape[1], slot2_top + display_bw.shape[0]), (80, 80, 80), 1)

    # Footer notice
    cv2.putText(panel, "AI SUPER-RESOLUTION APPLIED", (18, panel_height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1, cv2.LINE_AA)

    target_h = max(evidence_frame.shape[0], panel.shape[0])

    def fit_height(image: np.ndarray, target_height: int) -> np.ndarray:
        calculated_w = max(1, int(image.shape[1] * target_height / image.shape[0]))
        return cv2.resize(image, (calculated_w, target_height), interpolation=cv2.INTER_AREA)

    left = fit_height(evidence_frame, target_h)
    right = fit_height(panel, target_h)
    return np.hstack((left, right))

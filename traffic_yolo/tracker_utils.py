from typing import Dict, List, Tuple, Any
import numpy as np
from ultralytics import YOLO
from .geometry import clamp_box

VEHICLE_SIZE_LIMITS = {
    "motorcycle": (0.25, 0.55, 0.08),
    "car": (0.45, 0.70, 0.16),
    "bus": (0.70, 0.85, 0.25),
    "truck": (0.65, 0.85, 0.25),
}


def is_vehicle_box_plausible(
    box: Tuple[int, int, int, int] | list,
    vehicle_type: str,
    frame_width: int,
    frame_height: int,
) -> bool:
    """Check if the vehicle bounding box dimensions are physically realistic."""
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    if box_width <= 0 or box_height <= 0:
        return False

    area_fraction = (box_width * box_height) / (frame_width * frame_height)

    if vehicle_type not in VEHICLE_SIZE_LIMITS:
        return True

    max_width, max_height, max_area = VEHICLE_SIZE_LIMITS[vehicle_type]
    return (
        box_width <= frame_width * max_width
        and box_height <= frame_height * max_height
        and area_fraction <= max_area
    )


def is_track_continuous(
    previous_track: Dict[str, Any],
    current_box: Tuple[int, int, int, int] | list,
    vehicle_type: str,
    frame_number: int,
) -> bool:
    """Validate motion continuity to prevent track ID jumps."""
    if previous_track["vehicle_type"] != vehicle_type:
        return False

    elapsed = max(1, frame_number - previous_track["last_seen"])
    old_box = previous_track["box"]

    predicted_x = (old_box[0] + old_box[2]) / 2 + previous_track["velocity_x"] * elapsed
    predicted_y = (old_box[1] + old_box[3]) / 2 + previous_track["velocity_y"] * elapsed

    current_x = (current_box[0] + current_box[2]) / 2
    current_y = (current_box[1] + current_box[3]) / 2

    old_diagonal = np.hypot(old_box[2] - old_box[0], old_box[3] - old_box[1])
    center_distance = np.hypot(current_x - predicted_x, current_y - predicted_y)
    max_shift = max(35, old_diagonal * 0.35) + 12 * elapsed

    # For fast-moving motorcycles with small bounding boxes, IoU can easily be 0 between frames.
    # Relax IoU constraint for motorcycles while enforcing motion trajectory continuity.
    if vehicle_type == "motorcycle":
        return center_distance <= max_shift * 1.5

    # Calculate Intersection over Union (IoU) for cars, buses, trucks
    ix1 = max(old_box[0], current_box[0])
    iy1 = max(old_box[1], current_box[1])
    ix2 = min(old_box[2], current_box[2])
    iy2 = min(old_box[3], current_box[3])

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    old_area = (old_box[2] - old_box[0]) * (old_box[3] - old_box[1])
    current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
    union = old_area + current_area - intersection
    iou = intersection / union if union > 0 else 0

    return center_distance <= max_shift and iou >= 0.04


def recover_oversized_vehicle_boxes(
    model: YOLO,
    frame: np.ndarray,
    boxes: List[List[float]],
    class_ids: List[int],
    width: int,
    height: int,
    vehicle_classes: Dict[int, str],
    recovery_confidence: float = 0.25,
    image_size: int = 1280,
    device: str = "cpu",
) -> List[List[float]]:
    """Recover oversized or distorted tracked boxes using a fresh raw prediction."""
    needs_recovery = any(
        class_id in vehicle_classes
        and not is_vehicle_box_plausible(
            clamp_box(box, width, height),
            vehicle_classes[class_id],
            width,
            height,
        )
        for box, class_id in zip(boxes, class_ids)
    )

    if not needs_recovery:
        return boxes

    result = model.predict(
        frame,
        conf=recovery_confidence,
        imgsz=image_size,
        device=device,
        classes=list(vehicle_classes.keys()),
        verbose=False,
    )[0]

    if result.boxes is None or len(result.boxes) == 0:
        return boxes

    raw_boxes = result.boxes.xyxy.cpu().tolist()
    raw_classes = result.boxes.cls.int().cpu().tolist()
    original_boxes = [list(box) for box in boxes]
    used_raw_indices = set()

    for index, (tracked_box, class_id) in enumerate(zip(original_boxes, class_ids)):
        if class_id not in vehicle_classes:
            continue
        clamped_box = clamp_box(tracked_box, width, height)
        if is_vehicle_box_plausible(clamped_box, vehicle_classes[class_id], width, height):
            continue

        tx = (clamped_box[0] + clamped_box[2]) / 2
        ty = (clamped_box[1] + clamped_box[3]) / 2
        candidates = []

        for raw_index, (raw_box, raw_class) in enumerate(zip(raw_boxes, raw_classes)):
            if raw_class != class_id or raw_index in used_raw_indices:
                continue
            candidate = clamp_box(raw_box, width, height)
            if not is_vehicle_box_plausible(candidate, vehicle_classes[raw_class], width, height):
                continue

            cx = (candidate[0] + candidate[2]) / 2
            cy = (candidate[1] + candidate[3]) / 2
            if not (clamped_box[0] <= cx <= clamped_box[2] and clamped_box[1] <= cy <= clamped_box[3]):
                continue

            # Check overlap with other original valid boxes
            overlaps = False
            for other_index, (other_box, other_class) in enumerate(zip(original_boxes, class_ids)):
                if other_index == index or other_class not in vehicle_classes:
                    continue
                other_box = clamp_box(other_box, width, height)
                if not is_vehicle_box_plausible(other_box, vehicle_classes[other_class], width, height):
                    continue
                ix1, iy1 = max(candidate[0], other_box[0]), max(candidate[1], other_box[1])
                ix2, iy2 = min(candidate[2], other_box[2]), min(candidate[3], other_box[3])
                intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                cand_area = (candidate[2] - candidate[0]) * (candidate[3] - candidate[1])
                oth_area = (other_box[2] - other_box[0]) * (other_box[3] - other_box[1])
                union = cand_area + oth_area - intersection
                if union > 0 and intersection / union > 0.6:
                    overlaps = True
                    break
            if overlaps:
                continue

            candidates.append((np.hypot(cx - tx, cy - ty), raw_index, raw_box))

        if candidates:
            _, raw_index, replacement_box = min(candidates)
            boxes[index] = replacement_box
            used_raw_indices.add(raw_index)

    return boxes

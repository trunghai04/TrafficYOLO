import re
from typing import List, Optional, Tuple
import cv2
import numpy as np

_OCR_READER = None

# Real Vietnam province codes (Mã tỉnh thành Việt Nam theo Thông tư 24/2023/TT-BCA)
VALID_PROVINCES = {
    11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 43, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
    79, 81, 82, 83, 84, 85, 86, 88, 89, 90, 92, 93, 94, 95, 97, 98, 99
}

CHAR_TO_DIGIT = {
    'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'l': '1', '|': '1',
    'Z': '2', 'J': '3', 'A': '4', 'S': '5', 'G': '6', 'b': '6',
    'T': '7', 'B': '8', 'g': '9', 'q': '9'
}

CHAR_TO_LETTER = {
    '0': 'B', '8': 'B', '1': 'T', '5': 'S', '2': 'Z', '6': 'G', '4': 'A', '3': 'E'
}


def get_ocr_reader(use_gpu: bool = False):
    """Singleton getter for EasyOCR reader to avoid reloading models."""
    global _OCR_READER
    if _OCR_READER is None:
        try:
            import easyocr
            import torch
            gpu = use_gpu and torch.cuda.is_available()
            print(f"[INFO] Initializing EasyOCR Engine (GPU={gpu})...")
            _OCR_READER = easyocr.Reader(["en"], gpu=gpu, verbose=False)
        except Exception as e:
            print(f"[WARN] Failed to load EasyOCR: {e}")
            _OCR_READER = False
    return _OCR_READER if _OCR_READER is not False else None


def format_vietnam_plate(raw_parts: List[Tuple[str, float]], min_conf: float = 0.15) -> Tuple[str, float]:
    """
    Format and validate raw OCR tokens into official Vietnamese license plate syntax.
    Handles 2-line motorcycle plates (e.g. '90-B2 520.87') and 1-line car plates (e.g. '30E-514.31').
    """
    if not raw_parts:
        return "", 0.0

    tokens = []
    total_conf = 0.0
    for text, conf in raw_parts:
        if conf < min_conf:
            continue
        cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
        if cleaned:
            tokens.append(cleaned)
            total_conf += conf

    if not tokens:
        return "", 0.0

    avg_conf = total_conf / len(tokens)
    full_str = "".join(tokens)
    if len(full_str) < 5:
        return "", 0.0

    # Case A: Two-line layout (Motorcycles & square car plates)
    if len(tokens) >= 2:
        line1_raw = tokens[0]
        line2_raw = "".join(tokens[1:])

        if len(line1_raw) >= 3:
            p0 = CHAR_TO_DIGIT.get(line1_raw[0], line1_raw[0])
            p1 = CHAR_TO_DIGIT.get(line1_raw[1], line1_raw[1])
            if p0.isdigit() and p1.isdigit():
                prov = int(p0 + p1)
                if prov in VALID_PROVINCES:
                    ser_char = line1_raw[2]
                    if ser_char.isdigit():
                        ser_char = CHAR_TO_LETTER.get(ser_char, "B")
                    sub_num = ""
                    if len(line1_raw) >= 4:
                        sub_num = CHAR_TO_DIGIT.get(line1_raw[3], line1_raw[3])
                    line1_fmt = f"{p0}{p1}-{ser_char}{sub_num}"

                    # Line 2: 4 or 5 digits
                    digits_only = "".join(CHAR_TO_DIGIT.get(c, c) for c in line2_raw if CHAR_TO_DIGIT.get(c, c).isdigit())
                    if len(digits_only) == 5:
                        line2_fmt = f"{digits_only[:3]}.{digits_only[3:]}"
                        return f"{line1_fmt} {line2_fmt}", avg_conf
                    elif len(digits_only) == 4:
                        return f"{line1_fmt} {digits_only}", avg_conf
                    elif len(digits_only) > 5:
                        digits_only = digits_only[-5:]
                        return f"{line1_fmt} {digits_only[:3]}.{digits_only[3:]}", avg_conf
                    elif len(digits_only) >= 3:
                        return f"{line1_fmt} {digits_only}", avg_conf

    # Case B: Single line layout (Cars / Rectangular plates)
    if len(full_str) >= 6:
        p0 = CHAR_TO_DIGIT.get(full_str[0], full_str[0])
        p1 = CHAR_TO_DIGIT.get(full_str[1], full_str[1])
        if p0.isdigit() and p1.isdigit() and int(p0 + p1) in VALID_PROVINCES:
            ser_char = full_str[2]
            if ser_char.isdigit():
                ser_char = CHAR_TO_LETTER.get(ser_char, "A")
            rem_digits = "".join(CHAR_TO_DIGIT.get(c, c) for c in full_str[3:] if CHAR_TO_DIGIT.get(c, c).isdigit())
            if len(rem_digits) == 5:
                return f"{p0}{p1}{ser_char}-{rem_digits[:3]}.{rem_digits[3:]}", avg_conf
            elif len(rem_digits) == 4:
                return f"{p0}{p1}{ser_char}-{rem_digits}", avg_conf
            elif len(rem_digits) >= 3:
                return f"{p0}{p1}{ser_char}-{rem_digits}", avg_conf

    return "", 0.0


def recognize_license_plate_text(plate_img: np.ndarray, use_gpu: bool = False) -> Tuple[str, float]:
    """
    Run multi-pass OCR on an enhanced license plate crop and return verified Vietnamese plate string.
    
    Returns:
        (plate_text, confidence)
    """
    reader = get_ocr_reader(use_gpu=use_gpu)
    if reader is None or plate_img is None or plate_img.size == 0:
        return "", 0.0

    try:
        h, w = plate_img.shape[:2]

        # Pass 1: Full image recognition with tuned CRAFT thresholds
        results = reader.readtext(
            plate_img,
            detail=1,
            paragraph=False,
            text_threshold=0.20,
            low_text=0.15,
            allowlist="0123456789ABCDEFGHKLMNPSTUVXYZ-.",
        )
        if results:
            results_sorted = sorted(results, key=lambda item: (item[0][0][1], item[0][0][0]))
            raw_parts = [(text, float(conf)) for _, text, conf in results_sorted]
            plate_str, conf = format_vietnam_plate(raw_parts, min_conf=0.15)
            if plate_str:
                return plate_str, conf

        # Pass 2: Row-splitting for square / motorcycle plates (aspect ratio <= 1.6)
        if w / max(1, h) <= 1.6:
            top_half = plate_img[0:int(h * 0.55), :]
            bot_half = plate_img[int(h * 0.45):h, :]

            # Pad each row with white borders to allow CRAFT detector to capture edge characters
            pad_top = cv2.copyMakeBorder(top_half, 20, 20, 30, 30, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            pad_bot = cv2.copyMakeBorder(bot_half, 20, 20, 30, 30, cv2.BORDER_CONSTANT, value=[255, 255, 255])

            res_top = reader.readtext(
                pad_top,
                detail=1,
                paragraph=False,
                text_threshold=0.18,
                low_text=0.12,
                allowlist="0123456789ABCDEFGHKLMNPSTUVXYZ-.",
            )
            res_bot = reader.readtext(
                pad_bot,
                detail=1,
                paragraph=False,
                text_threshold=0.18,
                low_text=0.12,
                allowlist="0123456789ABCDEFGHKLMNPSTUVXYZ-.",
            )

            combined_parts = []
            if res_top:
                combined_parts.append((res_top[0][1], float(res_top[0][2])))
            if res_bot:
                combined_parts.append((res_bot[0][1], float(res_bot[0][2])))

            if combined_parts:
                plate_str, conf = format_vietnam_plate(combined_parts, min_conf=0.12)
                if plate_str:
                    return plate_str, conf

        # Pass 3: High-contrast adaptive thresholded version
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if len(plate_img.shape) == 3 else plate_img
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8)
        bw_bgr = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

        res_bw = reader.readtext(
            bw_bgr,
            detail=1,
            paragraph=False,
            text_threshold=0.20,
            low_text=0.15,
            allowlist="0123456789ABCDEFGHKLMNPSTUVXYZ-.",
        )
        if res_bw:
            res_bw_sorted = sorted(res_bw, key=lambda item: (item[0][0][1], item[0][0][0]))
            raw_parts = [(text, float(conf)) for _, text, conf in res_bw_sorted]
            plate_str, conf = format_vietnam_plate(raw_parts, min_conf=0.15)
            if plate_str:
                return plate_str, conf

        return "", 0.0

    except Exception as e:
        print(f"[WARN] OCR recognition error: {e}")
        return "", 0.0

"""Image quality checks: blur, brightness, contrast, face size.

Pure OpenCV/NumPy, well under 10 ms per frame, since the heavy metrics run on
the (small) face crop instead of the whole image.

assess_quality(image_bgr, bbox) is the one entry point used by the pipeline
for both photos and webcam frames.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from . import config


def _crop_bbox(image_bgr: np.ndarray, bbox: Optional[list]) -> Tuple[np.ndarray, list, list]:
    """Return (face_crop, bbox, [w, h] of the full image).

        bbox is [x1, y1, x2, y2] inside the image, or None for the whole frame.
    """
    im_h, im_w = image_bgr.shape[:2]
    if bbox is None:
        bbox = [0, 0, im_w, im_h]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    # Clamp to image bounds; guard against degenerate boxes.
    x2, y2 = max(x2, x1 + 1), max(y2, y1 + 1)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(im_w, x2), min(im_h, y2)
    return image_bgr[y1:y2, x1:x2], [x1, y1, x2, y2], [im_w, im_h]


def _linear_score(value: float, low: float, high: float) -> float:
    """Map value in [low, high] linearly to [0, 1] (clamped)."""
    if high <= low:
        return 1.0
    return float(min(1.0, max(0.0, (value - low) / (high - low))))


def check_blur(gray: np.ndarray) -> Tuple[float, float, bool]:
    """Return (raw_voL, normalized_score, ok) for sharpness.

        voL = variance of the Laplacian; higher = sharper. Normalized score runs
        BLUR_MIN -> 0 and BLUR_FULL -> 1 (src/config.py).
        """
    vol = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    score = _linear_score(vol, config.BLUR_MIN, config.BLUR_FULL)
    return vol, score, vol >= config.BLUR_MIN


def check_brightness(gray: np.ndarray) -> Tuple[float, float, bool]:
    """Return (mean_gray, normalized_score, ok) for exposure.

    Inside [BRIGHTNESS_LOW, BRIGHTNESS_HIGH] -> score 1.0. Outside, the score
    falls off linearly toward the nearest edge (0 or 255).
    """
    mean = float(np.mean(gray))
    ok = config.BRIGHTNESS_LOW <= mean <= config.BRIGHTNESS_HIGH
    if mean < config.BRIGHTNESS_LOW:
        # Too dark: 0 at black, 1.0 at BRIGHTNESS_LOW.
        score = 1.0 if config.BRIGHTNESS_LOW <= 0 else mean / config.BRIGHTNESS_LOW
    elif mean > config.BRIGHTNESS_HIGH:
        # Too bright: 1.0 at BRIGHTNESS_HIGH, 0 at pure white.
        top_gap = max(1.0, 255.0 - config.BRIGHTNESS_HIGH)
        score = max(0.0, (255.0 - mean) / top_gap)
    else:
        score = 1.0
    return mean, float(min(1.0, score)), bool(ok)


def check_contrast(gray: np.ndarray) -> Tuple[float, float, bool]:
    """Return (std_gray, normalized_score, ok) based on gray-level spread."""
    std = float(np.std(gray))
    score = _linear_score(std, config.CONTRAST_MIN, config.CONTRAST_FULL)
    return std, score, std >= config.CONTRAST_MIN


def check_face_size(face_height: int) -> Tuple[float, bool]:
    """Return (normalized_score, ok) based on face bbox height in pixels.

    Face-size ratio to the frame is reported by assess_quality; scoring uses
    MIN_FACE_PX -> 0 and FULL_FACE_PX -> 1 (src/config.py).
    """
    score = _linear_score(float(face_height), float(config.MIN_FACE_PX), float(config.FULL_FACE_PX))
    return score, face_height >= config.MIN_FACE_PX


def assess_quality(image_bgr: np.ndarray, bbox: Optional[list] = None) -> Dict[str, Any]:
    """Assess quality of the (cropped) face region of one frame.

    Args:
        image_bgr: full BGR image (photo frame or single webcam frame).
        bbox: face box [x1, y1, x2, y2] in image coords; None = whole image.

    Returns:
        {
            "blur_score":   raw VoL (float),          # used for display
            "brightness":   raw mean gray (float),
            "contrast":     raw std gray (float),
            "face_width":   int px,
            "face_height":  int px,
            "face_ratio":   face_height / image_height (float),
            "scores":       {"blur": f, "brightness": f, "contrast": f,
                             "face_size": f, "overall": f}  # each in [0, 1]
            "issues":       ["Image is blurry", ...]  # human-readable fixes
        }
    """
    crop, _, im_size = _crop_bbox(image_bgr, bbox)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    vol, blur_score, blur_ok = check_blur(gray)
    mean, bright_score, bright_ok = check_brightness(gray)
    std, contrast_score, contrast_ok = check_contrast(gray)

    face_width, face_height = im_size if bbox is None else (bbox[2] - bbox[0], bbox[3] - bbox[1])
    size_score, size_ok = check_face_size(int(face_height))
    ratio = float(face_height) / float(im_size[1])

    # Human-readable issues, in priority order of typical fixes.
    issues: list[str] = []
    if not blur_ok:
        issues.append("Image is blurry")
    if not bright_ok and mean < config.BRIGHTNESS_LOW:
        issues.append("Too dark")
    if not bright_ok and mean > config.BRIGHTNESS_HIGH:
        issues.append("Too bright/overexposed")
    if not contrast_ok:
        issues.append("Low contrast")
    if not size_ok:
        issues.append("Face too small / too far from camera")

    overall = (blur_score + bright_score + contrast_score + size_score) / 4.0

    return {
        "blur_score": vol,
        "brightness": mean,
        "contrast": std,
        "face_width": int(face_width),
        "face_height": int(face_height),
        "face_ratio": ratio,
        "scores": {
            "blur": blur_score,
            "brightness": bright_score,
            "contrast": contrast_score,
            "face_size": size_score,
            "overall": overall,
        },
        "issues": issues,
    }

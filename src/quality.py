"""Image quality checks: blur, brightness, contrast, face size.

Each check returns a score in [0, 1]; 1 = ideal quality.
"""

from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

from . import config


def check_blur(gray: np.ndarray) -> Tuple[float, bool]:
    """Return (score, ok) for sharpness.

    Uses the variance of the Laplacian: higher = sharper.
    """


def check_brightness(gray: np.ndarray) -> Tuple[float, bool]:
    """Return (score, ok) for mean brightness (not too dark / not too bright)."""


def check_contrast(gray: np.ndarray) -> Tuple[float, bool]:
    """Return (score, ok) based on the standard deviation of pixel intensities."""


def check_face_size(face_height_px: int) -> Tuple[float, bool]:
    """Return (score, ok) based on detected face height in pixels."""


def analyze_quality(frame_bgr: np.ndarray, face_height_px: int) -> Tuple[float, Dict[str, float]]:
    """Run all quality checks on one frame.

    Args:
        frame_bgr: full BGR image.
        face_height_px: height of the detected face bbox.

    Returns:
        (overall_quality_score in [0,1], per-check details dict).
    """

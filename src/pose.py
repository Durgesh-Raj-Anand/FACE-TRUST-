"""Head pose estimation (yaw / pitch / roll) from face landmarks.

Estimated from a 3D-2D PnP solve using standard facial landmark points
(nose, eyes, chin provided by InsightFace's 5-point kps).
"""

from typing import Dict, Tuple

import numpy as np


def estimate_pose(kps: np.ndarray) -> Dict[str, float]:
    """Estimate yaw / pitch / roll in degrees from 5 face keypoints.

    Args:
        kps: array of shape (5, 2) with (x, y) for
             [left_eye, right_eye, nose, left_mouth, right_mouth].

    Returns:
        {"yaw": float, "pitch": float, "roll": float}.
    """

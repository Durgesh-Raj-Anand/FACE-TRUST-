"""Single-frame pipeline shared by photos and live video.

analyze_frame() is THE core function both modes call; live mode only adds
temporal smoothing (moving average over the last SMOOTHING_WINDOW results,
with full analysis only every PROCESS_EVERY_N_FRAMES-th frame).
"""

from typing import Any, Dict, Optional

import numpy as np


class LiveSmoother:
    """Moving-average smoother for live Trust Scores."""

    def __init__(self, window: int) -> None:
        """Store the last `window` (score, decision) results."""

    def update(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Add a result and return the smoothed result dict."""


def analyze_frame(frame_bgr: np.ndarray, recognizer: Any) -> Dict[str, Any]:
    """Full single-frame analysis: detect -> identify -> quality -> pose -> Trust Score.

    Args:
        frame_bgr: one BGR image (from a file or a webcam frame).
        recognizer: initialized Recognizer with a loaded gallery.

    Returns:
        Dict with keys: face_found, person, similarity, quality_score,
        quality_details, pose (yaw/pitch/roll), trust_score, decision, message.
    """

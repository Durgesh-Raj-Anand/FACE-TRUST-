"""Single-frame pipeline shared by photos and live video - THE one analysis path.

analyze_frame() chains recognizer -> quality -> pose -> trust for one BGR
image (uploaded photo, webcam snapshot or live frame).

LiveSession adds ONLY the live-mode extras required by the project spec:
  - full analysis every PROCESS_EVERY_N_FRAMES frames, last result reused in
    between,
  - moving average over the last SMOOTHING_WINDOW Trust Scores,
  - identity picked by majority vote over the same window,
  - decision recomputed from the smoothed score,
  - measured processing FPS (heavy-analysis rate).
"""

from __future__ import annotations

import time
from collections import Counter, deque
from typing import Any, Callable, Dict, Optional

import numpy as np

from . import config
from .pose import assess_pose
from .quality import assess_quality
from .recognizer import FaceRecognizer, GALLERY_PATH
from .trust import compute_trust, decide, identity_of

_shared_recognizer: Optional[FaceRecognizer] = None


def get_default_recognizer() -> FaceRecognizer:
    """ lazily created shared FaceRecognizer for the app and the live session. """
    global _shared_recognizer
    if _shared_recognizer is None:
        _shared_recognizer = FaceRecognizer()
        if GALLERY_PATH.exists():
            _shared_recognizer.load_gallery()
    return _shared_recognizer


def analyze_frame(
    image_bgr: np.ndarray,
    recognizer: Optional[FaceRecognizer] = None,
    mode: str = "photo",
) -> Dict[str, Any]:
    """Full single-frame analysis - the same function for photos AND video.

        Args:
            image_bgr: one BGR image (file, snapshot or webcam frame).
            recognizer: a FaceRecognizer (defaults to a shared lazy instance).
            mode: "photo" or "live" (only changes the advice wording).

        Returns:
            {"face_found", "identity", "similarity", "det_score", "bbox",
             "recognition", "quality", "pose", "trust"} where "trust" holds
             trust_score, decision, explanation and the user-facing message.
        """
    rec = recognizer if recognizer is not None else get_default_recognizer()

    recognition = rec.identify(image_bgr)
    if not recognition["face_found"]:
        # No face: skip quality/pose and fail fast with the hard-rule REJECT.
        trust = compute_trust({"face_found": False}, {"scores": {}, "issues": []},
                              {"issues": []}, mode=mode)
        return {
            "face_found": False, "identity": "Unknown", "similarity": None,
            "det_score": None, "bbox": None,
            "recognition": recognition,
            "quality": {"scores": {}, "issues": ["No face detected"]},
            "pose": {"issues": []},
            "trust": trust,
        }

    bbox = recognition["bbox"]
    quality = assess_quality(image_bgr, bbox)
    pose = assess_pose(recognition["insightface_pose"], kps=recognition["landmarks"])
    trust = compute_trust(recognition, quality, pose, mode=mode)

    return {
        "face_found": True,
        "identity": identity_of(recognition),
        "similarity": recognition["similarity"],
        "det_score": recognition["det_score"],
        "bbox": bbox,
        "recognition": recognition,
        "quality": quality,
        "pose": pose,
        "trust": trust,
    }


class LiveSession:
    """Video wrapper around analyze_frame with smoothing; photos never use it."""

    def __init__(
        self,
        recognizer: Optional[FaceRecognizer] = None,
        analyzer: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> None:
        """`analyzer` can be injected for tests; default is the real pipeline."""
        self.analyzer = analyzer if analyzer is not None else analyze_frame
        self.recognizer = recognizer
        self.process_every = config.PROCESS_EVERY_N_FRAMES
        self.window: int = config.SMOOTHING_WINDOW
        self._frame_idx = 0
        self._last_raw: Dict[str, Any] = {"trust": {"trust_score": 0.0, "decision": "REJECT",
                                                    "explanation": ["No analysis yet"], "message": ""}}
        self._scores: deque = deque(maxlen=self.window)   # trust scores of the window
        self._votes: deque = deque(maxlen=self.window)    # identities of the window
        self._last_heavy: float | None = None             # time of last full analysis
        self.fps: float = 0.0                             # measured processing FPS

    def process(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """Analyze one video frame and return the smoothed result."""
        if self._frame_idx % self.process_every == 0:
            # Full analysis on the Nth frame, with FPS measured between runs.
            now = time.perf_counter()
            if self._last_heavy is not None:
                self.fps = self.process_every / (now - self._last_heavy)
            self._last_heavy = now

            self._last_raw = self.analyzer(frame_bgr, self.recognizer, mode="live")
            self._scores.append(float(self._last_raw["trust"]["trust_score"]))
            self._votes.append(self._last_raw["identity"])

        self._frame_idx += 1

        # Smoothed Trust Score + majority-vote identity over the window.
        smoothed = float(np.mean(self._scores)) if self._scores else 0.0
        identity = Counter(self._votes).most_common(1)[0][0] if self._votes else "Unknown"
        raw = self._last_raw["trust"]
        return {
            "identity": identity,
            "trust_score": round(smoothed, 1),
            "decision": decide(smoothed),                    # recomputed, not copied
            "explanation": raw.get("explanation", []),
            "message": raw.get("message", ""),
            "bbox": self._last_raw.get("bbox"),
            "raw": self._last_raw,                           # last full result
            "fps": self.fps,
        }

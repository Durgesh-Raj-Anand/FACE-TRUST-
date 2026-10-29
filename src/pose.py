"""Head pose: yaw / pitch / roll in degrees -> pose sub-score + issues.

What the three angles mean (camera in front of the face):
    yaw    = rotation around the horizontal axis (turning the face LEFT/RIGHT,
             nose pointing to the person's left or right shoulder)
    pitch  = rotation around the horizontal left-right axis through the head
             (nodding: chin DOWN = positive here, looking UP = negative)
    roll   = in-plane rotation of the face in the image itself (tilting the
             head sideways, like resting the cheek on the shoulder); roll
             barely changes identity, so it is weighted lowest in the score.

Conventions used by this module (all fixed in IMAGE space of an UN-mirrored
frame - what the camera sensor delivers, before any UI flip):
    yaw > 0   -> the nose points toward the RIGHT side of the image
                 (i.e. the person turned toward their own right shoulder)
    pitch > 0 -> chin lowered toward the chest ("looking down")
    roll  > 0 -> the head is tilted clockwise (the person's right eye lower)

Magnitudes come from InsightFace's face.pose (order = [pitch, yaw, roll],
degrees - verified in insightface/model_zoo/landmark.py). To make LEFT/RIGHT
and UP/DOWN advice reliable, the SIGNS are re-derived from the 5 keypoints in
image coordinates (nose vs. eye midpoint x for yaw; nose position between the
eye line and the mouth line for pitch; eye-line angle for roll). This makes
the signs correct by construction for un-mirrored frames.

NOTE on mirroring: if the UI previews the webcam mirrored (frame flipped
horizontally), the visual left/right of the overlay swaps even though the
numbers above stay in sensor space. The photo tabs feed un-mirrored frames,
so no swap happens there.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from . import config


def _soft_score(angle: float, limit: float) -> float:
    """Score in [0, 1]: 1.0 inside the soft zone, linear decay to 0 at `limit`."""
    soft = config.POSE_SOFT_FACTOR * limit
    a = abs(angle)
    if a <= soft:
        return 1.0
    if a >= limit:
        return 0.0
    return float((limit - a) / (limit - soft))


def _keypoint_sides(kps: np.ndarray) -> Dict[str, float]:
    """Derive yaw/pitch/roll SIGNS in image space from the 5 keypoints.

        kps rows (InsightFace 5-point order): two eyes, nose, two mouth corners.
        The left/right of each pair is found from x values, so no assumption
        about eye ordering is needed.

        Returns {"yaw_sign", "pitch_sign", "roll"} - roll is the full angle.
    """
    pts = np.asarray(kps, dtype=float)
    eye_l, eye_r, nose = pts[0], pts[1], pts[2]
    if eye_l[0] > eye_r[0]:  # normalize: eye_l = smaller x (image left)
        eye_l, eye_r = eye_r, eye_l
    mouth_l, mouth_r = sorted((pts[3], pts[4]), key=lambda p: p[0])
    eye_mid = (eye_l + eye_r) / 2.0
    mouth_mid = (mouth_l + mouth_r) / 2.0
    face_w = float(np.linalg.norm(eye_r - eye_l))

    # Yaw: horizontal nose offset from the eye midpoint.
    yaw_sign = 0.0
    if face_w > 1.0:  # avoid div-by-zero on degenerate boxes
        dx = float(nose[0] - eye_mid[0]) / face_w
        yaw_sign = float(np.sign(dx)) if abs(dx) > 0.02 else 0.0  # 2% dead zone

    # Pitch (chin-down > 0): nose sits between the eye line and mouth line.
    eye_to_nose = float(nose[1] - eye_mid[1])
    nose_to_mouth = float(mouth_mid[1] - nose[1])
    pitch_sign = float(np.sign(eye_to_nose - nose_to_mouth)) if np.sign(eye_to_nose - nose_to_mouth) != 0 else 0.0

    # Roll: angle of the eye line; right eye lower (dy > 0) = clockwise = +.
    roll = float(np.degrees(np.arctan2(eye_r[1] - eye_l[1], eye_r[0] - eye_l[0])))
    return {"yaw_sign": yaw_sign, "pitch_sign": pitch_sign, "roll": roll}


def _yaw_text(yaw: float) -> str:
    """Direction wording for yaw; mirrored displays swap it visually only."""
    return "Face turned too far right - face the camera" if yaw > 0 else \
           "Face turned too far left - face the camera"


def _pitch_text(pitch: float) -> str:
    """Direction wording for pitch."""
    return "Head tilted down - look straight into the camera" if pitch > 0 else \
           "Head tilted up - look straight into the camera"


def assess_pose(pose: Any, kps: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Compute yaw/pitch/roll, a pose sub-score and human-readable issues.

        Args:
            pose: raw InsightFace face.pose ([pitch, yaw, roll], degrees), or a
                  dict {"yaw": f, "pitch": f, "roll": f}. If the dict already
                  carries signed values, kps correction is skipped.
            kps:  5x2 keypoints (optional) to re-derive signs in image space.

        Returns:
            {"yaw", "pitch", "roll": signed degrees, "pose_score": [0, 1],
             "issues": [...]}.
        """
    issues: list[str] = []

    # Normalize input into signed (yaw, pitch, roll).
    if isinstance(pose, dict):
        yaw, pitch, roll = float(pose.get("yaw", 0.0)), float(pose.get("pitch", 0.0)), float(pose.get("roll", 0.0))
    else:
        p = np.asarray(pose, dtype=float).reshape(-1)
        pitch = float(p[0])                       # rx = pitch
        yaw, roll = (float(p[1]), float(p[2])) if p.size >= 3 else (0.0, 0.0)

    # InsightFace gives magnitudes reliably, but its left/right sign is model-
    # space; re-anchor sign in image space from keypoints where possible.
    if kps is not None:
        sides = _keypoint_sides(kps)
        yaw = sides["yaw_sign"] * abs(yaw) if sides["yaw_sign"] else yaw
        pitch = sides["pitch_sign"] * abs(pitch) if sides["pitch_sign"] else pitch
        roll = sides["roll"]  # kps roll is exact image-space geometry

    yaw_score = _soft_score(yaw, config.YAW_MAX)
    pitch_score = _soft_score(pitch, config.PITCH_MAX)
    roll_score = _soft_score(roll, config.ROLL_MAX)
    w_yaw, w_pitch, w_roll = config.POSE_SCORE_WEIGHTS
    pose_score = w_yaw * yaw_score + w_pitch * pitch_score + w_roll * roll_score

    if abs(yaw) > config.YAW_MAX:
        issues.append(_yaw_text(yaw))
    if abs(pitch) > config.PITCH_MAX:
        issues.append(_pitch_text(pitch))
    if abs(roll) > config.ROLL_MAX:
        issues.append("Head tilted sideways - straighten your head")

    return {"yaw": yaw, "pitch": pitch, "roll": roll, "pose_score": pose_score, "issues": issues}


def draw_pose(image: np.ndarray, pose: Dict[str, Any], bbox: Optional[list] = None) -> np.ndarray:
    """Overlay yaw/pitch/roll and decay arrows on one image (UI helper).

        Shows "Y: yaw P: pitch R: roll degrees" above the bbox and a small 2D
        arrow pointing where the nose points ((yaw, pitch) direction). The arrow
        x-direction matches an un-mirrored frame; if you mirror the frame for
        preview, either draw before flipping or mirror the arrow x.
        """
    image = image.copy()
    pitch, yaw = pose["pitch"], pose["yaw"]

    origin = (10, 12)
    if bbox is not None:
        origin = (int(min(max(bbox[0], 0), image.shape[1] - 1)), int(bbox[1] - 6 if bbox[1] >= 6 else 12))
    cv2.putText(image, f"Y:{yaw:+.1f} P:{pitch:+.1f} R:{pose['roll']:+.1f}", origin,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 200, 255), 2)

    if bbox is not None:
        cx, cy = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)
        side = int(max(bbox[3] - bbox[1], 1) * 0.18)
        # Arrow points where the nose does; x flip is manual for mirrored views.
        tip = (int(cx + (yaw / config.YAW_MAX) * side), int(cy + (pitch / config.PITCH_MAX) * side))
        cv2.line(image, (cx, cy), tip, (255, 120, 20), 2)
        cv2.circle(image, tip, 3, (255, 120, 20), -1)
    return image

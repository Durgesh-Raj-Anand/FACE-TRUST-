"""Calibrate quality thresholds from src/config.py.

Modes:
  1. --image path.jpg : prints blur/brightness/contrast/face-size metrics for
     the face crop of the photo AND for deliberately degraded copies
     (blurred, darker, brighter, lower contrast) so you can see how each
     metric reacts and choose thresholds.
  2. --webcam : live window showing the raw blur/brightness/contrast numbers
     of your face crop. Watch the values change (move closer for face size,
     add light for brightness, shake your head for blur) and tune src/config.py.

Usage:
    python scripts/calibrate_quality.py --image data/test/photo.jpg
    python scripts/calibrate_quality.py --webcam
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow `python scripts/calibrate_quality.py` from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import config          # noqa: E402
from src.quality import assess_quality  # noqa: E402

# Largest-face detector, shared by both modes (loaded once; ~1-2 s).
_face_app = None


def get_bbox(image_bgr: np.ndarray, use_model: bool = True):
    """Largest face bbox via InsightFace, observed every call (or None -> no crop)."""
    global _face_app
    if not use_model:
        return None
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        print("insightface not available; using the whole image as the crop")
        return None
    if _face_app is None:
        _face_app = FaceAnalysis(name=config.MODEL_NAME, providers=config.DET_PROVIDERS)
        _face_app.prepare(ctx_id=-1, det_size=(config.DET_PX, config.DET_PX))
    faces = _face_app.get(image_bgr)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return [float(v) for v in face.bbox]


def print_metrics(image_bgr: np.ndarray, bbox=None, tag: str = "image") -> None:
    """Pretty-print one assess_quality() result."""
    q = assess_quality(image_bgr, bbox)
    s = q["scores"]
    print(
        f"[{tag:<14}] blur(VoL)={q['blur_score']:8.1f}  brightness="
        f"{q['brightness']:6.1f}  contrast={q['contrast']:5.1f}  size="
        f"{q['face_width']}x{q['face_height']}px (frame-ratio {q['face_ratio']:.2f})"
        f" | subscores: blur={s['blur']:.2f} bright={s['brightness']:.2f} "
        f"contrast={s['contrast']:.2f} size={s['face_size']:.2f} overall={s['overall']:.2f}"
    )
    if q["issues"]:
        print(f"{'':16} issues: {'; '.join(q['issues'])}")


def degraded_variants(image_bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """(tag, image) pairs: blur 5/11, dark, bright, low-contrast variants."""
    out: list[tuple[str, np.ndarray]] = [
        ("blur5", cv2.GaussianBlur(image_bgr, (5, 5), 0)),
        ("blur11", cv2.GaussianBlur(image_bgr, (11, 11), 0)),
        ("dark_50%", cv2.convertScaleAbs(image_bgr, alpha=0.5, beta=0)),
        ("dark_25%", cv2.convertScaleAbs(image_bgr, alpha=0.25, beta=0)),
        ("bright_150%", cv2.convertScaleAbs(image_bgr, alpha=1.5, beta=0)),
        ("low_contrast", cv2.convertScaleAbs(image_bgr, alpha=0.4, beta=76)),
    ]
    return out


def run_image_mode(image_path: str) -> int:
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: cannot read {image_path}")
        return 2
    bbox = get_bbox(image)
    if bbox is None:
        print("No face detected - using the whole image as the crop.")
    print(f"Config: BLUR_MIN={config.BLUR_MIN} BLUR_FULL={config.BLUR_FULL} "
          f"BRIGHTNESS_RANGE=({config.BRIGHTNESS_LOW}, {config.BRIGHTNESS_HIGH}) "
          f"CONTRAST_MIN={config.CONTRAST_MIN} MIN_FACE_PX={config.MIN_FACE_PX}")
    print_metrics(image, bbox, "original")
    for tag, variant in degraded_variants(image):
        print_metrics(variant, bbox, tag)
    return 0


def run_webcam_mode() -> int:
    camera = cv2.VideoCapture(config.CAMERA_INDEX)
    if not camera.isOpened():
        print(f"ERROR: cannot open camera {config.CAMERA_INDEX}")
        return 2
    _bbox = None
    frame_idx = 0
    print("Live quality calibration. Press 'q' to quit. Tune src/config.py while watching values.")
    while True:
        ok, frame = camera.read()
        if not ok:
            print("frame read failed")
            break
        # Faces change slowly: full detection every Nth frame is enough here.
        if frame_idx % config.PROCESS_EVERY_N_FRAMES == 0:
            _bbox = get_bbox(frame)
        q = assess_quality(frame, _bbox)
        s = q["scores"]
        text = f"blur={q['blur_score']:.0f}/VoL  brightness={q['brightness']:.0f}  contrast={q['contrast']:.0f}  h={q['face_height']}px  overall={s['overall']:.2f}"
        issues = " | ".join(q["issues"])
        color = (0, 255, 0) if not issues else (0, 165, 255) if s["overall"] >= 0.5 else (0, 0, 255)
        if _bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in _bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, issues, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("calibrate_quality --webcam (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_idx += 1
    camera.release()
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Quality threshold calibration tool")
    parser.add_argument("--image", type=str, help="sample image path for mode (a)")
    parser.add_argument("--webcam", action="store_true", help="live calibration mode")
    args = parser.parse_args()
    if args.webcam:
        return run_webcam_mode()
    if args.image:
        return run_image_mode(args.image)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

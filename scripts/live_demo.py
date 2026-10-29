"""Live demo: webcam or video file driven by the one shared FaceTrust pipeline.

Usage:
    python scripts/live_demo.py                 # camera index from config.CAMERA_INDEX
    python scripts/live_demo.py --source 1      # another camera
    python scripts/live_demo.py --source video.mp4  # recorded video (backup demo)

Keys:
    q  quit                                  s  save annotated frame to results/
    c  toggle "standard recognition only"    m  mirror the display
       (shows what a plain face recognition
       system would say, with no Trust gating)

The heavy analysis is done by LiveSession (src.pipeline), which calls the same
analyze_frame() used for photos, so photo mode and live mode never diverge.
Display note: mirrored frames are flipped BEFORE drawing; bbox coordinates and
the pose arrow x-direction are remapped so overlays stay truthful. The computed
numbers always remain in sensor space.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2

# Allow `python scripts/live_demo.py` from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import config                                # noqa: E402
from src.pipeline import LiveSession, get_default_recognizer  # noqa: E402
from src.pose import draw_pose                        # noqa: E402
from src.recognizer import GALLERY_PATH               # noqa: E402

RESULTS_DIR: Path = REPO_ROOT / "results"

# Decision -> box color (BGR): green ACCEPT, orange WARN, red REJECT.
DECISION_COLOR = {"ACCEPT": (60, 200, 60), "WARN": (20, 160, 255), "REJECT": (40, 40, 230),
                  "PLAIN": (240, 240, 120)}          # cyan-ish for standard mode

# Human-friendly fixes shown at the bottom of the screen.
ISSUE_HINTS: list[tuple[str, str]] = [
    ("Image is blurry", "Too blurry: hold still"),
    ("Too bright", "Too bright: reduce lighting"),
    ("Too dark", "Too dark: improve lighting"),
    ("Low contrast", "Low contrast: even lighting helps"),
    ("Face too small", "Move closer to the camera"),
    ("turned too far", "Turn your face toward the camera"),
    ("tilted", "Look straight ahead"),
    ("not recognised", "Person not in gallery: shown, but flagged"),
    ("No face detected", "No face detected"),
]


def hint_for(issues: list[str]) -> str:
    """Friendly hint for the first matching issue; 'All good' when none."""
    for issue in issues:
        for key, hint in ISSUE_HINTS:
            if key.lower() in issue.lower():
                return hint
    return issues[0] if issues else "All good"


def _draw_badge(frame, bbox: list, decision: str, trust: float) -> None:
    """Filled decision badge at the top-right corner of the bbox."""
    color = DECISION_COLOR[decision]
    label = f"{decision} {trust:.0f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x1, y1 = int(bbox[2]) - tw - 8, max(0, int(bbox[1]) - th - 8)
    cv2.rectangle(frame, (x1, y1), (x1 + tw + 8, y1 + th + 8), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (40, 40, 40) if decision in ("WARN", "PLAIN") else (255, 255, 255), 2)


def annotate(frame, result: dict, standard_mode: bool, mirrored: bool) -> None:
    """Draw overlays for one frame (in place). `mirrored` flips coords/arrow."""
    raw = result["raw"]
    quality = raw.get("quality", {})
    pose = raw.get("pose", {})
    bbox = result.get("bbox")
    width = frame.shape[1]

    # Corner metrics panel and bottom bar are drawn on every frame.
    _draw_panel(frame, result, quality, pose)
    if bbox is None:  # no face in the last analysis (or none yet)
        _draw_bottom_bar(frame, "No face detected: step in front of the camera")
        return

    # Mirrored display: remap bbox x-coords so the box stays on the face.
    if mirrored:
        bbox = [width - int(bbox[2]), int(bbox[1]), width - int(bbox[0]), int(bbox[3])]
        pose_view = dict(pose)
        pose_view["yaw"] = -pose_view.get("yaw", 0.0)  # arrow x-direction flips
    else:
        pose_view = pose

    decision = "PLAIN" if standard_mode else result["decision"]
    color = DECISION_COLOR[decision]
    trust = result["trust_score"]
    similarity = raw.get("similarity")
    x1, y1, x2, y2 = [int(v) for v in bbox]

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{result['identity']}  {similarity:.2f}" if similarity is not None else str(result["identity"])
    cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if standard_mode:
        _draw_bottom_bar(frame, "PLAIN system: identity only, no quality gating "
                                "(compare with trust mode)")
    else:
        _draw_badge(frame, bbox, decision, trust)
        bar_w, bar_h = max(120, x2 - x1), 10
        cv2.rectangle(frame, (x1, y2 + 6), (x1 + bar_w, y2 + 6 + bar_h), (60, 60, 60), -1)
        fill = int(bar_w * max(0.0, min(1.0, trust / 100.0)))
        cv2.rectangle(frame, (x1, y2 + 6), (x1 + fill, y2 + 6 + bar_h), color, -1)
        draw_pose(frame, pose_view, bbox)             # degrees + nose-direction arrow
        _draw_bottom_bar(frame, hint_for(raw.get("trust", {}).get("explanation", [])))


def _draw_panel(frame, result, quality, pose) -> None:
    """Corner metrics panel: blur, brightness, yaw/pitch, trust, FPS."""
    lines = [
        f"blur(VoL): {quality.get('blur_score', 0):.0f}",
        f"brightness: {quality.get('brightness', 0):.0f}",
        f"yaw: {pose.get('yaw', 0):+.1f}  pitch: {pose.get('pitch', 0):+.1f}",
        f"trust: {result['trust_score']:.1f}",
        f"proc FPS: {result['fps']:.1f}",
    ]
    panel_w, line_h = 260, 22
    cv2.rectangle(frame, (0, 0), (panel_w, line_h * len(lines) + 8), (30, 30, 30), -1)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (8, 20 + i * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (220, 220, 220), 1)


def _draw_bottom_bar(frame, message: str) -> None:
    """Bottom strip with the current user-facing hint."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 32), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, message, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)


def main() -> int:
    parser = argparse.ArgumentParser(description="FaceTrust live webcam demo")
    parser.add_argument("--source", type=str, default=str(config.CAMERA_INDEX),
                        help="camera index or video file path (default: config.CAMERA_INDEX)")
    args = parser.parse_args()

    # --source can be a camera index ("0", "1") or a video file path.
    try:
        source: int | str = int(args.source)
    except ValueError:
        source = args.source

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        if isinstance(source, str):
            print(f"ERROR: cannot open video file {source!r} (check path/codec)")
        else:
            print(f"ERROR: cannot open camera {source} - connected in use by another app?")
        return 2

    if not GALLERY_PATH.exists():
        print("[demo] NOTE: no gallery yet (identities will be 'Unknown'). "
              "Enroll people first:  python -m src.recognizer --build-gallery")
    session = LiveSession(recognizer=get_default_recognizer())

    mirrored, standard_mode, fps_tip_done = False, False, False
    window = f"FaceTrust live ({source})"
    try:
        while True:
            ok, frame = capture.read()
            if not ok:  # camera glitch or end of video file
                if isinstance(source, str):
                    print("Video finished.")
                break
            if mirrored:  # flip only the display; computed numbers stay sensor-space
                frame = cv2.flip(frame, 1)

            result = session.process(frame)
            annotate(frame, result, standard_mode, mirrored)
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                path = RESULTS_DIR / f"live_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                cv2.imwrite(str(path), frame)
                print(f"saved {path}")
            if key == ord("c"):
                standard_mode = not standard_mode
                print(f"standard recognition only (no trust gating): {standard_mode}")
            if key == ord("m"):
                mirrored = not mirrored
                print(f"mirrored display: {mirrored}")

            if not fps_tip_done and 0.0 < session.fps < 5.0:
                fps_tip_done = True
                print('Slow analysis (<5 FPS): set MODEL_NAME="buffalo_s" in src/config.py to speed up.')
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""FaceTrust Streamlit app: three tabs (Upload photo / Camera snapshot / Live webcam).

Run locally only:   streamlit run app.py

Structure (kept small and linear for a student project):
    sidebar  -> enroll people, show gallery, trust-threshold sliders
    tabs     -> all three modes call src.pipeline.analyze_frame (photos) or
                LiveSession (live), i.e. the ONE shared analysis path.
Overlay drawing is reused from scripts/live_demo.py so the UI looks the same
as the demo video.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Make `src` and `scripts` importable when launched as `streamlit run app.py`.
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src import config as cfg                                  # noqa: E402
from src.pipeline import LiveSession, analyze_frame            # noqa: E402
from src.recognizer import FaceRecognizer, GALLERY_DIR, GALLERY_PATH  # noqa: E402
from scripts.live_demo import annotate                         # noqa: E402

st.set_page_config(page_title="FaceTrust", page_icon="😐", layout="wide")
st.title("FaceTrust - quality-aware face recognition")

BADGE_COLOR = {"ACCEPT": "#2e7d32", "WARN": "#f57c00", "REJECT": "#c62828"}


# ------------------------------------------------------------------ model load
@st.cache_resource(show_spinner="Loading InsightFace (buffalo_l)...")
def load_recognizer() -> FaceRecognizer:
    """Load the model ONCE per app session (Streamlit cache)."""
    recognizer = FaceRecognizer()
    if GALLERY_PATH.exists():
        recognizer.load_gallery()
    return recognizer


# ------------------------------------------------------------------- helpers
def _badge(decision: str, score: float) -> None:
    """Colored decision badge (green/orange/red) with the Trust Score."""
    st.markdown(
        f"<span style='color:white;background-color:{BADGE_COLOR[decision]};"
        f"padding:4px 14px;border-radius:8px;font-weight:600'>"
        f"{decision} &nbsp; Trust {score:.1f}</span>",
        unsafe_allow_html=True,
    )


def _table_with_marks(res: dict) -> pd.DataFrame:
    """Metrics table: raw value + PASS / FALL / blurry-ish sub-score mark."""
    q, p = res["quality"], res["pose"]
    scores = q.get("scores", {})
    rows = [
        ("blur (VoL)", f"{q.get('blur_score', 0):.0f}", scores.get("blur", 0.0)),
        ("brightness (mean)", f"{q.get('brightness', 0):.0f}", scores.get("brightness", 0.0)),
        ("contrast (std)", f"{q.get('contrast', 0):.0f}", scores.get("contrast", 0.0)),
        ("face size (px)", f"{q.get('face_height', 0)}", scores.get("face_size", 0.0)),
        ("yaw (deg)", f"{p.get('yaw', 0):+.1f}", 1.0 if abs(p.get("yaw", 0)) < cfg.YAW_MAX * cfg.POSE_SOFT_FACTOR else 0.5),
        ("pitch (deg)", f"{p.get('pitch', 0):+.1f}", 1.0 if abs(p.get("pitch", 0)) < cfg.PITCH_MAX * cfg.POSE_SOFT_FACTOR else 0.5),
        ("roll (deg)", f"{p.get('roll', 0):+.1f}", 1.0 if abs(p.get("roll", 0)) < cfg.ROLL_MAX * cfg.POSE_SOFT_FACTOR else 0.5),
    ]
    return pd.DataFrame(
        {
            "metric": [r[0] for r in rows],
            "value": [r[1] for r in rows],
            "status": ["✓ pass" if r[2] >= 1.0 else ("✗ fail" if r[2] <= 0.0 else "± partial") for r in rows],
            "sub-score": [f"{r[2]:.2f}" for r in rows],
        }
    )


def _plain_caption(res: dict) -> None:
    """One-line comparison to plain face recognition (no quality gating)."""
    sim = res.get("similarity")
    if sim is not None:
        st.caption(f"Standard (no quality check) would report: "
                   f"{res['identity']}  sim {sim:.2f} - same box, no Trust gating")


def _show_results(frame: np.ndarray, res: dict) -> None:
    """Common result panel for photo/snapshot: annotated image + trust info."""
    compare = st.session_state.get("compare_standard", False)
    pseudo = {  # annotate() expects the live-style wrapper; wrap the photo result
        "raw": res, "identity": res["identity"], "decision": res["trust"]["decision"],
        "trust_score": res["trust"]["trust_score"], "bbox": res.get("bbox"), "fps": 0.0,
    }
    annotate(frame, pseudo, standard_mode=compare, mirrored=False)

    image_col, info_col = st.columns([3, 2])
    with image_col:
        st.image(frame, channels="BGR")
        if compare:
            _plain_caption(res)
    with info_col:
        _badge(res["trust"]["decision"], res["trust"]["trust_score"])
        st.progress(min(100, max(0, int(res["trust"]["trust_score"]))),
                    text=f"Trust Score {res['trust']['trust_score']:.1f}")
        st.write(f"**Identity:** {res['identity']}   **Similarity:** "
                 f"{res['similarity']:.2f}" if res["similarity"] is not None else
                 f"**Identity:** {res['identity']}")
        if res["trust"]["decision"] != "ACCEPT":
            st.warning(res["trust"]["message"])     # what to fix / retake advice
    if res["face_found"]:
        st.dataframe(_table_with_marks(res), hide_index=True, use_container_width=True)


def _decode_image(data: bytes) -> np.ndarray | None:
    """UploadedFile/camera bytes -> BGR image (or None if unreadable)."""
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------- tabs
recognizer = load_recognizer()   # model cached once per user session
tab_photo, tab_snapshot, tab_live = st.tabs(["📤 Upload photo", "📷 Camera snapshot", "🎥 Live webcam"])

# 1) Upload photo --------------------------------------------------------------
with tab_photo:
    uploads = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])
    if uploads is None:
        st.caption("The photo is analyzed once you upload it.")
    else:
        frame = _decode_image(uploads.getvalue())
        if frame is None:
            st.error("Could not decode that file.")
        else:
            _show_results(frame, analyze_frame(frame, recognizer, mode="photo"))

# 2) Camera snapshot ----------------------------------------------------------
with tab_snapshot:
    snap_photo = st.camera_input("Take a photo with the webcam")
    if snap_photo is not None:
        frame = _decode_image(snap_photo.getvalue())
        if frame is None:
            st.error("Could not decode the capture.")
        else:
            _show_results(frame, analyze_frame(frame, recognizer, mode="photo"))

# 3) Live webcam --------------------------------------------------------------
with tab_live:
    state = st.session_state
    state.setdefault("live", False)

    st.caption("Same LiveSession pipeline as photos: analysis every "
               f"{cfg.PROCESS_EVERY_N_FRAMES}th frame, smoothing over "
               f"{cfg.SMOOTHING_WINDOW} results. Runs locally only.")

    cols = st.columns([1, 1, 6])
    if cols[0].button("▶ Start", disabled=state["live"]):
        state["live"] = True
        st.rerun()
    if cols[1].button("⏹ Stop", type="primary", disabled=not state["live"]):
        state["live"] = False
        st.rerun()

    frame_slot = st.empty()
    metrics_slot = st.empty()

    if state["live"]:
        capture = cv2.VideoCapture(cfg.CAMERA_INDEX)
        try:
            if not capture.isOpened():
                st.error(f"Cannot open camera {cfg.CAMERA_INDEX} (already in use?).")
                state["live"] = False
            else:
                session = LiveSession(recognizer=recognizer)
                while state["live"]:
                    # Stop must be a real widget INSIDE the loop: Streamlit
                    # interrupts the running script on widget events, so this
                    # is what actually ends the video loop.
                    if st.button("⏹ Stop", key="live_stop", type="primary"):
                        state["live"] = False
                        break
                    ok, frame = capture.read()  # rerun leaves this branch.
                    if not ok:
                        st.error("Could not read frame.")
                        break
                    result = session.process(frame)
                    annotate(frame, result,
                             standard_mode=st.session_state.get("compare_standard", False),
                             mirrored=False)
                    frame_slot.image(frame, channels="BGR")
                    metrics_slot.markdown(
                        f"Trust **{result['trust_score']:.1f}** ({result['decision']}) | "
                        f"identity **{result['identity']}** | "
                        f"analysis FPS **{result['fps']:.1f}** | "
                        f"{result['message'] or 'All good'} "
                        f"<span style='color:#888'>- press Stop to release the camera</span>",
                        unsafe_allow_html=True,
                    )
        finally:
            capture.release()   # always release, even on Stop/interrupt
            st.write("Camera released.")

# 4) Sidebar ---------------------------------------------------------------
st.sidebar.header("Enroll person")
name = st.sidebar.text_input("Person name")
new_photos = st.sidebar.file_uploader("…or upload photos", type=["jpg", "jpeg", "png"],
                                      accept_multiple_files=True)
snap = st.sidebar.camera_input("…or capture a photo here", key="enroll_cam",
                               help="Uses the browser camera - stop the Live tab first to avoid the webcam being locked by two streams.")
if st.sidebar.button("Add & rebuild gallery"):
    saved = 0
    if name and (new_photos or snap is not None):
        person_dir = GALLERY_DIR / name.strip()
        person_dir.mkdir(parents=True, exist_ok=True)
        for blob in (new_photos or []):
            (person_dir / f"up_{datetime.now():%Y%m%d_%H%M%S}_{saved}.jpg").write_bytes(blob.getvalue())
            saved += 1
        if snap is not None:       # save the snapshot too
            (person_dir / f"cam_{datetime.now():%Y%m%d_%H%M%S}.jpg").write_bytes(snap.getvalue())
            saved += 1
        if saved:
            with st.sidebar.spinner("Rebuilding gallery..."):
                recognizer.build_gallery()
            st.sidebar.success(f"Enrolled {name} with {saved} photo(s).")
    else:
        st.sidebar.warning("Need a name and at least one photo.")

st.sidebar.subheader("Enrolled people")
people = sorted(p.name for p in GALLERY_DIR.iterdir() if p.is_dir())
st.sidebar.write(people if people else "*none yet*")

st.sidebar.subheader("Trust thresholds")
cfg.ACCEPT_THRESHOLD = st.sidebar.slider("ACCEPT >= ?", 0, 100, int(cfg.ACCEPT_THRESHOLD))
cfg.WARN_THRESHOLD = st.sidebar.slider("WARN >= ?", 0, 100, int(cfg.WARN_THRESHOLD))
st.sidebar.toggle("Compare with standard face recognition (no quality check)",
                  key="compare_standard")

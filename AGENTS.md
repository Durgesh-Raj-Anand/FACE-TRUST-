# AGENTS.md - project spec for any agent (or the author) working on FaceTrust

## Idea
Face recognition that consciously measures HOW GOOD each image is before
trusting its answer. For every frame we compute:
1. Identity: InsightFace buffalo_l embedding vs. a small personal gallery
   (cosine similarity).
2. Quality: blur (Laplacian variance), brightness (mean), contrast (std),
   face size (bbox height in px).
3. Pose: yaw/pitch/roll from the 5 keypoints.
Combined via weights in `src/config.py` into a Trust Score 0-100 -> decision
ACCEPT (>=70) / WARN (40-70) / REJECT (<40) plus a one-line advice message
("move closer", "more light", "face the camera", ...).

## Hard rules (do not break)
- `src/pipeline.analyze_frame()` is the ONLY analysis path. Photos and video
  share it. Live mode only adds:
    - `LiveSmoother`: moving average over the last `SMOOTHING_WINDOW` (5) results.
    - Full analysis only every `PROCESS_EVERY_N_FRAMES` (5) frames; between
      runs reuse the smoothed last result.
- ALL thresholds, weights, `CAMERA_INDEX`, `MODEL_NAME`, smoothing constants
  live in `src/config.py`. Never hardcode numbers elsewhere.
- Keep code simple, typed, commented. This is a student project shown in a viva.
- Runs locally and offline-ish (models downloaded once); no cloud APIs.

## Stack
Python 3.10/3.11 (present: 3.11.9), insightface + onnxruntime (CPU only),
opencv-python, numpy, streamlit, pandas, matplotlib, pytest.
If insightface fails to install on the OS, ask the user BEFORE switching to
the fallback: facenet-pytorch (embeddings) + mediapipe (landmarks).

## Structure
see README.md. Data:
- `data/gallery/<person>/*.jpg` - enrollment photos (build the gallery).
- `data/test/<person>/*.jpg` - evaluation photos.
- `results/` - evaluation outputs, plots.

## Build order (see PROGRESS.md)
1. Setup: structure, venv, dependencies, InsightFace + webcam verification.  <- done now
2. recognizer.py (detection, embeddings, gallery add/save/load/match).
3. quality.py (four checks with scores + ok flags).
4. pose.py (yaw/pitch/roll from keypoints).
5. trust.py (weights -> Trust Score; decision + advice message).
6. pipeline.py (analyze_frame + LiveSmoother).
7. Enrollment script / gallery from data/gallery.
8. app.py Streamlit tabs (Upload / snapshot / live).
9. scripts/: evaluate.py, calibrate_quality.py, live_demo.py.
10. Tests + final README polish.
Update PROGRESS.md after every step.

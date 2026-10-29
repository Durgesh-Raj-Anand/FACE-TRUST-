# PROGRESS.md - build checklist (update after every step)

- [x] 1. Setup: folder structure, stub files with docstrings, requirements.txt
- [x] 1b. src/config.py with all thresholds/weights/model/camera/smoothing values
- [x] 1c. venv created (.venv, Python 3.11.9), all dependencies installed
      (numpy 2.4.6, opencv 4.11.0.86, onnxruntime 1.31.1, insightface 2.1,
      pandas 3.0.6, matplotlib 3.11.2, streamlit 1.66.1, pytest 9.2.0)
- [x] 1d. InsightFace verified: buffalo_l loaded from ~/.insightface/models,
      FaceAnalysis runs on a sample frame without errors
      (0 detections in test frames = no face in view at the time; confirm with
      a real photo: `python -m src.recognizer --image <photo.jpg>`)
- [x] 1e. Webcam verified: cv2.VideoCapture(0) opens, frame read OK, size 640x480
- [x] 1f. git init, .gitignore, first commit
- [x] 2. src/recognizer.py: FaceRecognizer - largest-face detection, 512-d
      embeddings, build_gallery()->data/gallery_embeddings.pkl, load_gallery(),
      identify() with cosine matching + second-best, CLI (--image / --webcam /
      --build-gallery). Imports + syntax OK.
- [ ] 2b. Live test of recognizer with real gallery photos (needs your photos
      in data/gallery/<person>/, then: `python -m src.recognizer --build-gallery`)
- [x] 3. src/quality.py: assess_quality(image_bgr, bbox) - Laplacian-VoL blur,
      brightness, contrast, face size (px + frame ratio), normalized subscores
      (0-1) from config thresholds, human-readable issue list. ~0.94 ms/call.
- [x] 3b. scripts/calibrate_quality.py: --image mode (original + blurred/dark/
      bright/low-contrast variants printed) and --webcam live overlay mode.
- [x] 4. src/pose.py: assess_pose(pose, kps) - magnitudes from InsightFace
      face.pose (order [pitch,yaw,roll], verified in insightface src), SIGNS
      re-derived in image space from the 5 kps (yaw: nose vs eye-midpoint x;
      pitch: nose between eye/mouth lines; roll: eye-line angle). Smooth decay
      score (POSE_SOFT_FACTOR..YAW_MAX/PITCH_MAX/ROLL_MAX), weighted
      YAW/PITCH/ROLL via POSE_SCORE_WEIGHTS; issues incl. left/right/up/down.
      draw_pose() overlays degrees + nose-direction arrow (un-mirrored frames).
- [ ] 4b. Live visual check of pose left/right wording on the webcam
      (calibrate once: turn your head left, watch yaw sign flip; one-word fix
      in _yaw_text() if any mirrored display is ever assumed)
- [ ] 5. src/trust.py: Trust Score, decision bands, advice messages
- [ ] 6. src/pipeline.py: analyze_frame() + LiveSmoother
- [ ] 7. Enrollment: build gallery from data/gallery/*/*.jpg (same as 2b)
- [ ] 8. app.py: Streamlit tabs (Upload photo / Camera snapshot / Live webcam)
- [ ] 9. scripts: evaluate.py, calibrate_quality.py, live_demo.py
- [ ] 10. pytest tests + README polish + final commit

## Notes
- Python on this machine: 3.11.9 (`python`), 3.13.15 (`py`). Project targets 3.11.
- insightface 2.1 is a pure-Python wheel on PyPI (no MSVC compiler needed).
- No Visual C++ build tools installed; do not depend on compiling extension modules.
- Network flaky for pypi.org but fast for github.com: model packs via insightface
  download OK; if pip is slow use `--timeout 60 --resume-retries 100`.
- buffalo_l pack is installed in ~/.insightface/models (also kept buffalo_l.zip
  workflow with curl -C - if a reinstall is ever needed).
- Webcam: 640x480, index 0 works.
- Distribution spec for runtime test of recognizer: set PYTHONPATH=. from the
  facetrust root, or run scripts with `python -m ...` from repo root.

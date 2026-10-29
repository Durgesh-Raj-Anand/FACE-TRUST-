# PROGRESS.md - build checklist (update after every step)

- [x] 1. Setup: folder structure, stub files with docstrings, requirements.txt
- [x] 1b. src/config.py with all thresholds/weights/model/camera/smoothing values
- [ ] 1c. venv created, dependencies installed (in progress - slow network,
      resumable pip installs running in background; everything cached except
      opencv + insightface/onnx/scipy/scikit-image chain)
- [ ] 1d. InsightFace verified on a sample image
- [ ] 1e. Webcam verified (cv2.VideoCapture(0) reads a frame)
- [x] 1f. git init, .gitignore, first commit
- [x] 2. src/recognizer.py: FaceRecognizer - largest-face detection, 512-d
      embeddings, build_gallery()->data/gallery_embeddings.pkl, load_gallery(),
      identify() with cosine matching + second-best, CLI (--image / --webcam /
      --build-gallery). Syntax-checked; runtime test pending install (1c).
- [ ] 3. src/quality.py: blur, brightness, contrast, face-size checks
- [ ] 4. src/pose.py: yaw / pitch / roll
- [ ] 5. src/trust.py: Trust Score, decision bands, advice messages
- [ ] 6. src/pipeline.py: analyze_frame() + LiveSmoother
- [ ] 7. Enrollment: build gallery from data/gallery/*/*.jpg
- [ ] 8. app.py: Streamlit tabs (Upload photo / Camera snapshot / Live webcam)
- [ ] 9. scripts: evaluate.py, calibrate_quality.py, live_demo.py
- [ ] 10. pytest tests + README polish + final commit

## Notes
- Python on this machine: 3.11.9 (`python`), 3.13.15 (`py`). Project targets 3.11.
- insightface 2.1 is a pure-Python wheel on PyPI (no MSVC compiler needed).
- Network is very slow/flaky here: use `pip install --timeout 60 --resume-retries 100`.
- No Visual C++ build tools installed; do not depend on compiling extension modules.

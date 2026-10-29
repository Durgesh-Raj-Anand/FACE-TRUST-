# PROGRESS.md - build checklist (update after every step)

- [x] 1. Setup: folder structure, stub files with docstrings, requirements.txt
- [x] 1b. src/config.py with all thresholds/weights/model/camera/smoothing values
- [ ] 1c. venv created, dependencies installed
- [ ] 1d. InsightFace verified on a sample image
- [ ] 1e. Webcam verified (cv2.VideoCapture(0) reads a frame)
- [ ] 1f. git init, .gitignore, first commit
- [ ] 2. src/recognizer.py: detect faces, embeddings, gallery add/save/load/match
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

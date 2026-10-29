# FaceTrust: Quality-Aware Face Recognition Under Challenging Visual Conditions

College computer-vision project (B.Tech 3rd year). Face recognition that also
checks **image quality** (blur, brightness, contrast, face size) and **head pose**
(yaw/pitch/roll), combines everything into a **Trust Score (0-100)** and returns:

| Trust Score | Decision | Meaning |
|---|---|---|
| >= 70 | ACCEPT | identity + conditions are good |
| 40 - 70 | WARN | recognized, but conditions poor - fix the listed issues |
| < 40 | REJECT | conditions too poor to trust the result |

Works on **still photos** (upload or camera snapshot) and **LIVE webcam video**.
Both modes share one core function, `src.pipeline.analyze_frame()`; live mode
only adds temporal smoothing (moving average over the last 5 results, full
analysis every 5th frame).

## Structure
```
facetrust/
  app.py                     # Streamlit UI (Upload / Camera snapshot / Live tabs)
  requirements.txt
  src/
    config.py                # ALL thresholds, weights, camera & smoothing settings
    recognizer.py            # InsightFace detection + embeddings + gallery matching
    quality.py               # blur / brightness / contrast / face-size checks
    pose.py                  # yaw / pitch / roll estimation
    trust.py                 # Trust Score + ACCEPT/WARN/REJECT decision + advice
    pipeline.py              # analyze_frame() + LiveSmoother (shared by both modes)
  data/gallery/<person>/*.jpg
  data/test/<person>/*.jpg
  scripts/                   # live_demo, evaluate, calibrate_quality
  results/                   # outputs, plots, evaluation reports
```

## Quick start
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Stack
Python 3.10/3.11, InsightFace (`buffalo_l`, CPUExecutionProvider) + onnxruntime,
opencv-python, numpy, streamlit, pandas, matplotlib, pytest.

Fallback (only if InsightFace fails to install on the OS): facenet-pytorch for
embeddings + mediapipe for landmarks. Not used unless necessary.

## Design rules
- One single-frame function (`analyze_frame`) powers photo AND live mode.
- Live mode adds ONLY smoothing (moving average over the last 5 results,
  full analysis every 5th frame). No separate logic.
- Every tunable number lives in `src/config.py`.
- Keep code simple, typed and commented - it must be explainable in a viva.

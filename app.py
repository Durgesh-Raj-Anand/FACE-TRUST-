"""FaceTrust Streamlit app.

Tabs:
    1. Upload photo    - still image -> single analyze_frame() call
    2. Camera snapshot - webcam snapshot -> same call
    3. Live webcam     - video loop with temporal smoothing only

Run:  streamlit run app.py
"""

from src import config  # noqa: F401  (logic added in a later step)

"""FaceTrust central configuration.

All tunable thresholds, weights and constants live here so experiments
and calibration only ever touch this file.
"""

# ----- Detection / recognition model ------------------------------------
MODEL_NAME: str = "buffalo_l"          # InsightFace model pack ("buffalo_l" or "buffalo_s")
DET_PROVIDERS = ["CPUExecutionProvider"]  # CPU-only execution
DET_PX: int = 640                      # detection input size

# ----- Camera / live mode ------------------------------------------------
CAMERA_INDEX: int = 0                  # laptop webcam
PROCESS_EVERY_N_FRAMES: int = 5        # full analysis only every Nth live frame
SMOOTHING_WINDOW: int = 5              # moving average over last N results

# ----- Trust Score thresholds (0-100) ------------------------------------
ACCEPT_THRESHOLD: float = 70.0         # >= ACCEPT  -> ACCEPT
WARN_THRESHOLD: float = 40.0           # >= WARN, < ACCEPT -> WARN  (< WARN -> REJECT)

# ----- Identity threshold -------------------------------------------------
RECOG_MIN_CONFIDENCE: float = 0.35     # cosine below this -> identity "Unknown"
RECOG_FULL_CONFIDENCE: float = 0.65    # cosine >= this -> full recognition score
MARGIN_FULL: float = 0.15              # sim - second_best >= this -> full margin credit

# ----- Detector score (InsightFace det_score is commonly ~0.5-1.0) ---------
DET_MIN_SCORE: float = 0.40            # mapped to 0
DET_FULL_SCORE: float = 0.90           # mapped to 1

# ----- Quality checks (all distances/ints in pixels or 8-bit gray levels) -----
# Blur: variance of the Laplacian (VoL). Higher = sharper.
BLUR_MIN: float = 100.0     # VoL below this -> "Image is blurry" (0 score)
BLUR_FULL: float = 300.0    # VoL >= this -> full blur score (1.0), reads crisp
# Brightness: mean gray intensity inside the face crop.
BRIGHTNESS_LOW: float = 60.0    # mean < this -> "Too dark"
BRIGHTNESS_HIGH: float = 200.0  # mean > this -> "Too bright/overexposed"
# Contrast: std of gray intensity inside the face crop.
CONTRAST_MIN: float = 40.0    # std below this -> "Low contrast" (0 score)
CONTRAST_FULL: float = 80.0   # std >= this -> full contrast score (1.0)
# Face size: bbox height in pixels relative to the frame.
MIN_FACE_PX: int = 80         # height < this -> "Face too small" (0 score)
FULL_FACE_PX: int = 240       # height >= this -> full face-size score (1.0)

# ----- Pose limits (degrees; score decays from SOFT_FACTOR*limit to limit) --
YAW_MAX: float = 35.0      # |yaw| >= this -> issue + zero yaw sub-score
PITCH_MAX: float = 25.0    # |pitch| >= this -> issue + zero pitch sub-score
ROLL_MAX: float = 30.0     # |roll| >= this -> issue (head tilted sideways)
POSE_SOFT_FACTOR: float = 0.5   # decay starts at max * this (smooth score)
# Sub-score weights for yaw / pitch / roll (must sum to 1.0).
POSE_SCORE_WEIGHTS: tuple = (0.45, 0.35, 0.20)

# ----- Score weights (must sum to 1.0) ------------------------------------
# Recognition block = 0.30 (similarity + margin + detector), quality/pose = 0.70,
# so a badly-lit or badly-posed frame clearly cannot reach ACCEPT on identity alone.
TRUST_WEIGHTS: dict = {
    "recognition": 0.20,   # cosine similarity mapped via RECOG_* thresholds
    "margin": 0.05,        # margin over second-best gallery match
    "detector": 0.05,      # InsightFace det_score mapped via DET_* thresholds
    "blur": 0.10,          # quality.py blur sub-score
    "lighting": 0.25,      # quality.py brightness+contrast (averaged)
    "face_size": 0.05,     # quality.py face-size sub-score
    "pose": 0.30,          # pose.py sub-score
}
UNKNOWN_SCORE_CAP: float = 55.0   # not-recognised faces never exceed this

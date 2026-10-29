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

# ----- Pose limits (degrees, approximate daily-use range) ------------------
YAW_LIMIT: float = 30.0
PITCH_LIMIT: float = 25.0
ROLL_LIMIT: float = 30.0

# ----- Score weights (must sum to 1.0) ------------------------------------
WEIGHT_IDENTITY: float = 0.6           # similarity to gallery match
WEIGHT_QUALITY: float = 0.25           # blur+brightness+contrast+size
WEIGHT_POSE: float = 0.15              # head pose penalty

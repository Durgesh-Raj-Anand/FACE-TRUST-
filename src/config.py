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

# ----- Quality checks (placeholder values; calibrate later) ---------------
BLUR_THRESHOLD: float = 60.0           # Laplacian variance >= this = sharp
BRIGHTNESS_LOW: float = 60.0           # mean intensity range considered OK
BRIGHTNESS_HIGH: float = 200.0
CONTRAST_THRESHOLD: float = 40.0       # std of gray image
FACE_SIZE_MIN_PX: int = 80             # min face bbox height in pixels

# ----- Pose limits (degrees, approximate daily-use range) ------------------
YAW_LIMIT: float = 30.0
PITCH_LIMIT: float = 25.0
ROLL_LIMIT: float = 30.0

# ----- Score weights (must sum to 1.0) ------------------------------------
WEIGHT_IDENTITY: float = 0.6           # similarity to gallery match
WEIGHT_QUALITY: float = 0.25           # blur+brightness+contrast+size
WEIGHT_POSE: float = 0.15              # head pose penalty

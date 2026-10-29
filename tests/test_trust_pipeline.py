"""Unit tests for trust.py and pipeline.py (synthetic inputs, no model needed).

Weights used (src/config.py): recognition .20, margin .05, detector .05,
blur .10, lighting .25, face_size .05, pose .35 -> total 1.00.
"""

import numpy as np

from src.pipeline import LiveSession
from src.trust import compute_trust

# Synthetic building blocks for compute_trust().
GOOD_REC = {"face_found": True, "identity": "alice", "similarity": 0.80,
            "second_best_similarity": 0.40, "det_score": 0.90}
GOOD_QUALITY = {"scores": {"blur": 1.0, "brightness": 1.0, "contrast": 1.0,
                           "face_size": 1.0, "overall": 1.0}, "issues": []}
GOOD_POSE = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "pose_score": 1.0, "issues": []}


def test_no_face_is_hard_reject() -> None:
    """Spec hard rule: no face -> trust 0 -> REJECT, regardless of quality."""
    result = compute_trust({"face_found": False}, GOOD_QUALITY, GOOD_POSE, mode="photo")
    assert result["trust_score"] == 0.0
    assert result["decision"] == "REJECT"
    assert "No face" in result["message"]


def test_high_quality_frontal_match_accepts() -> None:
    """Known person + perfect quality/pose -> all inputs 1.0 -> 100 -> ACCEPT."""
    result = compute_trust(GOOD_REC, GOOD_QUALITY, GOOD_POSE, mode="photo")
    assert result["trust_score"] >= 70.0
    assert result["decision"] == "ACCEPT"
    assert result["explanation"] == []
    assert "alice" in result["message"]


def test_dark_blurry_photo_warns() -> None:
    """Blurry + very dark: loses blur (0.10) and most lighting (0.25) -> < 70."""
    bad = {"scores": {"blur": 0.0, "brightness": 0.1, "contrast": 0.1,
                      "face_size": 1.0, "overall": 0.275},
           "issues": ["Image is blurry", "Too dark", "Low contrast"]}
    result = compute_trust(GOOD_REC, bad, GOOD_POSE, mode="photo")
    good = compute_trust(GOOD_REC, GOOD_QUALITY, GOOD_POSE, mode="photo")
    assert result["trust_score"] < good["trust_score"]
    assert result["trust_score"] < 70.0
    assert result["decision"] == "WARN"
    assert any("blurry" in e.lower() for e in result["explanation"])
    assert "clearer" in result["message"]            # photo-mode wording


def test_extreme_pose_warns_with_live_wording() -> None:
    """Head far to the side: pose input 0 (-0.30) and slim margin -> WARN."""
    turned_rec = {"face_found": True, "identity": "alice", "similarity": 0.80,
                  "second_best_similarity": 0.75, "det_score": 0.90}  # margin 0.05
    turned = {"yaw": 50.0, "pitch": 0.0, "roll": 0.0, "pose_score": 0.0,
              "issues": ["Face turned too far right - face the camera"]}
    result = compute_trust(turned_rec, GOOD_QUALITY, turned, mode="live")
    assert result["trust_score"] < 70.0
    assert result["decision"] == "WARN"
    assert any("right" in e for e in result["explanation"])
    assert "face the camera" in result["message"]    # live-mode wording


def test_live_session_smooths_and_majority_votes() -> None:
    """LiveSession: moving average, majority vote, heavy-run count and FPS."""
    calls = {"n": 0}

    def fake_analyzer(frame, recognizer, mode):
        calls["n"] += 1
        score = 100.0 if calls["n"] % 2 else 0.0     # heavy runs give 100,0,100,...
        identity = "alice" if calls["n"] == 1 else "bob"  # bob from run 2 on
        return {"identity": identity, "similarity": 0.9, "det_score": 0.9,
                "bbox": [0, 0, 10, 10],
                "trust": {"trust_score": score, "decision": "ACCEPT",
                          "explanation": [], "message": ""}}

    session = LiveSession(analyzer=fake_analyzer)
    frame = np.zeros((48, 48, 3), dtype=np.uint8)
    results = [session.process(frame) for _ in range(11)]  # heavy at frames 1, 6, 11

    # Heavy analysis every PROCESS_EVERY_N_FRAMES (5) frames.
    assert calls["n"] == 3
    assert results[0]["trust_score"] == 100.0
    assert results[4]["trust_score"] == 100.0        # between runs: last raw kept
    assert results[4]["identity"] == "alice"         # single vote so far
    # Third heavy run: window scores [100, 0, 100] -> mean; votes bob majority.
    assert results[10]["trust_score"] == round((100.0 + 0.0 + 100.0) / 3.0, 1)
    assert results[10]["identity"] == "bob"
    assert results[10]["fps"] > 0.0

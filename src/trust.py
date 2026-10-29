"""Trust Score: combine identity, quality and pose into 0-100 with a decision.

Decision bands (src/config.py): >= ACCEPT_THRESHOLD -> ACCEPT,
>= WARN_THRESHOLD -> WARN, otherwise REJECT. Hard rule: no face found -> REJECT.

Design notes (viva-friendly):
- Recognition block (similarity + margin over runner-up + detector) is 0.30 of
  the total, quality/pose are 0.70, so poor visual conditions can clearly
  downgrade even a confident match (that is the whole point of FaceTrust).
- A face the gallery does not know may still be well photographed; it is capped
  at UNKNOWN_SCORE_CAP so "Unknown" can never be ACCEPT-ed.
"""

from __future__ import annotations

from typing import Any, Dict

from . import config


def _clip01(value: float) -> float:
    """Clamp a value into [0, 1]."""
    return float(min(1.0, max(0.0, value)))


def identity_of(recognition: Dict[str, Any]) -> str:
    """Return the person name, or "Unknown" when similarity is too low."""
    sim = recognition.get("similarity")
    if sim is None:
        return "Unknown"
    return recognition["identity"] if sim >= config.RECOG_MIN_CONFIDENCE else "Unknown"


def decide(trust_score: float) -> str:
    """Map a Trust Score to ACCEPT / WARN / REJECT per config thresholds."""
    if trust_score >= config.ACCEPT_THRESHOLD:
        return "ACCEPT"
    if trust_score >= config.WARN_THRESHOLD:
        return "WARN"
    return "REJECT"


def _advice(mode: str) -> str:
    """Mode-dependent user advice ("photo" vs "live" wording)."""
    return ("Please upload a clearer, front-facing image."
            if mode == "photo" else
            "Please face the camera / improve lighting.")


def _message(decision: str, name: str, is_known: bool,
             issues: list[str], mode: str) -> str:
    """Build one user-facing sentence depending on decision, issues and mode."""
    if not is_known:
        return (f"Person not recognised{f': {issues[0]}' if issues else ''}. "
                f"{_advice(mode)}")
    if decision == "ACCEPT" and not issues:
        return f"Good conditions, identity is {name}."
    if decision == "ACCEPT":
        return f"Accepted, but note: {'; '.join(issues)}."
    return f"Result unreliable: {'; '.join(issues)}. {_advice(mode)}"


def compute_trust(
    recognition: Dict[str, Any],
    quality: Dict[str, Any],
    pose: Dict[str, Any],
    mode: str = "photo",
) -> Dict[str, Any]:
    """Weighted combination of recognition, quality and pose into Trust Score.

        Args:
            recognition: identify() result (face_found, similarity, det_score,
                         identity, second_best_similarity ...).
            quality: assess_quality() result (scores dict + issues).
            pose: assess_pose() result (pose_score + issues).
            mode: "photo" or "live" (only changes the advice wording).

        Returns:
            {"trust_score": 0-100, "decision", "explanation", "message", "mode"}.
        """
    # ---- hard rule: no face -> nothing to trust ----
    if not recognition.get("face_found", False):
        return {"trust_score": 0.0, "decision": "REJECT",
                "explanation": ["No face detected"],
                "message": "No face detected.", "mode": mode}

    # ---- input mappings, each in [0, 1] ---
    known = identity_of(recognition) != "Unknown"
    sim = float(recognition.get("similarity") or 0.0)
    rec_input = (_clip01((sim - config.RECOG_MIN_CONFIDENCE) /
                         max(1e-6, config.RECOG_FULL_CONFIDENCE - config.RECOG_MIN_CONFIDENCE))
                 if known else 0.0)          # unknown contributes no identity confidence
    margin = None
    second = recognition.get("second_best_similarity")
    if is_margin_defined := (known and second is not None):
        margin = sim - float(second)
    margin_input = _clip01(margin / config.MARGIN_FULL) if margin is not None else (
        1.0 if known else 0.0)               # single-person gallery: no confusable
    det = float(recognition.get("det_score") or 0.0)
    det_input = _clip01((det - config.DET_MIN_SCORE) /
                        max(1e-6, config.DET_FULL_SCORE - config.DET_MIN_SCORE))

    qs = quality.get("scores", {})
    lighting_input = (float(qs.get("brightness", 0.0)) + float(qs.get("contrast", 0.0))) / 2.0
    inputs = {
        "recognition": rec_input,
        "margin": margin_input,
        "detector": det_input,
        "blur": float(qs.get("blur", 0.0)),
        "lighting": lighting_input,
        "face_size": float(qs.get("face_size", 0.0)),
        "pose": float(pose.get("pose_score", 0.0)),
    }
    score = 100.0 * float(sum(config.TRUST_WEIGHTS[key] * inputs[key] for key in config.TRUST_WEIGHTS))
    if not known:
        score = min(score, config.UNKNOWN_SCORE_CAP)  # Unknown can never ACCEPT

    # ---- explanation (quality issues + pose issues + identity note) ----
    explanation = list(quality.get("issues", [])) + list(pose.get("issues", []))
    name = identity_of(recognition)
    decision = decide(score)

    return {
        "trust_score": round(score, 1),
        "decision": decision,
        "explanation": explanation,
        "message": _message(decision, name, known, explanation, mode),
        "mode": mode,
    }

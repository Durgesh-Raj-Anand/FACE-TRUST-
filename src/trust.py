"""Trust Score: combine identity, quality and pose into 0-100 with a decision.

Decision bands (see src/config.py):
    score >= ACCEPT_THRESHOLD  -> ACCEPT
    score >= WARN_THRESHOLD    -> WARN
    else                       -> REJECT
"""

from typing import Dict, Tuple


def combine_scores(
    identity_score: float,   # similarity mapped into [0, 1]
    quality_score: float,    # [0, 1] from quality.py
    pose_score: float,       # [0, 1] from pose.py
) -> float:
    """Weighted combination of the three sub-scores -> Trust Score in [0, 100]."""


def decide(trust_score: float) -> Tuple[str, str]:
    """Map a Trust Score to (decision, advice_message).

    Returns:
        ("ACCEPT" | "WARN" | "REJECT", human-readable message on what to fix).
    """

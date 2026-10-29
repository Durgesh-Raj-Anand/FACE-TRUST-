"""Face recognition via InsightFace: detection + embedding + gallery matching.

Embeddings are 512-d vectors compared with cosine similarity.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


class Recognizer:
    """Thin wrapper around InsightFace FaceAnalysis (buffalo_l, CPU)."""

    def __init__(self) -> None:
        """Load the model pack once and prepare an empty gallery."""

    def detect_single_face(self, frame_bgr: np.ndarray) -> Optional[object]:
        """Detect the largest face in the frame; return its Face object or None."""

    def get_embedding(self, frame_bgr: np.ndarray, face: object) -> np.ndarray:
        """Return the 512-d normalized embedding for a detected face."""

    def add_to_gallery(self, person: str, frame_bgr: np.ndarray, face: object) -> None:
        """Add an embedding for `person` to the in-memory gallery."""

    def save_gallery(self, path: Path) -> None:
        """Persist gallery embeddings to disk."""

    def load_gallery(self, path: Path) -> None:
        """Load gallery embeddings from disk if the file exists."""

    def match(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Match an embedding against the gallery.

        Returns:
            (best_person_name or None, cosine_similarity in [-1, 1]).
        """

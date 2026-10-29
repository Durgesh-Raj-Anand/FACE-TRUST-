"""Face recognition via InsightFace: detection + embedding + gallery matching.

One class does everything on single BGR image arrays, so uploaded photos and
webcam frames go through the exact same code path.

Embeddings are 512-d vectors; identity matching uses cosine similarity
(normalized embeddings -> dot product).
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from . import config

# Default gallery location: project_root/data
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
GALLERY_PATH: Path = PROJECT_ROOT / "data" / "gallery_embeddings.pkl"
GALLERY_DIR: Path = PROJECT_ROOT / "data" / "gallery"

VALID_IMAGE_EXT: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")


class FaceRecognizer:
    """Thin wrapper around InsightFace FaceAnalysis (CPU) + a small gallery."""

    def __init__(self, gallery_path: Path = GALLERY_PATH) -> None:
        """Load the InsightFace model once and start with an empty gallery.

        Args:
            gallery_path: where the gallery pickle is stored / loaded from.
        """
        # Import here so importing this module stays fast and testable.
        from insightface.app import FaceAnalysis

        self.gallery_path = gallery_path
        self.gallery: Dict[str, np.ndarray] = {}

        # Model pack name comes from src/config.py (e.g. "buffalo_l").
        # Models are downloaded once to ~/.insightface/models on first run.
        self.app = FaceAnalysis(
            name=config.MODEL_NAME,
            providers=config.DET_PROVIDERS,
        )
        self.app.prepare(ctx_id=-1, det_size=(config.DET_PX, config.DET_PX))  # CPU

    # ---------------------------------------------------------------- faces
    def _largest_face(self, image_bgr: np.ndarray) -> Optional[Any]:
        """Return the InsightFace Face object with the biggest bbox, or None."""
        faces = self.app.get(image_bgr)
        if not faces:
            return None

        def area(face: Any) -> float:
            x1, y1, x2, y2 = face.bbox
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)

        return max(faces, key=area)

    # ------------------------------------------------------------------ I/O
    def build_gallery(self, gallery_dir: Path = GALLERY_DIR) -> Dict[str, np.ndarray]:
        """Build the gallery from data/gallery/<person>/*.jpg and save it.

        For every person folder: extract the largest face's embedding from each
        valid image, average them, L2-normalize the centroid, store under the
        person's name.

        Returns:
            The gallery dict {person_name: (512,) normalized embedding}.
        """
        gallery_dir = Path(gallery_dir)
        gallery: Dict[str, np.ndarray] = {}

        for person_dir in sorted(gallery_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            embeddings: list[np.ndarray] = []
            for img_path in sorted(person_dir.rglob("*")):
                if img_path.suffix.lower() not in VALID_IMAGE_EXT:
                    continue
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"[gallery] could not read {img_path}, skipped")
                    continue
                face = self._largest_face(image)
                if face is None or face.embedding is None:
                    print(f"[gallery] no face in {img_path.name}, skipped")
                    continue
                embeddings.append(self._normalize(face.embedding))

            if not embeddings:
                print(f"[gallery] WARNING: no usable faces for '{person_dir.name}'")
                continue

            # Average then re-normalize, so the centroid stays a unit vector.
            centroid = np.mean(np.stack(embeddings), axis=0)
            gallery[person_dir.name] = self._normalize(centroid)
            print(f"[gallery] {person_dir.name}: {len(embeddings)} image(s) enrolled")

        self.gallery = gallery
        self.save_gallery()
        return gallery

    def save_gallery(self) -> None:
        """Persist the in-memory gallery to a pickle (no-op warns if empty)."""
        if not self.gallery:
            print("[gallery] WARNING: gallery is empty, nothing saved")
            return
        with open(self.gallery_path, "wb") as fh:
            pickle.dump(self.gallery, fh)
        print(f"[gallery] saved {len(self.gallery)} person(s) -> {self.gallery_path}")

    def load_gallery(self) -> Dict[str, np.ndarray]:
        """Load the gallery pickle if it exists; leave gallery empty otherwise."""
        if not self.gallery_path.exists():
            print(f"[gallery] no gallery file at {self.gallery_path}; run build_gallery() first")
            return {}
        with open(self.gallery_path, "rb") as fh:
            self.gallery = pickle.load(fh)
        print(f"[gallery] loaded {len(self.gallery)} person(s) from {self.gallery_path}")
        return self.gallery

    # ------------------------------------------------------------ identify
    def identify(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """Detect the largest face in one image and match it to the gallery.

        Same function for uploaded photos, snapshots and webcam frames.

        Returns:
            {"face_found": bool, "bbox", "landmarks", "det_score",
             "identity", "similarity", "second_best_similarity",
             "embedding", "insightface_pose"}
            bbox = [x1, y1, x2, y2]; identity is None if the gallery is empty;
            similarity is the best cosine similarity, second_best_similarity the
            runner-up (None if only one person is enrolled).
        """
        result: Dict[str, Any] = {
            "face_found": False,
            "bbox": None,
            "landmarks": None,
            "det_score": None,
            "identity": None,
            "similarity": None,
            "second_best_similarity": None,
            "embedding": None,
            "insightface_pose": None,
        }

        face = self._largest_face(image_bgr)
        if face is None:
            return result  # no face detected; the rest stays None/False

        result.update(
            face_found=True,
            bbox=[float(v) for v in face.bbox],
            landmarks=[(float(x), float(y)) for x, y in face.kps],  # 5 keypoints
            det_score=float(face.det_score),
            insightface_pose=[float(v) for v in face.pose],  # raw model pose values
        )

        if face.embedding is not None:
            embedding = self._normalize(face.embedding)
            result["embedding"] = embedding
            if self.gallery:
                result["identity"], result["similarity"], result["second_best_similarity"] = self._cosine_match(
                    embedding
                )
        return result

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        """L2-normalize an embedding so cosine similarity becomes a dot product."""
        norm = float(np.linalg.norm(embedding))
        return embedding / norm if norm > 0 else embedding

    def _cosine_match(self, embedding: np.ndarray) -> tuple[Optional[str], float, Optional[float]]:
        """Match an embedding against the gallery by cosine similarity.

        Returns:
            (best_person, best_similarity, second_best_similarity_or_None).
        """
        sims: list[tuple[str, float]] = [
            (name, float(np.dot(embedding, ref))) for name, ref in self.gallery.items()
        ]
        sims.sort(key=lambda item: item[1], reverse=True)
        best_name, best_sim = sims[0]
        second_best = sims[1][1] if len(sims) > 1 else None
        return best_name, best_sim, second_best


# ---------------------------------------------------------------- CLI test
def _draw_and_show(frame: np.ndarray, result: Dict[str, Any]) -> None:
    """Overlay the bbox + identity on a frame (demo helper, no module logic)."""
    if result["face_found"]:
        x1, y1, x2, y2 = [int(v) for v in result["bbox"]]
        label = f"{result['identity'] or 'unknown'} {result['similarity']:.2f}" if result["similarity"] is not None else "no gallery"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)


def main() -> int:
    """CLI: --image for a single photo test, --webcam for live identify (q quits)."""
    parser = argparse.ArgumentParser(description="FaceRecognizer quick test")
    parser.add_argument("--image", type=str, help="path to a test image")
    parser.add_argument("--webcam", action="store_true", help="live webcam test")
    parser.add_argument("--build-gallery", action="store_true", help="rebuild data/gallery_embeddings.pkl")
    args = parser.parse_args()

    recognizer = FaceRecognizer()
    if args.build_gallery or not GALLERY_PATH.exists():
        recognizer.build_gallery()
    else:
        recognizer.load_gallery()

    if args.webcam:
        camera = cv2.VideoCapture(config.CAMERA_INDEX)
        if not camera.isOpened():
            print(f"ERROR: could not open camera {config.CAMERA_INDEX}")
            return 2
        frame_idx, cached = 0, None
        print("Webcam live: processing every", config.PROCESS_EVERY_N_FRAMES, "frames. Press 'q' to quit.")
        while True:
            ok, frame = camera.read()
            if not ok:
                print("ERROR: could not read frame")
                break
            if frame_idx % config.PROCESS_EVERY_N_FRAMES == 0:  # full analyze on the 5th frame
                cached = recognizer.identify(frame)
            if cached is not None:
                _draw_and_show(frame, cached)
            cv2.imshow("FaceRecognizer --webcam (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_idx += 1
        camera.release()
        cv2.destroyAllWindows()
        return 0

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            print(f"ERROR: could not read image {args.image}")
            return 2
        result = recognizer.identify(image)
        print(result)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

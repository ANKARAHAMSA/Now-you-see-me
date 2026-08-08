"""
core/face_recognizer.py — DeepFace + ArcFace + Grassmann Subspace Matching

Multi-frame face recognition pipeline:
1. Detect faces using RetinaFace (most accurate detector)
2. Extract ArcFace embeddings for each detected face
3. Accumulate embeddings over N frames (Grassmann approach)
4. Compute subspace distance via SVD for robust identification
5. Match against enrolled known-faces database
"""

from __future__ import annotations

import os
import json
import pickle
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import numpy as np
from utils.logger import get_logger

logger = get_logger("face_recognizer")

KNOWN_FACES_DIR = Path("database/known_faces")
EMBEDDINGS_DB = Path("database/face_embeddings.pkl")


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two embedding vectors."""
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(1.0 - np.dot(a, b))


def _grassmann_distance(embeddings_a: list[np.ndarray], embeddings_b: list[np.ndarray]) -> float:
    """
    Compute Grassmann manifold distance between two sets of embeddings.

    Each set of embeddings forms a linear subspace (Grassmann point).
    We compute the principal angles between the two subspaces using SVD.

    Returns a scalar distance in [0, 1].
    """
    # Stack into matrices (n_frames × embedding_dim)
    A = np.stack(embeddings_a, axis=0)   # shape: (Na, D)
    B = np.stack(embeddings_b, axis=0)   # shape: (Nb, D)

    # Orthonormal basis via SVD
    Qa, _ = np.linalg.qr(A.T)  # shape: (D, Na)
    Qb, _ = np.linalg.qr(B.T)  # shape: (D, Nb)

    # Principal angles via SVD of Qa^T @ Qb
    M = Qa.T @ Qb
    singular_values = np.linalg.svd(M, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)

    # Geodesic distance on Grassmann manifold
    principal_angles = np.arccos(singular_values)
    return float(np.sqrt(np.sum(principal_angles ** 2)))


class FaceRecognizer:
    """
    Face recognition engine combining DeepFace embeddings with
    Grassmann multi-frame subspace matching.
    """

    def __init__(self, config: dict):
        rec_cfg = config.get("face_recognition", {})
        self.model_name: str = rec_cfg.get("model", "ArcFace")
        self.detector_backend: str = rec_cfg.get("detector_backend", "retinaface")
        self.match_threshold: float = float(
            os.getenv("FACE_MATCH_THRESHOLD", rec_cfg.get("match_threshold", 0.4))
        )
        self.grassmann_frames: int = int(
            os.getenv("GRASSMANN_FRAME_COUNT", rec_cfg.get("grassmann_frames", 10))
        )
        self.min_face_size: int = rec_cfg.get("min_face_size", 40)

        # Per-track embedding buffer for Grassmann matching
        # track_id → deque of embedding arrays
        self._embedding_buffer: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.grassmann_frames)
        )

        # Known faces database: name → list of embedding arrays
        self._known_db: dict[str, list[np.ndarray]] = {}
        self._load_embeddings()
        logger.info(
            f"Face recognizer ready | model={self.model_name} | "
            f"backend={self.detector_backend} | threshold={self.match_threshold}"
        )

    # ─── Database Management ─────────────────────────────────────────────────

    def _load_embeddings(self):
        """Load pre-computed embeddings from pickle cache."""
        if EMBEDDINGS_DB.exists():
            try:
                with open(EMBEDDINGS_DB, "rb") as f:
                    self._known_db = pickle.load(f)
                logger.info(f"Loaded embeddings for {len(self._known_db)} known persons")
            except Exception as e:
                logger.error(f"Failed to load embeddings: {e}")
                self._known_db = {}
        else:
            logger.warning("No face embeddings DB found. Run enroll_face.py first.")

    def save_embeddings(self):
        """Persist current embeddings database."""
        EMBEDDINGS_DB.parent.mkdir(parents=True, exist_ok=True)
        with open(EMBEDDINGS_DB, "wb") as f:
            pickle.dump(self._known_db, f)
        logger.info(f"Saved embeddings for {len(self._known_db)} persons")

    def enroll_person(self, name: str, image_paths: list[str]) -> int:
        """
        Enroll a new person by computing embeddings from their images.
        Uses 'skip' detector since faces are already cropped during capture.

        Args:
            name: Person's name.
            image_paths: List of image file paths.

        Returns:
            Number of successfully processed images.
        """
        from deepface import DeepFace

        embeddings = []
        total = len(image_paths)
        print(f"  Processing {total} images with ArcFace...")

        for i, path in enumerate(image_paths, 1):
            print(f"  [{i}/{total}] Encoding image...", end="\r")
            try:
                result = DeepFace.represent(
                    img_path=path,
                    model_name=self.model_name,
                    detector_backend="skip",   # Face already cropped — skip detection
                    enforce_detection=False,
                    align=False,
                )
                if result:
                    embeddings.append(np.array(result[0]["embedding"]))
                    print(f"  [{i}/{total}] ✓ Encoded")
            except Exception as e:
                logger.warning(f"Skipping {path}: {e}")

        if embeddings:
            # APPEND to existing embeddings (don't overwrite!)
            existing = self._known_db.get(name, [])
            merged = existing + embeddings
            # Keep max 30 embeddings (most recent) to avoid DB bloat
            self._known_db[name] = merged[-30:]
            self.save_embeddings()
            total_count = len(self._known_db[name])
            logger.info(f"✓ Enrolled '{name}' with {len(embeddings)} new + {len(existing)} existing = {total_count} total embedding(s)")
        return len(embeddings)

    # ─── Recognition ─────────────────────────────────────────────────────────

    def extract_embedding(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract ArcFace embedding from a pre-cropped 224×224 face image.

        Args:
            face_crop: BGR face image (already cropped and resized by caller).

        Returns:
            Embedding array or None if extraction fails.
        """
        if face_crop is None or face_crop.size == 0:
            return None
        h, w = face_crop.shape[:2]
        if h < self.min_face_size or w < self.min_face_size:
            return None

        try:
            import cv2
            from deepface import DeepFace
            
            # Convert BGR (OpenCV format) to RGB (DeepFace expected format)
            if isinstance(face_crop, np.ndarray):
                face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            else:
                face_crop_rgb = face_crop

            result = DeepFace.represent(
                img_path=face_crop_rgb,
                model_name=self.model_name,
                detector_backend="skip",   # Already a face crop
                enforce_detection=False,
                align=False,
            )
            if result:
                return np.array(result[0]["embedding"])
        except Exception as e:
            logger.debug(f"Embedding extraction failed: {e}")
        return None

    def identify(
        self,
        embedding: np.ndarray,
        track_id: Optional[int] = None,
    ) -> tuple[str, float]:
        """
        Identify a person from their embedding.

        Compares query embedding against enrolled embeddings using Cosine Distance.
        ArcFace standard cosine distance threshold: 0.68 (or configurable).

        Args:
            embedding: ArcFace embedding array.
            track_id: Optional tracker ID for multi-frame buffer smoothing.

        Returns:
            (name, distance) — name is 'UNKNOWN' if no match.
        """
        if not self._known_db:
            return "UNKNOWN", 1.0

        # Buffer embedding for temporal smoothing if track_id provided
        if track_id is not None:
            self._embedding_buffer[track_id].append(embedding)
            # Use mean embedding over recent buffered frames for temporal stability
            recent_embs = list(self._embedding_buffer[track_id])
            query_emb = np.mean(recent_embs, axis=0)
        else:
            query_emb = embedding

        best_name = "UNKNOWN"
        best_dist = float("inf")

        # Compare query against all enrolled embeddings
        for name, known_embeddings in self._known_db.items():
            for known_emb in known_embeddings:
                dist = _cosine_distance(query_emb, known_emb)
                if dist < best_dist:
                    best_dist = dist
                    best_name = name

        # Strict ArcFace matching threshold (0.42)
        threshold = self.match_threshold
        
        logger.info(f"Face match check: best='{best_name}' dist={best_dist:.3f} (thresh={threshold})")

        if best_dist > threshold:
            return "UNKNOWN", best_dist

        return best_name, best_dist

    def _cosine_identify(self, embedding: np.ndarray) -> tuple[str, float]:
        """Match embedding against known database using cosine distance."""
        best_name = "UNKNOWN"
        best_dist = float("inf")

        for name, known_embeddings in self._known_db.items():
            for known_emb in known_embeddings:
                dist = _cosine_distance(embedding, known_emb)
                if dist < best_dist:
                    best_dist = dist
                    best_name = name

        if best_dist > self.match_threshold:
            return "UNKNOWN", best_dist
        return best_name, best_dist

    def _grassmann_identify(self, query_embeddings: list[np.ndarray]) -> tuple[str, float]:
        """Match a sequence of embeddings using Grassmann manifold distance."""
        best_name = "UNKNOWN"
        best_dist = float("inf")

        for name, known_embeddings in self._known_db.items():
            if len(known_embeddings) < 2:
                # Fall back to cosine for persons with only 1 image
                avg_emb = np.mean(query_embeddings, axis=0)
                dist = _cosine_distance(avg_emb, np.array(known_embeddings[0]))
            else:
                dist = _grassmann_distance(query_embeddings, known_embeddings)

            if dist < best_dist:
                best_dist = dist
                best_name = name

        # Grassmann distances have different scale — use a looser threshold
        grassmann_threshold = self.match_threshold * 2.0
        if best_dist > grassmann_threshold:
            return "UNKNOWN", best_dist
        return best_name, best_dist

    def clear_buffer(self, track_id: int):
        """Clear embedding buffer for a specific track ID."""
        if track_id in self._embedding_buffer:
            del self._embedding_buffer[track_id]

    @property
    def known_persons(self) -> list[str]:
        """Return list of enrolled person names."""
        return list(self._known_db.keys())

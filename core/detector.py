"""
core/detector.py — YOLOv8 Multi-Class Object Detector

Detects persons, animals, and vehicles in real-time using
the Ultralytics YOLOv8 model with configurable confidence thresholds.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger("detector")

# Suppress YOLO verbose output
os.environ["YOLO_VERBOSE"] = "False"


@dataclass
class Detection:
    """Single detection result."""
    bbox: tuple[int, int, int, int]       # (x1, y1, x2, y2)
    class_name: str
    confidence: float
    category: str = "unknown"             # 'person', 'animal', 'vehicle'
    is_harmful_animal: bool = False
    track_id: Optional[int] = None


class YOLODetector:
    """
    Wraps Ultralytics YOLOv8 for multi-class detection.
    Auto-downloads the model on first run.
    """

    ANIMAL_CLASSES = {
        "dog", "cat", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "bird",
    }
    HARMFUL_ANIMALS = {"bear", "elephant", "snake", "wolf"}
    VEHICLE_CLASSES = {"car", "motorcycle", "bicycle", "bus", "truck"}
    PERSON_CLASSES = {"person"}

    def __init__(self, config: dict):
        from ultralytics import YOLO

        det_cfg = config.get("detection", {})
        model_name = os.getenv("YOLO_MODEL", "yolov8n.pt")
        model_path = Path("models") / model_name

        logger.info(f"Loading YOLOv8 model: {model_name}")
        self._model = YOLO(str(model_path) if model_path.exists() else model_name)
        logger.info("YOLOv8 model ready ✓")

        self.confidence: float = float(os.getenv("DETECTION_CONFIDENCE", det_cfg.get("confidence_threshold", 0.5)))

        # Override class sets from config if provided
        if det_cfg.get("animal_classes"):
            self.ANIMAL_CLASSES = set(det_cfg["animal_classes"])
        if det_cfg.get("harmful_animals"):
            self.HARMFUL_ANIMALS = set(det_cfg["harmful_animals"])

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run YOLOv8 inference on a single frame.

        Args:
            frame: BGR numpy array.

        Returns:
            List of Detection objects.
        """
        results = self._model(frame, imgsz=640, conf=self.confidence, verbose=False)
        detections: list[Detection] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = self._model.names[cls_id].lower()
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                category, is_harmful = self._classify(class_name)
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    class_name=class_name,
                    confidence=conf,
                    category=category,
                    is_harmful_animal=is_harmful,
                ))

        return detections

    def _classify(self, class_name: str) -> tuple[str, bool]:
        """Map YOLO class name to our category system."""
        if class_name in self.PERSON_CLASSES:
            return "person", False
        if class_name in self.ANIMAL_CLASSES:
            return "animal", class_name in self.HARMFUL_ANIMALS
        if class_name in self.VEHICLE_CLASSES:
            return "vehicle", False
        return "object", False

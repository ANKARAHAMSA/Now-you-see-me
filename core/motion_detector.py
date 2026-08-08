"""
core/motion_detector.py — MOG2 Background Subtraction Motion Gating

Only activates expensive AI pipelines (YOLO + face recognition) when
significant motion is detected, dramatically reducing CPU/GPU usage.
"""

import cv2
import numpy as np
from utils.logger import get_logger

logger = get_logger("motion_detector")


class MotionDetector:
    """
    Motion detector using Gaussian Mixture Model background subtraction.
    Acts as a gate: detection pipeline only runs when motion is present.
    """

    def __init__(self, config: dict):
        cfg = config.get("motion", {})
        self.enabled: bool = cfg.get("enabled", True)
        self.min_area: int = cfg.get("min_area", 5000)

        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=cfg.get("history", 500),
            varThreshold=cfg.get("var_threshold", 50),
            detectShadows=cfg.get("detect_shadows", True),
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.is_active: bool = False
        logger.info(f"Motion detector initialized (min_area={self.min_area})")

    def update(self, frame: np.ndarray) -> bool:
        """
        Process a frame and return True if motion above threshold is detected.

        Args:
            frame: BGR frame from camera.

        Returns:
            True if motion is detected, False otherwise.
        """
        if not self.enabled:
            self.is_active = True
            return True

        # Apply background subtractor
        fg_mask = self._subtractor.apply(frame)

        # Remove shadows (gray pixels → 0)
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

        # Morphological cleanup
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._kernel)
        fg_mask = cv2.dilate(fg_mask, self._kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if any contour exceeds minimum area
        motion_detected = any(cv2.contourArea(c) >= self.min_area for c in contours)
        self.is_active = motion_detected
        return motion_detected

    def get_motion_mask(self, frame: np.ndarray) -> np.ndarray:
        """Return the foreground mask for visualization."""
        fg_mask = self._subtractor.apply(frame, learningRate=0)
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        return fg_mask

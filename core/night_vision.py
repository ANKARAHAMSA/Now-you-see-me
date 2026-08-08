"""
core/night_vision.py — CLAHE + Gamma Correction for Low-Light Enhancement

Automatically enhances frames when ambient brightness is low,
ensuring reliable detection in night or poorly lit environments.
"""

import cv2
import numpy as np
from utils.logger import get_logger

logger = get_logger("night_vision")


class NightVisionEnhancer:
    """
    Enhances low-light frames using CLAHE (Contrast Limited Adaptive
    Histogram Equalization) on the L channel of LAB color space,
    followed by optional gamma correction.
    """

    def __init__(self, config: dict):
        cfg = config.get("night_vision", {})
        self.enabled: bool = cfg.get("enabled", True)
        self.brightness_threshold: int = cfg.get("brightness_threshold", 80)
        self.gamma: float = cfg.get("gamma", 1.5)
        self.active: bool = False

        tile_grid = cfg.get("clahe_tile_grid", [8, 8])
        self._clahe = cv2.createCLAHE(
            clipLimit=cfg.get("clahe_clip_limit", 3.0),
            tileGridSize=tuple(tile_grid),
        )
        # Precompute gamma lookup table
        self._gamma_table = self._build_gamma_table(self.gamma)
        logger.info(
            f"Night vision initialized (threshold={self.brightness_threshold}, gamma={self.gamma})"
        )

    def _build_gamma_table(self, gamma: float) -> np.ndarray:
        inv_gamma = 1.0 / gamma
        table = np.array(
            [(i / 255.0) ** inv_gamma * 255 for i in range(256)],
            dtype=np.uint8,
        )
        return table

    def _is_dark(self, frame: np.ndarray) -> bool:
        """Return True if frame mean brightness is below threshold."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(gray.mean()) < self.brightness_threshold

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance frame if darkness is detected.

        Args:
            frame: Input BGR frame.

        Returns:
            Enhanced BGR frame (same size).
        """
        if not self.enabled:
            self.active = False
            return frame

        if not self._is_dark(frame):
            self.active = False
            return frame

        self.active = True

        # Convert to LAB and apply CLAHE on L channel
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_chan = self._clahe.apply(l_chan)
        enhanced_lab = cv2.merge([l_chan, a_chan, b_chan])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Apply gamma correction for very dark scenes
        if cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean() < 40:
            enhanced = cv2.LUT(enhanced, self._gamma_table)

        return enhanced

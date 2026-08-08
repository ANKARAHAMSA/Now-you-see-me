"""
tests/test_night_vision.py — Unit test for CLAHE Night Vision Enhancer
"""

import sys
import unittest
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.night_vision import NightVisionEnhancer


class TestNightVision(unittest.TestCase):

    def setUp(self):
        self.enhancer = NightVisionEnhancer({"night_vision": {"brightness_threshold": 80}})

    def test_dark_frame_enhancement(self):
        """Test that dark frames are enhanced properly."""
        dark_frame = np.ones((200, 200, 3), dtype=np.uint8) * 30
        enhanced = self.enhancer.enhance(dark_frame)
        self.assertTrue(self.enhancer.active)
        self.assertEqual(enhanced.shape, dark_frame.shape)

    def test_bright_frame_passthrough(self):
        """Test that bright frames pass through unenhanced."""
        bright_frame = np.ones((200, 200, 3), dtype=np.uint8) * 180
        enhanced = self.enhancer.enhance(bright_frame)
        self.assertFalse(self.enhancer.active)


if __name__ == "__main__":
    unittest.main()

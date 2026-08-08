"""
tests/test_face_recognizer.py — Unit tests for ArcFace Face Recognizer & Anti-Spoofing
"""

import sys
import unittest
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import load_config
from core.face_recognizer import FaceRecognizer


class TestFaceRecognizer(unittest.TestCase):

    def setUp(self):
        self.config = load_config()
        self.recognizer = FaceRecognizer(self.config)

    def test_liveness_detection_valid_crop(self):
        """Test liveness checker with synthetic valid face crop."""
        synthetic_crop = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        is_live, score, reason = self.recognizer.check_liveness(synthetic_crop)
        self.assertTrue(is_live)
        self.assertGreater(score, 0.0)

    def test_liveness_detection_empty_crop(self):
        """Test liveness checker handles empty crops safely."""
        empty_crop = np.array([], dtype=np.uint8)
        is_live, score, reason = self.recognizer.check_liveness(empty_crop)
        self.assertFalse(is_live)
        self.assertEqual(reason, "Empty crop")


if __name__ == "__main__":
    unittest.main()

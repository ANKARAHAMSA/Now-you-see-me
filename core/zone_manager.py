"""
core/zone_manager.py — Restricted Zone Polygon Management

Loads zone definitions from config/zones.json and provides
point-in-polygon checks for detection centroids.
"""

import json
import numpy as np
import cv2
from pathlib import Path
from utils.logger import get_logger

logger = get_logger("zone_manager")

ZONES_FILE = Path("config/zones.json")


class ZoneManager:
    """Manages restricted zone polygons and zone-violation checks."""

    def __init__(self):
        self.zones: list[dict] = []
        self._load_zones()

    def _load_zones(self):
        """Load zone definitions from JSON config."""
        if not ZONES_FILE.exists():
            logger.warning(f"Zones file not found: {ZONES_FILE}")
            return
        try:
            with open(ZONES_FILE) as f:
                data = json.load(f)
            self.zones = data.get("zones", [])
            logger.info(f"Loaded {len(self.zones)} restricted zones")
        except Exception as e:
            logger.error(f"Failed to load zones: {e}")

    def reload(self):
        """Hot-reload zone definitions."""
        self._load_zones()

    def get_violated_zones(self, bbox: tuple[int, int, int, int]) -> list[dict]:
        """
        Check which zones the given bounding box centroid falls into.

        Args:
            bbox: (x1, y1, x2, y2) bounding box.

        Returns:
            List of violated zone dicts.
        """
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        point = (cx, cy)

        violated = []
        for zone in self.zones:
            pts = np.array(zone["points"], dtype=np.int32)
            result = cv2.pointPolygonTest(pts, point, False)
            if result >= 0:
                violated.append(zone)
        return violated

    def is_in_any_zone(self, bbox: tuple[int, int, int, int]) -> bool:
        """Quick check — returns True if centroid is inside any zone."""
        return len(self.get_violated_zones(bbox)) > 0

    def get_highest_priority(self, violated_zones: list[dict]) -> str:
        """Return the highest alert priority from violated zones."""
        priority_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        if not violated_zones:
            return "NONE"
        return max(violated_zones, key=lambda z: priority_order.get(z.get("alert_priority", "LOW"), 0)).get("alert_priority", "LOW")

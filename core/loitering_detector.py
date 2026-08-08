"""
core/loitering_detector.py — Dwell-Time Analysis for Suspicious Behavior

Tracks how long each detected person (by tracker ID) has been present
in the frame or restricted zone. Raises alerts when dwell time
exceeds the configured threshold.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from utils.logger import get_logger

logger = get_logger("loitering")


@dataclass
class DwellRecord:
    """Tracks presence of a single detected entity."""
    track_id: int
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    alert_sent: bool = False

    @property
    def dwell_seconds(self) -> float:
        return self.last_seen - self.first_seen

    def touch(self):
        self.last_seen = time.time()


class LoiteringDetector:
    """
    Maintains per-ID dwell time records and flags loitering when
    an unknown person exceeds the configured threshold.
    """

    def __init__(self, config: dict):
        cfg = config.get("loitering", {})
        self.enabled: bool = cfg.get("enabled", True)
        self.threshold_seconds: float = cfg.get("threshold_seconds", 30.0)
        self.alert_interval: float = cfg.get("alert_interval_seconds", 60.0)
        self._records: dict[int, DwellRecord] = {}
        self._cleanup_interval: float = 10.0
        self._last_cleanup: float = time.time()
        logger.info(
            f"Loitering detector initialized (threshold={self.threshold_seconds}s)"
        )

    def update(self, track_ids: list[int]) -> list[int]:
        """
        Update dwell records for currently visible track IDs.

        Args:
            track_ids: List of currently detected track IDs.

        Returns:
            List of track IDs that are loitering (dwell > threshold).
        """
        if not self.enabled:
            return []

        now = time.time()

        # Update or create records
        for tid in track_ids:
            if tid in self._records:
                self._records[tid].touch()
            else:
                self._records[tid] = DwellRecord(track_id=tid)

        # Identify loiterers
        loiterers = []
        for tid, record in self._records.items():
            if tid not in track_ids:
                continue
            if record.dwell_seconds >= self.threshold_seconds:
                # Only alert once per alert_interval
                time_since_last = now - record.first_seen
                if not record.alert_sent or (time_since_last % self.alert_interval < 1.5):
                    loiterers.append(tid)
                    record.alert_sent = True

        # Cleanup stale records
        if now - self._last_cleanup > self._cleanup_interval:
            stale = [
                tid for tid, rec in self._records.items()
                if now - rec.last_seen > 15.0
            ]
            for tid in stale:
                del self._records[tid]
            self._last_cleanup = now

        return loiterers

    def get_dwell_time(self, track_id: int) -> float:
        """Return current dwell time in seconds for a given track ID."""
        record = self._records.get(track_id)
        return record.dwell_seconds if record else 0.0

    def reset(self):
        """Clear all records."""
        self._records.clear()

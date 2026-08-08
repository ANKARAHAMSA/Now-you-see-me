"""
core/tracker.py — ByteTrack Object Tracker via Supervision

Assigns persistent IDs to detections across frames, enabling:
- Loitering detection (same ID tracked over time)
- Behavior analysis
- Identity-consistent face recognition
"""

from __future__ import annotations

import numpy as np
import supervision as sv
from utils.logger import get_logger
from core.detector import Detection

logger = get_logger("tracker")


class ObjectTracker:
    """Wraps supervision ByteTrack to assign persistent tracking IDs."""

    def __init__(self, config: dict):
        track_cfg = config.get("tracking", {})
        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_cfg.get("track_thresh", 0.5),
            lost_track_buffer=track_cfg.get("track_buffer", 30),
            minimum_matching_threshold=track_cfg.get("match_thresh", 0.8),
        )
        logger.info("ByteTrack tracker initialized ✓")

    def update(self, detections: list[Detection], frame_shape: tuple) -> list[Detection]:
        """
        Update tracker with current detections and assign track IDs.

        Args:
            detections: List of Detection objects from YOLODetector.
            frame_shape: (height, width) of the frame.

        Returns:
            Same detections with track_id populated.
        """
        if not detections:
            return detections

        # Build supervision Detections object
        xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
        conf = np.array([d.confidence for d in detections], dtype=np.float32)
        class_ids = np.zeros(len(detections), dtype=int)

        sv_dets = sv.Detections(
            xyxy=xyxy,
            confidence=conf,
            class_id=class_ids,
        )

        tracked = self._tracker.update_with_detections(sv_dets)

        # Map tracked IDs back to Detection objects by IoU overlap
        if tracked.tracker_id is not None:
            for i, det in enumerate(detections):
                if i < len(tracked.tracker_id):
                    det.track_id = int(tracked.tracker_id[i])

        return detections

    def reset(self):
        """Reset tracker state (call on camera reconnect)."""
        self._tracker.reset()
        logger.info("Tracker reset")

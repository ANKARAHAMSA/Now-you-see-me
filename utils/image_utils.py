"""
utils/image_utils.py — Drawing utilities for bounding boxes, labels, and overlays
"""

import cv2
import numpy as np
from typing import Tuple, Optional


# ─── Color Palette ───────────────────────────────────────────────────────────
COLORS = {
    "intruder": (0, 0, 255),       # Red — unknown person
    "known": (0, 200, 0),          # Green — recognized person
    "animal_harmful": (0, 0, 200), # Dark Red — dangerous animal
    "animal_safe": (200, 100, 0),  # Orange — harmless animal
    "vehicle": (200, 200, 0),      # Yellow — vehicle
    "zone_high": (0, 0, 255),      # Red zone
    "zone_medium": (0, 165, 255),  # Orange zone
    "zone_low": (0, 255, 255),     # Yellow zone
    "text_bg": (20, 20, 20),
    "fps": (0, 255, 100),
    "night_vision": (0, 255, 0),   # Green tint indicator
}


def draw_detection(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str,
    color_key: str = "intruder",
    confidence: Optional[float] = None,
    track_id: Optional[int] = None,
) -> np.ndarray:
    """Draw bounding box with label on frame."""
    x1, y1, x2, y2 = bbox
    color = COLORS.get(color_key, (255, 255, 255))

    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Build label text
    text = label
    if track_id is not None:
        text = f"[{track_id}] {text}"
    if confidence is not None:
        text = f"{text} {confidence:.0%}"

    # Draw text background
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)

    # Draw text
    cv2.putText(
        frame, text, (x1 + 3, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )
    return frame


def draw_zones(frame: np.ndarray, zones: list) -> np.ndarray:
    """Draw restricted zone polygons on frame."""
    overlay = frame.copy()
    for zone in zones:
        pts = np.array(zone["points"], dtype=np.int32)
        color = tuple(zone.get("color", [0, 0, 255]))
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(frame, [pts], True, color, 2)
        # Label
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(
            frame, zone["name"], (cx - 40, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draw FPS counter."""
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLORS["fps"], 2, cv2.LINE_AA
    )
    return frame


def draw_mode_indicator(frame: np.ndarray, night_mode: bool, motion_active: bool) -> np.ndarray:
    """Draw system mode indicators at top-right."""
    h, w = frame.shape[:2]
    indicators = []
    if night_mode:
        indicators.append(("NIGHT VISION", COLORS["night_vision"]))
    if motion_active:
        indicators.append(("MOTION", (0, 165, 255)))
    else:
        indicators.append(("IDLE", (100, 100, 100)))

    for i, (text, color) in enumerate(indicators):
        cv2.putText(
            frame, text, (w - 160, 25 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA
        )
    return frame


def draw_alert_banner(frame: np.ndarray, message: str, color_key: str = "intruder") -> np.ndarray:
    """Draw a flashing alert banner at the bottom of the frame."""
    h, w = frame.shape[:2]
    color = COLORS.get(color_key, (0, 0, 255))
    cv2.rectangle(frame, (0, h - 40), (w, h), color, -1)
    cv2.putText(
        frame, f"⚠  {message}", (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
    )
    return frame


def add_timestamp(frame: np.ndarray) -> np.ndarray:
    """Add current timestamp watermark to frame."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    h = frame.shape[0]
    cv2.putText(
        frame, ts, (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA
    )
    return frame

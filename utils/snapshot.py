"""
utils/snapshot.py — Save annotated detection snapshots to disk
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from utils.logger import get_logger

logger = get_logger("snapshot")

SNAPSHOT_DIR = Path("database/snapshots")
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def save_snapshot(frame: np.ndarray, event_type: str, label: str = "") -> str:
    """
    Save an annotated frame to disk.

    Args:
        frame: The annotated BGR frame to save.
        event_type: Category string (e.g. 'intruder', 'animal', 'loitering').
        label: Optional label for the filename.

    Returns:
        Absolute path string of the saved image.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:21]
    safe_label = label.replace(" ", "_")[:20] if label else "unknown"
    filename = f"{event_type}_{safe_label}_{ts}.jpg"
    path = SNAPSHOT_DIR / filename

    try:
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        logger.debug(f"Snapshot saved: {path}")
        return str(path.resolve())
    except Exception as e:
        logger.error(f"Failed to save snapshot: {e}")
        return ""


def get_recent_snapshots(n: int = 20) -> list[dict]:
    """Return the N most recent snapshots as a list of dicts."""
    files = sorted(SNAPSHOT_DIR.glob("*.jpg"), key=lambda f: f.stat().st_mtime, reverse=True)
    results = []
    for f in files[:n]:
        parts = f.stem.split("_")
        results.append({
            "path": str(f.resolve()),
            "event_type": parts[0] if parts else "unknown",
            "filename": f.name,
            "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
        })
    return results

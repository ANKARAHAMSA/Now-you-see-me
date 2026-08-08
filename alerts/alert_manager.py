"""
alerts/alert_manager.py — Alert Coordination with Cooldown & Deduplication

Prevents alert spam by enforcing per-event-type cooldown periods.
Orchestrates alarm, Telegram, snapshot, and database logging.
"""

from __future__ import annotations

import time
import os
from typing import Optional
import numpy as np

from utils.logger import get_logger
from utils.snapshot import save_snapshot
from alerts.alarm import AlarmSystem
from alerts.telegram_alert import TelegramAlerter
from database.db_manager import DatabaseManager

logger = get_logger("alert_manager")


class AlertManager:
    """
    Central alert coordinator.
    Enforces cooldown windows per event type and orchestrates all outputs.
    """

    def __init__(self, config: dict):
        cfg = config.get("alerts", {})
        self.cooldown: float = float(
            os.getenv("ALERT_COOLDOWN_SECONDS", cfg.get("cooldown_seconds", 30))
        )
        self.save_snapshots: bool = cfg.get("save_snapshots", True)
        self.play_alarm: bool = cfg.get("play_alarm", True)
        self.send_telegram: bool = cfg.get("send_telegram", True)

        self._last_alert: dict[str, float] = {}

        self._alarm = AlarmSystem()
        self._telegram = TelegramAlerter()
        self._db = DatabaseManager()

        logger.info(f"Alert manager ready (cooldown={self.cooldown}s)")

    def trigger(
        self,
        event_type: str,
        label: str,
        frame: Optional[np.ndarray] = None,
        confidence: float = 0.0,
        zone_name: str = "",
        priority: str = "HIGH",
        notes: str = "",
    ):
        """
        Trigger an alert if not in cooldown period.

        Args:
            event_type: Category ('intruder', 'animal', 'loitering', 'vehicle').
            label: Specific label (e.g., 'UNKNOWN', 'bear', 'Person ID 3').
            frame: Annotated frame to snapshot.
            confidence: Detection confidence score.
            zone_name: Zone where detection occurred.
            priority: Alert priority ('HIGH', 'MEDIUM', 'LOW').
            notes: Additional context notes.
        """
        cooldown_key = f"{event_type}:{label}"
        now = time.time()

        if now - self._last_alert.get(cooldown_key, 0) < self.cooldown:
            logger.debug(f"Alert suppressed (cooldown): {cooldown_key}")
            return

        self._last_alert[cooldown_key] = now

        # ── Snapshot ─────────────────────────────────────────────────────────
        snapshot_path = ""
        if self.save_snapshots and frame is not None:
            snapshot_path = save_snapshot(frame, event_type, label)

        # ── Build message ─────────────────────────────────────────────────────
        zone_str = f" in [{zone_name}]" if zone_name else ""
        time_str = time.strftime("%H:%M:%S")
        message = f"[{priority}] {event_type.upper()} DETECTED: {label}{zone_str} at {time_str}"
        if confidence > 0:
            message += f" (conf: {confidence:.0%})"

        logger.warning(f"🚨 ALERT: {message}")

        # ── Database log ─────────────────────────────────────────────────────
        self._db.log_event(
            event_type=event_type,
            label=label,
            confidence=confidence,
            zone_name=zone_name,
            priority=priority,
            snapshot=snapshot_path,
            notes=notes,
        )

        # ── Alarm ─────────────────────────────────────────────────────────────
        if self.play_alarm and priority == "HIGH":
            self._alarm.trigger(repeat=2 if priority == "HIGH" else 1)

        # ── Telegram ──────────────────────────────────────────────────────────
        if self.send_telegram:
            self._telegram.send_alert(message, snapshot_path or None)

    @property
    def db(self) -> DatabaseManager:
        return self._db

    @property
    def telegram(self) -> TelegramAlerter:
        return self._telegram

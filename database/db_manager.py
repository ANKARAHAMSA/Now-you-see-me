"""
database/db_manager.py — SQLite Event Logger

Records all detection events for audit, analytics, and the
Streamlit dashboard. Provides query interface for filtered log retrieval.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional
from utils.logger import get_logger

logger = get_logger("db_manager")

DB_PATH = Path("database/events.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    event_type  TEXT    NOT NULL,
    label       TEXT,
    confidence  REAL,
    zone_name   TEXT,
    priority    TEXT,
    snapshot    TEXT,
    notes       TEXT
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_event_type ON events(event_type);
"""


class DatabaseManager:
    """Thread-safe SQLite event log manager."""

    def __init__(self, db_path: Path | str = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()
        logger.info(f"Database ready: {self.db_path.resolve()}")

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._get_conn()
            conn.executescript(CREATE_TABLE_SQL + CREATE_INDEX_SQL)
            conn.commit()
            conn.close()

    def log_event(
        self,
        event_type: str,
        label: str = "",
        confidence: float = 0.0,
        zone_name: str = "",
        priority: str = "MEDIUM",
        snapshot: str = "",
        notes: str = "",
    ) -> int:
        """
        Insert a detection event into the database.

        Returns:
            Row ID of the inserted event.
        """
        timestamp = datetime.now().isoformat(timespec="seconds")
        sql = """
        INSERT INTO events (timestamp, event_type, label, confidence, zone_name, priority, snapshot, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(sql, (
                timestamp, event_type, label, round(confidence, 4),
                zone_name, priority, snapshot, notes
            ))
            conn.commit()
            row_id = cursor.lastrowid
            conn.close()
        logger.debug(f"Event logged: [{event_type}] {label} @ {timestamp}")
        return row_id

    def get_recent_events(self, limit: int = 100) -> list[dict]:
        """Retrieve the most recent N events, newest first."""
        sql = "SELECT * FROM events ORDER BY id DESC LIMIT ?"
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(sql, (limit,)).fetchall()
            conn.close()
        return [dict(row) for row in rows]

    def get_events_by_type(self, event_type: str, limit: int = 50) -> list[dict]:
        sql = "SELECT * FROM events WHERE event_type = ? ORDER BY id DESC LIMIT ?"
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(sql, (event_type, limit)).fetchall()
            conn.close()
        return [dict(row) for row in rows]

    def get_event_counts(self) -> dict:
        """Return counts grouped by event_type."""
        sql = "SELECT event_type, COUNT(*) as count FROM events GROUP BY event_type"
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(sql).fetchall()
            conn.close()
        return {row["event_type"]: row["count"] for row in rows}

    def get_today_events(self) -> list[dict]:
        today = datetime.now().strftime("%Y-%m-%d")
        sql = "SELECT * FROM events WHERE timestamp LIKE ? ORDER BY id DESC"
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(sql, (f"{today}%",)).fetchall()
            conn.close()
        return [dict(row) for row in rows]

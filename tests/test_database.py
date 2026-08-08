"""
tests/test_database.py — Unit test for SQLite Event Database Manager
"""

import sys
import os
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager


class TestDatabaseManager(unittest.TestCase):

    def setUp(self):
        self.test_db_path = Path("database/test_events.db")
        if self.test_db_path.exists():
            os.remove(self.test_db_path)
        self.db = DatabaseManager(str(self.test_db_path))

    def tearDown(self):
        if self.test_db_path.exists():
            os.remove(self.test_db_path)

    def test_log_and_retrieve_event(self):
        """Test logging an event and retrieving it from SQLite."""
        event_id = self.db.log_event(
            event_type="intruder",
            label="UNKNOWN (ID:1)",
            confidence=0.95,
            zone_name="Main Entrance",
            priority="HIGH",
            snapshot="database/snapshots/test.jpg"
        )
        self.assertIsNotNone(event_id)

        events = self.db.get_recent_events(limit=5)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "intruder")
        self.assertEqual(events[0]["priority"], "HIGH")


if __name__ == "__main__":
    unittest.main()

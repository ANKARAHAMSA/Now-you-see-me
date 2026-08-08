"""
alerts/alarm.py — Audio Alarm System

Plays an alarm sound when an intruder is detected.
Uses system beep as fallback if playsound is unavailable.
"""

import os
import sys
import threading
from pathlib import Path
from utils.logger import get_logger

logger = get_logger("alarm")

ALARM_FILE = Path("assets/alarm.wav")


class AlarmSystem:
    """Thread-safe audio alarm trigger."""

    def __init__(self):
        self._playing = False
        self._lock = threading.Lock()
        self._check_audio()

    def _check_audio(self):
        """Check if audio playback is available."""
        try:
            import playsound  # noqa: F401
            self._use_playsound = True
            logger.info("Audio: playsound available ✓")
        except ImportError:
            self._use_playsound = False
            logger.warning("playsound not available — using system beep fallback")

    def trigger(self, repeat: int = 3):
        """
        Trigger the alarm in a background thread.

        Args:
            repeat: Number of times to repeat the alarm sound.
        """
        if not self._playing:
            t = threading.Thread(target=self._play, args=(repeat,), daemon=True)
            t.start()

    def _play(self, repeat: int):
        with self._lock:
            self._playing = True
            try:
                for _ in range(repeat):
                    if self._use_playsound and ALARM_FILE.exists():
                        from playsound import playsound
                        playsound(str(ALARM_FILE.resolve()))
                    else:
                        self._system_beep()
            except Exception as e:
                logger.error(f"Alarm playback error: {e}")
                self._system_beep()
            finally:
                self._playing = False

    def _system_beep(self):
        """Cross-platform system beep."""
        try:
            if sys.platform == "darwin":
                os.system("afplay /System/Library/Sounds/Sosumi.aiff 2>/dev/null &")
            elif sys.platform == "win32":
                import winsound
                winsound.Beep(1000, 500)
            else:
                print("\a", end="", flush=True)
        except Exception:
            print("\a", end="", flush=True)

    def stop(self):
        """Stop alarm playback (best effort)."""
        self._playing = False

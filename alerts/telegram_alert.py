"""
alerts/telegram_alert.py — Async Telegram Bot Alert System

Sends annotated snapshot images and alert messages to a Telegram chat.
Runs in a background thread to never block the detection loop.
"""

from __future__ import annotations

import os
import asyncio
import threading
from pathlib import Path
from typing import Optional
from utils.logger import get_logger

logger = get_logger("telegram")


class TelegramAlerter:
    """
    Non-blocking Telegram alert sender.
    Runs asyncio event loop in a dedicated background thread.
    """

    def __init__(self):
        self._bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
        self._configured: bool = bool(self._bot_token and self._chat_id and
                                       self._bot_token != "YOUR_BOT_TOKEN_HERE")
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

        if self._configured:
            self._start_loop()
            logger.info("Telegram alerter ready ✓")
        else:
            logger.warning(
                "Telegram not configured — copy config/.env.example to .env and set credentials"
            )

    def _start_loop(self):
        """Start a dedicated asyncio event loop in a background thread."""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True
        )
        self._thread.start()

    def send_alert(
        self,
        message: str,
        snapshot_path: Optional[str] = None,
    ):
        """
        Schedule a Telegram alert (non-blocking).

        Args:
            message: Alert text to send.
            snapshot_path: Optional path to an image file to attach.
        """
        if not self._configured:
            logger.debug(f"[Telegram MOCK] {message}")
            return

        asyncio.run_coroutine_threadsafe(
            self._send_async(message, snapshot_path),
            self._loop,
        )

    async def _send_async(self, message: str, snapshot_path: Optional[str]):
        """Async implementation of Telegram message sending."""
        try:
            from telegram import Bot

            bot = Bot(token=self._bot_token)
            if snapshot_path and Path(snapshot_path).exists():
                with open(snapshot_path, "rb") as photo:
                    await bot.send_photo(
                        chat_id=self._chat_id,
                        photo=photo,
                        caption=f"🚨 {message}",
                    )
            else:
                await bot.send_message(
                    chat_id=self._chat_id,
                    text=f"🚨 {message}",
                )
            logger.info(f"Telegram alert sent: {message}")
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")

    def send_system_message(self, message: str):
        """Send a non-alert system status message."""
        if not self._configured:
            return
        asyncio.run_coroutine_threadsafe(
            self._send_async(message, None),
            self._loop,
        )

    @property
    def is_configured(self) -> bool:
        return self._configured

"""
test_telegram.py — Telegram Alert Setup & Test Utility

Helps set up Telegram bot notifications and tests sending instant alerts to your phone.
"""

import os
import sys
from pathlib import Path
import requests

ENV_FILE = Path(".env")
ENV_EXAMPLE = Path("config/.env.example")


def setup_and_test():
    print("═" * 60)
    print("  📱 Telegram Alert Setup & Connection Test")
    print("═" * 60)

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not bot_token or bot_token == "YOUR_BOT_TOKEN_HERE":
        print("\nFollow these 2 quick steps on Telegram:")
        print("  1. Open Telegram → Search '@BotFather' → Send '/newbot' → Copy Token")
        print("  2. Search '@userinfobot' on Telegram → Send any message → Copy ID\n")
        bot_token = input("  Enter Telegram Bot Token: ").strip()
        chat_id = input("  Enter Telegram Chat ID: ").strip()

    if not bot_token or not chat_id:
        print("\n✗ Bot token and Chat ID are required.")
        return

    print(f"\n📡 Testing connection to Telegram (Chat ID: {chat_id})...")

    # Send test text message via Telegram API
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": "🛡️ *Intruder Detection System*\n\n✅ Telegram alerts configured successfully! You will receive instant notifications with photos when an intruder or event is detected.",
        "parse_mode": "Markdown",
    }

    try:
        res = requests.post(url, json=payload, timeout=10)
        data = res.json()
        if res.status_code == 200 and data.get("ok"):
            print("\n🎉 SUCCESS! Test message sent to your Telegram phone app!")

            # Save to .env file
            env_content = f"""CAMERA_SOURCE=0
YOLO_MODEL=yolov8n.pt
DETECTION_CONFIDENCE=0.50
FACE_MATCH_THRESHOLD=0.38
TELEGRAM_BOT_TOKEN={bot_token}
TELEGRAM_CHAT_ID={chat_id}
"""
            ENV_FILE.write_text(env_content)
            print(f"✅ Saved credentials to {ENV_FILE.resolve()}")
            print("\nNow run `python3 main.py` — live alerts will be sent straight to your phone! 🚀")
        else:
            print(f"\n✗ Telegram API Error: {data.get('description', 'Failed')}")
            print("Please double check your Bot Token and Chat ID.")
    except Exception as e:
        print(f"\n✗ Connection Error: {e}")


if __name__ == "__main__":
    setup_and_test()

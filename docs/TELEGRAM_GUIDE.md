# 📱 Telegram Bot Setup & Connection Guide

This guide explains how to connect your **Intruder Detection System** to Telegram for instant mobile photo alerts.

---

## 🚀 Setup Instructions

### 1. Create your Telegram Bot
1. Open **Telegram** on your mobile phone or desktop.
2. Search for `@BotFather` and tap **Start**.
3. Send `/newbot` and choose a name (e.g. `MySecurityBot`).
4. Copy the HTTP API **Bot Token** provided.

### 2. Get your Chat ID
1. Search for `@userinfobot` on Telegram and tap **Start**.
2. Copy the numerical **`Id:`** displayed (e.g. `123456789`).

### 3. Run Automatic Setup Helper
Run the setup tool in your terminal:
```bash
python3 test_telegram.py
```
Paste your **Bot Token** and **Chat ID**. It will verify the connection and automatically update your `.env` configuration file!

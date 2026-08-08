<div align="center">

# 👁️ Now You See Me — Real-Time AI Security & Intruder Detection System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/Object_Detection-YOLOv8-000000?style=for-the-badge&logo=ultralytics&logoColor=white)](https://ultralytics.com)
[![DeepFace](https://img.shields.io/badge/Face_Recognition-ArcFace-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://github.com/serengil/deepface)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Telegram](https://img.shields.io/badge/Alerts-Telegram_Bot-26A5E4?style=for-the-badge&logo=telegram&logoColor=white)](docs/TELEGRAM_GUIDE.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

*⚡ Next-generation AI security suite featuring 60 FPS ArcFace face recognition, 2D FFT Face Anti-Spoofing, Armed/Disarmed security modes, CSV/Report exporter, Rapid 30-photo auto-enrollment, YOLOv8 multi-class detection, ByteTrack tracking, Telegram photo alerts, and an interactive Streamlit command center.*

[Features](#-key-features) • [Architecture](#-system-architecture) • [Quick Start](#-quick-start) • [Telegram Bot](#-telegram-bot-setup) • [Face Enrollment](#-face-enrollment) • [Web Dashboard](#-streamlit-dashboard) • [Testing](#-automated-unit-testing)

</div>

---

## 📸 Overview

**Now You See Me** turns any ordinary laptop camera, webcam, or IP security camera into an autonomous security sentinel. By combining ultra-fast **YOLOv8** object detection, **ByteTrack** multi-object tracking, **ArcFace** deep facial recognition, and **2D FFT Moiré Anti-Spoofing**, the system accurately distinguishes between authorized family/staff members and unknown intruders.

```text
       +-------------------------------------------------------------+
       |                  INPUT: Camera Feed (Webcam/IP)             |
       +-------------------------------------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |               Night Vision (CLAHE + Gamma Auto)             |
       +-------------------------------------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |               Motion Gating (MOG2 — 0% CPU Idle)            |
       +-------------------------------------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |               YOLOv8 Object & Person Detection              |
       +-------------------------------------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |               ByteTrack Multi-Object Tracking               |
       +-------------------------------------------------------------+
                  /                    |                    \
                 /                     |                     \
                v                      v                      v
     +-------------------+    +------------------+   +-------------------+
     | Face Recognition  |    | Anti-Spoofing    |   | Security Modes    |
     | (ArcFace + Async) |    | (2D FFT Moiré)   |   | (Armed/Disarmed)  |
     +-------------------+    +------------------+   +-------------------+
                \                      |                     /
                 \                     |                    /
                  v                    v                   v
       +-------------------------------------------------------------+
       |         Alert Engine (Telegram Photo + Audio Alarm + DB)    |
       +-------------------------------------------------------------+
```

---

## ✨ Key Features

- ⚡ **60 FPS Real-Time Performance:** Threaded async face recognition and 640px inference deliver a ultra-smooth camera feed.
- 👤 **ArcFace Facial Recognition:** High-precision 512-D face vector embeddings (match threshold: `0.38`) accurately identify registered household members.
- 👁️ **2D FFT Face Anti-Spoofing:** Analyzes high-frequency 2D Fourier Transform spectra and texture sharpness to detect paper prints or screen photo attacks, while ignoring mobile phones used naturally by real people.
- 🔒 **Armed / Disarmed / Scheduled Modes:** Switch security modes on the fly via the web dashboard or enable automated night-time arming (e.g. 22:00 to 06:00).
- ⚡ **Rapid 30-Photo Auto-Enrollment:** Captures 30 high-quality face crops in just **~4.5 seconds** (0.15s rapid-fire interval) for maximum training variation.
- 📥 **CSV & Security Incident Report Exporter:** Download complete audit logs and executive security incident summaries directly from the dashboard.
- 📱 **Async Telegram Notifications:** Instantly sends photo snapshots with annotated bounding boxes and alert metadata directly to your phone.
- 👥 **Interactive Enrolled Persons Gallery:** View primary photo thumbnails, expand full 30-photo galleries, and delete single photos with automatic real-time embedding rebuilding.
- 🌙 **Adaptive Night Vision:** Automatically activates CLAHE and gamma correction under poor lighting.

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment

```bash
# Clone the repository
git clone https://github.com/ANKARAHAMSA/Now-you-see-me.git
cd Now-you-see-me

# Create virtual environment
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Telegram & Credentials

Run the automated Telegram setup tool:

```bash
python3 test_telegram.py
```

- Enter your **Telegram Bot Token** (from `@BotFather`) and **Chat ID** (from `@userinfobot`).
- It will verify the connection, send a test notification to your phone, and save credentials to `.env`.

### 3. Enroll Your Face (30-Photo Rapid Capture)

Run the interactive face enrollment tool:

```bash
python3 enroll_face.py
```

- Type your name (e.g., `shahin`)
- Look at the camera — **rapidly captures 30 frames in ~4.5 seconds!**
- Embeddings are encoded using ArcFace and saved to `database/face_embeddings.pkl`.

### 4. Start Security Detection System

```bash
python3 main.py
```

> **Display Controls:**
> - Press **`Q`** or **`ESC`** to quit.
> - Press **`S`** to take a manual snapshot.

### 5. Launch Web Command Center

In a new terminal window:

```bash
streamlit run dashboard.py
```

Open your browser at `http://localhost:8501` to view live feeds, change security modes, and download security logs!

---

## 📱 Telegram Bot Setup

For a detailed step-by-step guide with screenshots, see [docs/TELEGRAM_GUIDE.md](docs/TELEGRAM_GUIDE.md).

```bash
# Verify your Telegram Bot configuration anytime:
python3 test_telegram.py
```

---

## 🧪 Automated Unit Testing

Run the full automated test suite:

```bash
python3 -m unittest discover tests
```

Tests included:
- `tests/test_face_recognizer.py`: Unit tests for ArcFace embeddings and liveness anti-spoofing.
- `tests/test_night_vision.py`: Unit tests for CLAHE low-light enhancement.
- `tests/test_database.py`: Unit tests for thread-safe SQLite event logging.

---

## 🛠️ Project Structure

```text
Now-you-see-me/
├── main.py                  # 🚀 Main detection & tracking engine
├── dashboard.py             # 📊 Streamlit web command center
├── enroll_face.py           # 👤 Rapid 30-photo face enrollment CLI
├── test_telegram.py         # 📱 Telegram connection setup & tester
├── requirements.txt         # 📦 Dependencies list
├── config/
│   ├── settings.json        # ⚙️ Detection thresholds & security modes
│   └── .env                 # 🔑 Private Telegram tokens & credentials
├── docs/
│   ├── ARCHITECTURE.md      # 🏗 Detailed technical pipeline design
│   └── TELEGRAM_GUIDE.md    # 📱 Telegram bot integration guide
├── tests/
│   ├── test_face_recognizer.py # 🧪 ArcFace & Anti-spoofing unit test
│   ├── test_night_vision.py    # 🧪 CLAHE night vision unit test
│   └── test_database.py        # 🧪 SQLite DB logger unit test
├── core/
│   ├── detector.py          # 🎯 YOLOv8 multi-class detection
│   ├── face_recognizer.py   # 🧠 ArcFace 512-D recognition & 2D FFT Anti-Spoofing
│   ├── tracker.py           # 🔄 ByteTrack multi-object tracking
│   ├── night_vision.py      # 🌙 CLAHE low-light enhancement
│   └── loitering_detector.py# ⏱️ Dwell-time monitoring
├── alerts/
│   ├── alert_manager.py     # 🔔 Alert orchestrator
│   └── telegram_alert.py    # 📱 Async Telegram Bot notifier
└── database/
    ├── db_manager.py        # 🗄️ SQLite event database logging
    └── known_faces/         # 📁 Enrolled face photos & embeddings
```

---

## ⚙️ Configuration Specs (`config/settings.json`)

| Setting | Default | Description |
|---|---|---|
| `confidence_threshold` | `0.50` | YOLOv8 object detection confidence cutoff |
| `match_threshold` | `0.38` | Strict ArcFace cosine distance threshold (lower = stricter) |
| `armed_mode` | `"ARMED"` | System security mode (`ARMED`, `DISARMED`, `SCHEDULED`) |
| `schedule_start` / `end` | `"22:00"` / `"06:00"` | Automated night-time arming hours |
| `detector_backend` | `"opencv"` | Fast face crop detector backend |
| `alert_cooldown` | `30.0` | Cooldown period between repetitive alerts (seconds) |

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.

<div align="center">
  <sub>Built with ❤️ using Python, OpenCV, YOLOv8, ArcFace, and Streamlit.</sub>
</div>

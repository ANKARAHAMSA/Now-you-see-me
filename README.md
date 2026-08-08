<div align="center">

# 👁️ Now You See Me — Real-Time AI Security & Intruder Detection System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/Object_Detection-YOLOv8-000000?style=for-the-badge&logo=ultralytics&logoColor=white)](https://ultralytics.com)
[![DeepFace](https://img.shields.io/badge/Face_Recognition-ArcFace-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://github.com/serengil/deepface)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

*⚡ Next-generation AI security suite featuring 60 FPS ArcFace face recognition, YOLOv8 multi-class detection, ByteTrack tracking, loitering analysis, restricted zone monitoring, async Telegram alerts, and an interactive Streamlit command center.*

[Features](#-key-features) • [Architecture](#-system-architecture) • [Quick Start](#-quick-start) • [Face Enrollment](#-face-enrollment) • [Web Dashboard](#-stream-dashboard) • [Configuration](#-configuration)

</div>

---

## 📸 Overview

**Now You See Me** turns any ordinary webcam or IP security camera into a smart, autonomous security sentinel. By combining ultra-fast **YOLOv8** object detection, **ByteTrack** multi-object tracking, and **ArcFace** deep facial recognition, the system accurately distinguishes between authorized family/staff members and unknown intruders.

```
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
    | Face Recognition  |    | Loitering Check  |   | Restricted Zones  |
    | (ArcFace + Async) |    | (Dwell-Time)     |   | (Polygon Overlay) |
    +-------------------+    +------------------+   +-------------------+
               \                      |                     /
                \                     |                    /
                 v                    v                   v
       +-------------------------------------------------------------+
       |            Alert Engine (Telegram + Audio Alarm + DB)       |
       +-------------------------------------------------------------+
```

---

## ✨ Key Features

- ⚡ **Zero-Lag High FPS Stream:** Threaded async face recognition and frame-interleaving deliver a smooth **30–60 FPS** camera feed.
- 👤 **ArcFace Facial Recognition:** High-accuracy 512-D face embeddings eliminate false positives and distinguish known individuals from unknown intruders.
- 🐶 **Animal & Vehicle Classification:** Automatically identifies animals (safe vs. dangerous like bears/elephants) and vehicles (cars, motorcycles).
- ⏱️ **Dwell-Time Loitering Alerts:** Monitors how long individuals stand around. Triggers high-priority loitering alarms after a configurable threshold (e.g. 30s).
- 📍 **Custom Polygon Restricted Zones:** Define custom restricted zones (e.g., Main Entrance, Perimeter). Entering restricted areas escalates alert priority instantly.
- 🌙 **Adaptive Night Vision:** Automatically activates CLAHE (Contrast Limited Adaptive Histogram Equalization) and gamma enhancement under low-light conditions.
- 📱 **Async Telegram Notifications:** Instantly sends photo snapshots with bounding box overlays and metadata directly to your Telegram chat without freezing the video feed.
- 📊 **Streamlit Command Center:** A sleek web dashboard to monitor live camera stats, review historical intruder logs, analyze analytics charts, and manage enrolled persons.

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

### 2. Configure Settings (Optional)

Copy the environment template:

```bash
cp config/.env.example .env
```

Edit `.env` to set your camera source (`0` for laptop webcam, or RTSP stream URL) and Telegram Bot credentials:

```ini
CAMERA_SOURCE=0
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

### 3. Enroll Your Face

Run the interactive face enrollment tool to register known household members:

```bash
python3 enroll_face.py
```

- Type your name (e.g., `shahin`)
- Look at the camera — **auto-captures 15 frames automatically!**
- The system encodes your face using ArcFace and saves it to the local secure database.

### 4. Run Security System

```bash
python3 main.py
```

> **Display Controls:**
> - Press **`Q`** or **`ESC`** to quit.
> - Press **`S`** to take a manual snapshot.

### 5. Launch Web Dashboard

In a new terminal window:

```bash
streamlit run dashboard.py
```

Open your browser at `http://localhost:8501` to view real-time logs and security analytics!

---

## 👤 Face Enrollment CLI Usage

The `enroll_face.py` tool provides an intuitive interface for managing authorized faces:

```bash
# Interactive mode (prompts for name & auto-captures from webcam):
python3 enroll_face.py

# Direct enrollment for a specific name:
python3 enroll_face.py --name "Alex"

# Enroll using existing photo files:
python3 enroll_face.py --name "Sarah" --images photo1.jpg photo2.jpg

# List all enrolled persons in database:
python3 enroll_face.py --list

# Remove a person from database:
python3 enroll_face.py --remove "Alex"
```

---

## 📊 Streamlit Dashboard

The web dashboard (`dashboard.py`) includes:

1. 📹 **Live Feed & Camera Controls:** Toggle detection modes (person, animal, vehicle) on the fly.
2. 📋 **Event Audit Log:** Filter historical events by type, priority, or date, and view captured intruder snapshots.
3. 📈 **Security Analytics:** Interactive Plotly charts displaying peak intruder activity hours and event breakdowns.
4. 👥 **Enrolled Persons Directory:** Manage and verify registered household members.

---

## 🛠️ Project Structure

```text
Now-you-see-me/
├── main.py                  # 🚀 Main detection & tracking engine
├── dashboard.py             # 📊 Streamlit web dashboard
├── enroll_face.py           # 👤 Interactive face enrollment CLI
├── requirements.txt         # 📦 Python package dependencies
├── config/
│   ├── settings.json        # ⚙️ Detection thresholds & zone configs
│   ├── zones.json           # 📍 Polygon zone coordinates
│   └── .env.example         # 🔑 Telegram & environment template
├── core/
│   ├── detector.py          # 🎯 YOLOv8 multi-class detection
│   ├── face_recognizer.py   # 🧠 ArcFace 512-D face recognition
│   ├── tracker.py           # 🔄 ByteTrack multi-object tracking
│   ├── motion_detector.py   # 🏃 MOG2 motion gating
│   ├── night_vision.py      # 🌙 CLAHE low-light enhancement
│   ├── loitering_detector.py# ⏱️ Dwell-time monitoring
│   └── zone_manager.py      # 📍 Polygon zone checking
├── alerts/
│   ├── alert_manager.py     # 🔔 Alert orchestrator
│   ├── telegram_alert.py    # 📱 Async Telegram Bot notifier
│   └── alarm.py             # 🔊 Audio alarm sound generator
└── database/
    ├── db_manager.py        # 🗄️ SQLite event database logging
    └── known_faces/         # 📁 Enrolled face embeddings & crops
```

---

## ⚙️ Configuration Specs (`config/settings.json`)

| Setting | Default | Description |
|---|---|---|
| `confidence_threshold` | `0.50` | YOLOv8 object detection confidence cutoff |
| `match_threshold` | `0.42` | Strict ArcFace cosine distance threshold (lower = stricter) |
| `loitering_threshold_seconds` | `30` | Time (in seconds) before flagging loitering |
| `detector_backend` | `opencv` | Fast face detector backend |
| `alert_cooldown` | `30.0` | Cooldown period between repetitive alerts (seconds) |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/ANKARAHAMSA/Now-you-see-me/issues).

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.

<div align="center">
  <sub>Built with ❤️ using Python, OpenCV, YOLOv8, ArcFace, and Streamlit.</sub>
</div>

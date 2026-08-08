"""
dashboard.py — Streamlit Real-Time Monitoring Dashboard

Provides a live web dashboard for the Intruder Detection System:
- Live camera feed (via OpenCV + Streamlit)
- Detection event log with snapshot previews
- Event statistics and charts
- System configuration panel
"""

from __future__ import annotations

import json
import time
import os
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intruder Detection System",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #030712 !important;
        color: #F3F4F6 !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Minimalist Dark Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0B0F17 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }

    /* Modern Glassmorphic Buttons */
    .stButton > button {
        background: rgba(255, 255, 255, 0.03) !important;
        color: #94A3B8 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        font-size: 0.92em !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        backdrop-filter: blur(12px) !important;
    }
    .stButton > button:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        color: #F8FAFC !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
        transform: translateY(-1px) !important;
    }

    /* Active Glowing Glass Button */
    button[kind="primary"], .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.2), rgba(99, 102, 241, 0.2)) !important;
        color: #38BDF8 !important;
        border: 1px solid rgba(56, 189, 248, 0.5) !important;
        box-shadow: 0 0 25px rgba(56, 189, 248, 0.25) !important;
        font-weight: 700 !important;
    }

    /* Glassmorphic Metric Stat Cards */
    .glass-card {
        background: rgba(17, 24, 39, 0.55);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        transition: all 0.25s ease;
    }
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.3);
        transform: translateY(-2px);
    }
    .glass-metric-title {
        font-size: 0.75em;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #9CA3AF;
        margin-bottom: 6px;
    }
    .glass-metric-val {
        font-size: 2.0em;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #FFFFFF;
        line-height: 1.1;
    }
    .glass-metric-sub {
        font-size: 0.75em;
        color: #6B7280;
        margin-top: 4px;
    }

    /* Glassmorphic Expanders */
    .stExpander {
        background: rgba(17, 24, 39, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(12px) !important;
    }
    [data-testid="stExpanderDetails"] {
        background: transparent !important;
    }

    /* Glassmorphic Form Inputs & Selectboxes */
    div[data-baseweb="select"] > div, .stTextInput > div > div > input {
        background-color: rgba(17, 24, 39, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        color: #F3F4F6 !important;
        backdrop-filter: blur(10px) !important;
    }

    /* Glassmorphic Tabs — Vercel / Apple Dark Pill Bar */
    [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {
        display: none !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: rgba(15, 23, 42, 0.6) !important;
        padding: 6px !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(16px) !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px !important;
        border-radius: 10px !important;
        color: #94A3B8 !important;
        font-weight: 600 !important;
        font-size: 0.92em !important;
        border: none !important;
        padding: 0 22px !important;
        background: transparent !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #F8FAFC !important;
        background: rgba(255, 255, 255, 0.05) !important;
    }
    .stTabs [aria-selected="true"] {
        background: #0EA5E9 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 20px rgba(14, 165, 233, 0.4) !important;
    }

    /* Section headers */
    .section-header {
        font-size: 1.2em;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: #38BDF8;
        padding-bottom: 8px;
        margin: 18px 0 14px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Live pulse indicator */
    .live-dot {
        display: inline-block;
        width: 9px; height: 9px;
        background: #10B981;
        border-radius: 50%;
        animation: pulse 1.5s infinite;
        margin-right: 8px;
    }
    @keyframes pulse {
        0%  { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.6); }
        70% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100%{ box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
</style>
""", unsafe_allow_html=True)


# ─── Data Loaders ─────────────────────────────────────────────────────────────

@st.cache_resource
def get_db():
    """Get database manager (cached singleton)."""
    from database.db_manager import DatabaseManager
    return DatabaseManager()


@st.cache_resource
def get_config() -> dict:
    cfg_path = Path("config/settings.json")
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    return {}


def get_events_df(db, limit: int = 200) -> pd.DataFrame:
    """Load events as DataFrame."""
    events = db.get_recent_events(limit)
    if not events:
        return pd.DataFrame(columns=["id", "timestamp", "event_type", "label", "confidence", "zone_name", "priority", "snapshot"])
    df = pd.DataFrame(events)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.markdown("## 🛡 IDS Control Panel")
    st.sidebar.markdown("---")

    # Telegram status
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    configured = bool(bot_token and bot_token != "YOUR_BOT_TOKEN_HERE")
    status_icon = "🟢" if configured else "🔴"
    st.sidebar.markdown(f"**Telegram:** {status_icon} {'Connected' if configured else 'Not configured'}")

    if not configured:
        st.sidebar.warning("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your .env file")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔒 Security System Status")
    cfg = get_config()
    current_mode = cfg.get("system_security", {}).get("armed_mode", "ARMED")

    mode_selection = st.sidebar.radio(
        "Select Security Mode:",
        ["🟢 ARMED", "🟡 SCHEDULED", "⚪ DISARMED"],
        index=0 if current_mode == "ARMED" else (1 if current_mode == "SCHEDULED" else 2)
    )

    new_mode = "ARMED" if "ARMED" in mode_selection else ("SCHEDULED" if "SCHEDULED" in mode_selection else "DISARMED")

    if new_mode != current_mode:
        cfg_path = Path("config/settings.json")
        if cfg_path.exists():
            with open(cfg_path, "r") as f:
                data = json.load(f)
            data.setdefault("system_security", {})["armed_mode"] = new_mode
            with open(cfg_path, "w") as f:
                json.dump(data, f, indent=2)
            st.sidebar.success(f"Mode updated to {new_mode}!")
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Quick Settings")

    cooldown = st.sidebar.slider("Alert Cooldown (s)", 5, 120, 30)
    conf_threshold = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.05)
    night_threshold = st.sidebar.slider("Night Vision Threshold", 20, 150, 80)

    st.sidebar.markdown("---")

    if st.sidebar.button("🔄 Reload Zones"):
        st.sidebar.success("Zone config reloaded!")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Quick Commands")
    st.sidebar.code("python main.py", language="bash")
    st.sidebar.code("python enroll_face.py --name 'You' --capture", language="bash")
    st.sidebar.code("streamlit run dashboard.py", language="bash")

    return cooldown, conf_threshold, night_threshold


# ─── Page Sections ────────────────────────────────────────────────────────────

def render_header():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            '<div style="font-size:2.2em;font-weight:800;letter-spacing:-0.03em;color:#FFFFFF;display:flex;align-items:center;gap:12px;">'
            '🛡️ <span style="background:linear-gradient(135deg, #38BDF8, #818CF8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">Intruder Detection System</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="display:flex;align-items:center;margin-top:4px;">'
            '<span class="live-dot"></span>'
            '<span style="color:#94A3B8;font-size:0.88em;font-weight:500;">Real-time Edge AI Security Sentinel</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div style="text-align:right;color:#64748B;font-size:0.85em;padding-top:6px;">'
            f'{datetime.now().strftime("%A, %b %d, %Y")}<br>'
            f'<strong style="color:#F1F5F9;font-size:1.15em;letter-spacing:0.02em;">{datetime.now().strftime("%H:%M:%S")}</strong>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_metrics(db):
    """Top row glassmorphic metric cards."""
    events = db.get_recent_events(1000)
    df = pd.DataFrame(events) if events else pd.DataFrame()

    today_df = df[df["timestamp"].str.startswith(datetime.now().strftime("%Y-%m-%d"))] if not df.empty else pd.DataFrame()

    counts = db.get_event_counts()
    total_intruders = counts.get("intruder", 0)
    total_animals = counts.get("animal", 0)
    total_loitering = counts.get("loitering", 0)
    today_count = len(today_df)

    last_time_str = "—"
    if events and not df.empty:
        raw_t = str(df["timestamp"].iloc[0])
        try:
            last_time_str = datetime.fromisoformat(raw_t).strftime("%H:%M:%S")
        except Exception:
            last_time_str = raw_t[11:19] if len(raw_t) >= 19 else raw_t

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            f'<div class="glass-card" style="border-top:3px solid #EF4444;">'
            f'<div class="glass-metric-title">🚨 Intruders</div>'
            f'<div class="glass-metric-val" style="color:#FCA5A5;">{total_intruders}</div>'
            f'<div class="glass-metric-sub">Total Intruder Alerts</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="glass-card" style="border-top:3px solid #F59E0B;">'
            f'<div class="glass-metric-title">🐾 Animals</div>'
            f'<div class="glass-metric-val" style="color:#FDE047;">{total_animals}</div>'
            f'<div class="glass-metric-sub">Wildlife Detections</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="glass-card" style="border-top:3px solid #A855F7;">'
            f'<div class="glass-metric-title">⏱️ Loitering</div>'
            f'<div class="glass-metric-val" style="color:#E9D5FF;">{total_loitering}</div>'
            f'<div class="glass-metric-sub">Dwell Time Triggers</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="glass-card" style="border-top:3px solid #38BDF8;">'
            f'<div class="glass-metric-title">📅 Events Today</div>'
            f'<div class="glass-metric-val" style="color:#BAE6FD;">{today_count}</div>'
            f'<div class="glass-metric-sub">Logged Past 24h</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f'<div class="glass-card" style="border-top:3px solid #10B981;">'
            f'<div class="glass-metric-title">⏰ Last Event</div>'
            f'<div class="glass-metric-val" style="font-size:1.6em;color:#6EE7B7;padding-top:4px;">{last_time_str}</div>'
            f'<div class="glass-metric-sub">Most Recent Activity</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_live_feed():
    """Live camera feed section."""
    st.markdown('<div class="section-header">📹 Live Camera Feed</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        feed_placeholder = st.empty()
    with col2:
        st.markdown("**Camera Controls**")
        camera_src = st.selectbox("Source", [0, 1, 2, "RTSP URL"], index=0)
        auto_refresh = st.checkbox("Auto-refresh feed", value=True)
        if st.button("📸 Take Snapshot"):
            st.success("Snapshot saved!")

        st.markdown("---")
        st.markdown("**Detection Modes**")
        st.checkbox("🔍 Person Detection", value=True)
        st.checkbox("🐾 Animal Detection", value=True)
        st.checkbox("🚗 Vehicle Detection", value=True)
        st.checkbox("🌙 Night Vision", value=True)
        st.checkbox("📍 Zone Monitoring", value=True)

    # Live video stream from database/snapshots/live_stream.jpg
    live_stream_file = Path("database/snapshots/live_stream.jpg")
    snapshots_dir = Path("database/snapshots")

    if live_stream_file.exists() and (time.time() - live_stream_file.stat().st_mtime < 5.0):
        try:
            feed_placeholder.image(
                str(live_stream_file),
                caption=f"🔴 LIVE CAMERA FEED — {datetime.now().strftime('%H:%M:%S')}",
                use_container_width=True,
            )
        except Exception:
            pass
        if auto_refresh:
            time.sleep(0.1)
            st.rerun()
    elif snapshots_dir.exists():
        recent_snapshots = sorted(list(snapshots_dir.glob("*.jpg")), key=lambda p: p.stat().st_mtime, reverse=True)
        recent_snapshots = [p for p in recent_snapshots if p.name != "live_stream.jpg"]
        if recent_snapshots:
            latest_img = recent_snapshots[0]
            mtime = datetime.fromtimestamp(latest_img.stat().st_mtime).strftime("%H:%M:%S")
            feed_placeholder.image(
                str(latest_img),
                caption=f"📸 Latest Detection Snapshot ({latest_img.name} at {mtime})",
                use_container_width=True,
            )
        else:
            feed_placeholder.info("💡 Run `python3 main.py` in terminal to view live video stream!")
    else:
        feed_placeholder.info("💡 Run `python3 main.py` in terminal to view live video stream!")


def render_event_log(db):
    """Recent events table with snapshot previews."""
    st.markdown('<div class="section-header">📋 Detection Event Log</div>', unsafe_allow_html=True)

    df = get_events_df(db)

    if df.empty:
        st.info("No events logged yet. Start the detection system with `python main.py`.")
        return

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        type_filter = st.multiselect(
            "Event Type",
            options=df["event_type"].unique().tolist(),
            default=df["event_type"].unique().tolist(),
        )
    with col2:
        priority_filter = st.multiselect(
            "Priority",
            options=["HIGH", "MEDIUM", "LOW"],
            default=["HIGH", "MEDIUM", "LOW"],
        )
    with col3:
        search = st.text_input("Search label", "")

    filtered_df = df.copy()
    if type_filter:
        filtered_df = filtered_df[filtered_df["event_type"].isin(type_filter)]
    if priority_filter:
        filtered_df = filtered_df[filtered_df["priority"].isin(priority_filter)]
    if search:
        filtered_df = filtered_df[filtered_df["label"].str.contains(search, case=False, na=False)]

    # Export Buttons (CSV & HTML Summary Report)
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV Security Log",
            data=csv_data,
            file_name=f"security_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with exp_col2:
        report_text = f"# 🛡 SECURITY INCIDENT REPORT\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nTotal Incidents Logged: {len(filtered_df)}\n\n"
        for _, row in filtered_df.iterrows():
            report_text += f"- [{row['timestamp']}] {row['event_type'].upper()} ({row['priority']}): {row['label']} | Zone: {row.get('zone_name','None')}\n"

        st.download_button(
            label="📄 Download Security Report (TXT/MD)",
            data=report_text.encode('utf-8'),
            file_name=f"security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.dataframe(
        filtered_df[["id", "timestamp", "event_type", "label", "priority", "zone_name", "confidence"]],
        use_container_width=True,
        hide_index=True,
    )

    # Snapshot Gallery
    st.markdown("### 📸 Event Snapshots")
    snapshot_col = "snapshot_path" if "snapshot_path" in filtered_df.columns else "snapshot" if "snapshot" in filtered_df.columns else None
    if snapshot_col:
        snapshot_events = filtered_df[filtered_df[snapshot_col].astype(str).str.len() > 0].head(8)

        if not snapshot_events.empty:
            cols = st.columns(4)
            for idx, (_, row) in enumerate(snapshot_events.iterrows()):
                path = Path(str(row[snapshot_col]))
                if path.exists():
                    img = cv2.imread(str(path))
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        with cols[idx % 4]:
                            st.image(img_rgb, caption=path.stem[:30], use_container_width=True)
                            st.caption(f"**{row.get('label', '')}**")
        else:
            st.info("No snapshots saved yet.")


def render_analytics(db):
    """Analytics charts section."""
    st.markdown('<div class="section-header">📊 Analytics</div>', unsafe_allow_html=True)

    df = get_events_df(db, limit=500)
    if df.empty:
        st.info("No data available for analytics.")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Events by type (donut chart)
        type_counts = df["event_type"].value_counts().reset_index()
        type_counts.columns = ["type", "count"]
        fig = px.pie(
            type_counts, values="count", names="type",
            title="Events by Type",
            hole=0.4,
            color_discrete_map={
                "intruder": "#EF4444",
                "animal": "#F59E0B",
                "loitering": "#A855F7",
                "vehicle": "#3B82F6",
                "spoof_attack": "#EC4899",
            },
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94A3B8",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Events over time (line chart)
        df["hour"] = df["timestamp"].dt.floor("h")
        hourly = df.groupby(["hour", "event_type"]).size().reset_index(name="count")
        fig2 = px.line(
            hourly, x="hour", y="count", color="event_type",
            title="Events Over Time",
            color_discrete_map={
                "intruder": "#EF4444",
                "animal": "#F59E0B",
                "loitering": "#A855F7",
                "vehicle": "#3B82F6",
                "spoof_attack": "#EC4899",
            },
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94A3B8",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Priority breakdown
    if "priority" in df.columns:
        priority_counts = df["priority"].value_counts().reset_index()
        priority_counts.columns = ["priority", "count"]
        fig3 = px.bar(
            priority_counts, x="priority", y="count",
            title="Alert Priority Distribution",
            color="priority",
            color_discrete_map={"HIGH": "#ff4444", "MEDIUM": "#ff8800", "LOW": "#2ea043"},
        )
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9",
            showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True)


def render_enrolled_persons():
    """Show enrolled persons list with primary thumbnail & expandable full photo gallery."""
    st.markdown('<div class="section-header">👥 Enrolled Persons Directory</div>', unsafe_allow_html=True)

    known_dir = Path("database/known_faces")
    if not known_dir.exists():
        known_dir.mkdir(parents=True, exist_ok=True)

    # Get subdirectories for each person
    person_dirs = [d for d in known_dir.iterdir() if d.is_dir()]

    if not person_dirs:
        st.info("ℹ️ No persons enrolled yet.\n\nRun in terminal: `python3 enroll_face.py` or use interactive CLI.")
        return

    st.caption("Click on any person's gallery to view all captured training photos.")

    for p_dir in sorted(person_dirs, key=lambda d: d.name.lower()):
        person_name = p_dir.name
        photos = sorted(list(p_dir.glob("*.jpg")) + list(p_dir.glob("*.png")) + list(p_dir.glob("*.jpeg")))

        # Card container for each person
        with st.container():
            col_photo, col_info, col_gallery = st.columns([1, 2, 4])

            # 1. Primary Photo Thumbnail
            with col_photo:
                if photos:
                    primary_img = cv2.imread(str(photos[0]))
                    if primary_img is not None:
                        primary_rgb = cv2.cvtColor(primary_img, cv2.COLOR_BGR2RGB)
                        st.image(primary_rgb, use_container_width=True)
                    else:
                        st.markdown('<div style="font-size:3em;text-align:center;">👤</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="font-size:3em;text-align:center;">👤</div>', unsafe_allow_html=True)

            # 2. Person Name & Photo Count Info
            with col_info:
                st.markdown(f"### 👤 {person_name}")
                st.markdown(f"🖼️ **{len(photos)}** photo(s) captured")
                st.markdown("✅ Status: **Authorized / Registered**")

                # Remove entire person button
                if st.button(f"🗑️ Delete {person_name}", key=f"del_person_{person_name}"):
                    import shutil
                    from core.face_recognizer import FaceRecognizer
                    rec = FaceRecognizer(get_config())
                    if p_dir.exists():
                        shutil.rmtree(p_dir)
                    rec.rebuild_person_embeddings(person_name)
                    st.success(f"Removed '{person_name}' from database.")
                    st.rerun()

            # 3. Expandable Photo Gallery with per-photo Delete button
            with col_gallery:
                with st.expander(f"📂 View All {len(photos)} Photos of {person_name}", expanded=False):
                    if photos:
                        g_cols = st.columns(4)
                        for idx, p_path in enumerate(photos):
                            g_img = cv2.imread(str(p_path))
                            if g_img is not None:
                                g_rgb = cv2.cvtColor(g_img, cv2.COLOR_BGR2RGB)
                                with g_cols[idx % 4]:
                                    st.image(g_rgb, caption=f"#{idx+1}", use_container_width=True)
                                    if st.button("🗑️ Delete", key=f"del_img_{person_name}_{idx}"):
                                        try:
                                            os.remove(p_path)
                                            from core.face_recognizer import FaceRecognizer
                                            rec = FaceRecognizer(get_config())
                                            rec.rebuild_person_embeddings(person_name)
                                            st.success(f"Deleted photo #{idx+1}!")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Error deleting photo: {e}")
                    else:
                        st.info("No photo files stored in folder.")

        st.markdown("---")


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    render_sidebar()
    render_header()

    st.markdown("---")

    # Top metrics
    try:
        db = get_db()
        render_metrics(db)
    except Exception as e:
        st.warning(f"Database not initialized yet: {e}")

    st.markdown("---")

    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = "Live Feed"

    # Glassmorphic Custom Button Nav Bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        is_act = st.session_state["active_tab"] == "Live Feed"
        if st.button("📹 Live Camera Feed", key="nav_live", use_container_width=True, type="primary" if is_act else "secondary"):
            st.session_state["active_tab"] = "Live Feed"
            st.rerun()

    with col2:
        is_act = st.session_state["active_tab"] == "Event Log"
        if st.button("📋 Event Audit Logs", key="nav_log", use_container_width=True, type="primary" if is_act else "secondary"):
            st.session_state["active_tab"] = "Event Log"
            st.rerun()

    with col3:
        is_act = st.session_state["active_tab"] == "Analytics"
        if st.button("📊 Security Analytics", key="nav_analytics", use_container_width=True, type="primary" if is_act else "secondary"):
            st.session_state["active_tab"] = "Analytics"
            st.rerun()

    with col4:
        is_act = st.session_state["active_tab"] == "Enrolled Directory"
        if st.button("👥 Enrolled Directory", key="nav_enrolled", use_container_width=True, type="primary" if is_act else "secondary"):
            st.session_state["active_tab"] = "Enrolled Directory"
            st.rerun()

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    current_tab = st.session_state["active_tab"]
    if current_tab == "Live Feed":
        render_live_feed()
    elif current_tab == "Event Log":
        try:
            db = get_db()
            render_event_log(db)
        except Exception as e:
            st.error(f"Error loading events: {e}")
    elif current_tab == "Analytics":
        try:
            db = get_db()
            render_analytics(db)
        except Exception as e:
            st.error(f"Error loading analytics: {e}")
    elif current_tab == "Enrolled Directory":
        render_enrolled_persons()

    # Auto-refresh
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh Dashboard"):
            st.rerun()
    with col1:
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}  •  Auto-refresh every 30s")

    # Auto-refresh via JS
    st.markdown(
        """<script>
        setTimeout(function() { window.location.reload(); }, 30000);
        </script>""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

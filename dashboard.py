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
    /* Dark theme overrides */
    .main { background-color: #0d1117; }
    .stApp { background-color: #0d1117; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e, #252d3d);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 6px 0;
    }

    /* Alert badges */
    .badge-high   { background:#ff4444; color:#fff; padding:3px 10px; border-radius:20px; font-size:0.78em; }
    .badge-medium { background:#ff8800; color:#fff; padding:3px 10px; border-radius:20px; font-size:0.78em; }
    .badge-low    { background:#2ea043; color:#fff; padding:3px 10px; border-radius:20px; font-size:0.78em; }

    /* Section headers */
    .section-header {
        font-size: 1.1em;
        font-weight: 700;
        color: #58a6ff;
        border-bottom: 1px solid #30363d;
        padding-bottom: 6px;
        margin: 16px 0 10px 0;
    }

    /* Live indicator */
    .live-dot {
        display: inline-block;
        width: 10px; height: 10px;
        background: #2ea043;
        border-radius: 50%;
        animation: pulse 1.5s infinite;
        margin-right: 8px;
    }
    @keyframes pulse {
        0%  { box-shadow: 0 0 0 0 rgba(46,160,67,0.6); }
        70% { box-shadow: 0 0 0 8px rgba(46,160,67,0); }
        100%{ box-shadow: 0 0 0 0 rgba(46,160,67,0); }
    }

    /* Snapshot grid */
    .snapshot-img { border-radius: 8px; border: 1px solid #30363d; }
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
            '<div style="font-size:2.2em;font-weight:800;color:#58a6ff;">'
            '🛡 Intruder Detection System'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<span class="live-dot"></span>'
            '<span style="color:#8b949e;font-size:0.9em;">Real-time AI Security Monitoring</span>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div style="text-align:right;color:#8b949e;font-size:0.85em;margin-top:12px;">'
            f'{datetime.now().strftime("%A, %B %d, %Y")}<br>'
            f'<strong style="color:#c9d1d9">{datetime.now().strftime("%H:%M:%S")}</strong>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_metrics(db):
    """Top row metric cards."""
    events = db.get_recent_events(1000)
    df = pd.DataFrame(events) if events else pd.DataFrame()

    today_df = df[df["timestamp"].str.startswith(datetime.now().strftime("%Y-%m-%d"))] if not df.empty else pd.DataFrame()

    counts = db.get_event_counts()
    total_intruders = counts.get("intruder", 0)
    total_animals = counts.get("animal", 0)
    total_loitering = counts.get("loitering", 0)
    today_count = len(today_df)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("🚨 Total Intruder Events", total_intruders)
    with col2:
        st.metric("🐾 Animal Events", total_animals)
    with col3:
        st.metric("⏱ Loitering Events", total_loitering)
    with col4:
        st.metric("📅 Events Today", today_count)
    with col5:
        last_event_time = df["timestamp"].iloc[0] if not df.empty else "No events yet"
        st.metric("⏰ Last Event", str(last_event_time)[:19] if events else "—")


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
        feed_placeholder.image(
            str(live_stream_file),
            caption=f"🔴 LIVE CAMERA FEED — {datetime.now().strftime('%H:%M:%S')}",
            use_container_width=True,
        )
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

    st.dataframe(
        filtered_df[["id", "timestamp", "event_type", "label", "priority", "zone_name", "confidence"]],
        use_container_width=True,
        hide_index=True,
    )

    # Snapshot Gallery
    st.markdown("### 📸 Event Snapshots")
    if "snapshot" in filtered_df.columns:
        snapshot_events = filtered_df[filtered_df["snapshot"].str.len() > 0].head(8)

        if not snapshot_events.empty:
            cols = st.columns(4)
                if path.exists():
                    img = cv2.imread(str(path))
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        with cols[i % 4]:
                            st.image(img_rgb, caption=path.stem[:30], use_column_width=True)
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
                "intruder": "#ff4444",
                "animal": "#ff8800",
                "loitering": "#aa44ff",
                "vehicle": "#ffcc00",
            },
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Events over time (line chart)
        df["hour"] = df["timestamp"].dt.floor("H")
        hourly = df.groupby(["hour", "event_type"]).size().reset_index(name="count")
        fig2 = px.line(
            hourly, x="hour", y="count", color="event_type",
            title="Events Over Time",
            color_discrete_map={
                "intruder": "#ff4444",
                "animal": "#ff8800",
                "loitering": "#aa44ff",
                "vehicle": "#ffcc00",
            },
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9",
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
    """Show enrolled persons."""
    st.markdown('<div class="section-header">👤 Enrolled Persons</div>', unsafe_allow_html=True)

    from core.face_recognizer import FaceRecognizer, EMBEDDINGS_DB
    import pickle

    if not EMBEDDINGS_DB.exists():
        st.info("No persons enrolled yet.\n\nRun: `python enroll_face.py --name 'Your Name' --capture`")
        return

    try:
        with open(EMBEDDINGS_DB, "rb") as f:
            db = pickle.load(f)
        cols = st.columns(4)
        for i, (name, embeddings) in enumerate(db.items()):
            with cols[i % 4]:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div style="font-size:2em;text-align:center;">👤</div>'
                    f'<div style="text-align:center;font-weight:600;color:#c9d1d9">{name}</div>'
                    f'<div style="text-align:center;color:#8b949e;font-size:0.8em">{len(embeddings)} embedding(s)</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    except Exception as e:
        st.error(f"Error loading person database: {e}")


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

    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📹 Live Feed", "📋 Event Log", "📊 Analytics", "👥 Enrolled Persons"])

    with tab1:
        render_live_feed()

    with tab2:
        try:
            db = get_db()
            render_event_log(db)
        except Exception as e:
            st.error(f"Error loading events: {e}")

    with tab3:
        try:
            db = get_db()
            render_analytics(db)
        except Exception as e:
            st.error(f"Error loading analytics: {e}")

    with tab4:
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

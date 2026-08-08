"""
main.py — Intruder Detection System Entry Point

Orchestrates the full detection pipeline:
  Camera → Night Vision → Motion Gate → YOLO → Tracking →
  Face Recognition → Zone Check → Loitering → Alerts → Display
"""

from __future__ import annotations

import json
import os
import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from utils.logger import get_logger
from utils.image_utils import (
    draw_detection, draw_zones, draw_fps,
    draw_mode_indicator, draw_alert_banner, add_timestamp,
)
from core.detector import YOLODetector
from core.face_recognizer import FaceRecognizer
from core.motion_detector import MotionDetector
from core.tracker import ObjectTracker
from core.night_vision import NightVisionEnhancer
from core.loitering_detector import LoiteringDetector
from core.zone_manager import ZoneManager
from alerts.alert_manager import AlertManager

logger = get_logger("main")

SETTINGS_FILE = Path("config/settings.json")


def load_config() -> dict:
    """Load settings.json configuration."""
    if not SETTINGS_FILE.exists():
        logger.warning(f"Settings file not found: {SETTINGS_FILE}. Using defaults.")
        return {}
    with open(SETTINGS_FILE) as f:
        return json.load(f)


def open_camera(source) -> cv2.VideoCapture:
    """Open camera source (int index or RTSP URL string)."""
    try:
        src = int(source)
    except (ValueError, TypeError):
        src = str(source)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        logger.error(f"Failed to open camera: {src}")
        sys.exit(1)

    # Optimize capture settings for low latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    logger.info(f"Camera opened: {src} ({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})")
    return cap


def get_color_key(detection) -> str:
    """Map detection to drawing color key."""
    if detection.category == "person":
        return "intruder"  # Overridden to 'known' after face recognition
    if detection.category == "animal":
        return "animal_harmful" if detection.is_harmful_animal else "animal_safe"
    if detection.category == "vehicle":
        return "vehicle"
    return "intruder"


def is_system_armed(config: dict) -> bool:
    """Check if security system is ARMED based on settings.json mode or schedule."""
    try:
        cfg = load_config()
        sec = cfg.get("system_security", {})
        mode = sec.get("armed_mode", "ARMED")

        if mode == "DISARMED":
            return False
        if mode == "ARMED":
            return True

        if mode == "SCHEDULED":
            from datetime import datetime, time as dt_time
            now_t = datetime.now().time()
            s_h, s_m = map(int, sec.get("schedule_start", "22:00").split(":"))
            e_h, e_m = map(int, sec.get("schedule_end", "06:00").split(":"))

            start_t = dt_time(s_h, s_m)
            end_t = dt_time(e_h, e_m)

            if start_t > end_t:  # Overnight schedule e.g. 22:00 to 06:00
                return now_t >= start_t or now_t <= end_t
            else:
                return start_t <= now_t <= end_t
    except Exception:
        pass
    return True


# ── Face detection helpers ─────────────────────────────────────────────────────

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _detect_faces_in_frame(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Detect all faces in a full frame using Haar cascade.
    Downscales frame to 640px width for fast 30+ FPS detection.
    """
    h, w = frame.shape[:2]
    scale = 640.0 / max(w, 1)
    if scale < 1.0:
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
    else:
        small = frame
        scale = 1.0

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
    )

    if len(faces) == 0:
        return []

    # Map bounding boxes back to original full frame resolution
    orig_faces = []
    for (x, y, fw, fh) in faces:
        orig_faces.append((
            int(x / scale),
            int(y / scale),
            int(fw / scale),
            int(fh / scale),
        ))
    return orig_faces


def _get_face_crop_for_person(
    frame: np.ndarray,
    person_bbox: tuple[int, int, int, int],
    face_boxes: list[tuple[int, int, int, int]],
    size: int = 224,
) -> np.ndarray | None:
    """
    Given a YOLO person bounding box, find the best overlapping face detection
    and return a resized 224×224 face crop for ArcFace embedding.

    Falls back to the top-40% of the person crop if no face box overlaps.
    """
    px1, py1, px2, py2 = person_bbox

    best_crop = None
    best_overlap = 0

    for (fx, fy, fw, fh) in face_boxes:
        # Compute overlap between face box and person box
        ix1, iy1 = max(px1, fx), max(py1, fy)
        ix2, iy2 = min(px2, fx + fw), min(py2, fy + fh)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        overlap = (ix2 - ix1) * (iy2 - iy1)
        if overlap > best_overlap:
            best_overlap = overlap
            # Crop with 20% padding
            pad = int(max(fw, fh) * 0.2)
            x1 = max(0, fx - pad)
            y1 = max(0, fy - pad)
            x2 = min(frame.shape[1], fx + fw + pad)
            y2 = min(frame.shape[0], fy + fh + pad)
            best_crop = frame[y1:y2, x1:x2]

    if best_crop is not None and best_crop.size > 0:
        return cv2.resize(best_crop, (size, size))

    # NO FALLBACK to non-face crops — if no face detected in person box, return None
    return None


def run_detection(config: dict, args: argparse.Namespace):
    """Main detection loop."""
    logger.info("═" * 60)
    logger.info("  Intruder Detection System — Starting")
    logger.info("═" * 60)

    # ── Initialize components ─────────────────────────────────────────────
    camera_source = os.getenv("CAMERA_SOURCE", config.get("camera_source", 0))
    cap = open_camera(camera_source)

    detector = YOLODetector(config)
    face_rec = FaceRecognizer(config)
    motion_det = MotionDetector(config)
    tracker = ObjectTracker(config)
    night_vision = NightVisionEnhancer(config)
    loitering = LoiteringDetector(config)
    zones = ZoneManager()
    alerts = AlertManager(config)

    show_display = config.get("display", {}).get("show_fps", True) and not args.headless
    show_zones = config.get("display", {}).get("show_zones", True)
    window_name = config.get("display", {}).get("window_name", "Intruder Detection System")

    logger.info(f"Known persons: {face_rec.known_persons or ['None enrolled — run enroll_face.py']}")

    # Send startup message to Telegram
    alerts.telegram.send_system_message(
        "🛡 Intruder Detection System started.\n"
        f"Camera: {camera_source} | Known persons: {len(face_rec.known_persons)}"
    )

    # ── Threaded Async Face Recognition Worker ──────────────────────────────
    import queue
    import threading

    face_queue: queue.Queue = queue.Queue(maxsize=4)
    face_cache: dict[int, tuple[str, float]] = {}  # track_id -> (name, distance)
    pending_checks: set[int] = set()
    last_submit_time: dict[int, float] = {}
    first_seen_time: dict[int, float] = {}
    FACE_CHECK_INTERVAL = 1.5

    def _face_worker():
        while True:
            try:
                item = face_queue.get(timeout=1.0)
                if item is None:
                    break
                tid, f_crop = item
                try:
                    # Check Liveness / Anti-Spoofing
                    is_live, liveness_score, reason = face_rec.check_liveness(f_crop)
                    if not is_live:
                        logger.warning(f"Anti-spoofing triggered for ID {tid}: {reason}")
                        face_cache[tid] = ("SPOOF ATTACK!", 1.0)
                    else:
                        emb = face_rec.extract_embedding(f_crop)
                        if emb is not None:
                            name, dist = face_rec.identify(emb, tid)
                            face_cache[tid] = (name, dist)
                except Exception as e:
                    logger.debug(f"Async face worker error: {e}")
                finally:
                    pending_checks.discard(tid)
                    face_queue.task_done()
            except queue.Empty:
                continue

    worker_thread = threading.Thread(target=_face_worker, daemon=True)
    worker_thread.start()

    # ── FPS & Performance Tracking ─────────────────────────────────────────
    fps_counter = 0
    fps_start = time.time()
    current_fps = 0.0
    active_alert_banner: tuple[str, str] | None = None
    banner_until: float = 0.0

    frame_index = 0
    last_detections = []

    logger.info("Detection loop running — Press Q to quit")
    if show_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Frame read failed — attempting reconnect...")
            time.sleep(0.5)
            cap.release()
            cap = open_camera(camera_source)
            tracker.reset()
            face_cache.clear()
            continue

        now = time.time()
        frame_index += 1

        # ── Preprocessing ─────────────────────────────────────────────────
        display_frame = frame.copy()
        enhanced_frame = night_vision.enhance(frame)

        # ── Motion Gating ─────────────────────────────────────────────────
        motion_active = motion_det.update(enhanced_frame)

        detections = []
        if motion_active:
            # ── Object Detection (every 2nd frame for 2x FPS boost) ────────
            if frame_index % 2 == 0 or not last_detections:
                raw_dets = detector.detect(enhanced_frame)
                detections = tracker.update(raw_dets, frame.shape[:2])
                last_detections = detections
            else:
                detections = last_detections

            # ── Loitering Check ────────────────────────────────────────────
            person_track_ids = [
                d.track_id for d in detections
                if d.category == "person" and d.track_id is not None
            ]
            loiterers = loitering.update(person_track_ids)

            # ── Face Detection ──────────────────────────────────────────────
            face_boxes = _detect_faces_in_frame(frame) if any(d.category == "person" for d in detections) else []

            # ── Process each detection ─────────────────────────────────────
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                color_key = get_color_key(det)
                label = det.class_name

                # Zone check
                violated_zones = zones.get_violated_zones(det.bbox)
                zone_name = violated_zones[0]["name"] if violated_zones else ""
                priority = zones.get_highest_priority(violated_zones) if violated_zones else "MEDIUM"

                # ── Person: Async Face Recognition ────────────────────────
                if det.category == "person":
                    track_id = det.track_id

                    # Extract face crop
                    face_crop = _get_face_crop_for_person(frame, det.bbox, face_boxes)

                    # Check if we need to submit face for async recognition
                    if track_id is not None:
                        last_t = last_submit_time.get(track_id, 0.0)
                        if (now - last_t > FACE_CHECK_INTERVAL) and (track_id not in pending_checks):
                            if face_crop is not None:
                                try:
                                    pending_checks.add(track_id)
                                    last_submit_time[track_id] = now
                                    face_queue.put_nowait((track_id, face_crop))
                                except queue.Full:
                                    pending_checks.discard(track_id)

                    # Lookup cached face recognition result ONLY if face crop is valid
                    if track_id is not None and track_id in face_cache and face_crop is not None:
                        name, distance = face_cache[track_id]
                    else:
                        name, distance = "UNKNOWN", 1.0
                        if track_id is not None and face_crop is None:
                            face_cache.pop(track_id, None)

                    det.class_name = name
                    label = name

                    # Track when person ID was first seen
                    if track_id is not None and track_id not in first_seen_time:
                        first_seen_time[track_id] = now

                    track_duration = (now - first_seen_time.get(track_id, now)) if track_id is not None else 2.0

                    if name == "UNKNOWN" or name == "SPOOF ATTACK!":
                        color_key = "intruder"
                        label = name if name == "SPOOF ATTACK!" else "UNKNOWN"
                        # Only trigger intruder alert if system is ARMED and after 1.2s grace period
                        if track_duration >= 1.2 and is_system_armed(config):
                            alerts.trigger(
                                event_type="intruder" if name != "SPOOF ATTACK!" else "spoof_attack",
                                label=f"{label} (ID:{det.track_id})",
                                frame=display_frame,
                                confidence=1.0 - distance,
                                zone_name=zone_name,
                                priority="HIGH" if name == "SPOOF ATTACK!" else priority,
                            )
                            active_alert_banner = (f"{label} DETECTED{' in ' + zone_name if zone_name else ''}", "intruder")
                            banner_until = time.time() + 5.0
                    else:
                        color_key = "known"
                        label = f"OK {name}"

                    # Loitering alert (only for UNKNOWN intruders when ARMED)
                    if name == "UNKNOWN" and det.track_id in loiterers and is_system_armed(config):
                        dwell = loitering.get_dwell_time(det.track_id)
                        alerts.trigger(
                            event_type="loitering",
                            label=f"UNKNOWN (ID:{det.track_id})",
                            frame=display_frame,
                            zone_name=zone_name,
                            priority="HIGH",
                            notes=f"Dwell time: {dwell:.0f}s",
                        )
                        active_alert_banner = (f"UNKNOWN LOITERING: {dwell:.0f}s", "intruder")
                        banner_until = time.time() + 5.0

                # ── Animal Detection ───────────────────────────────────────
                elif det.category == "animal":
                    animal_priority = "HIGH" if det.is_harmful_animal else "LOW"
                    if det.is_harmful_animal or violated_zones:
                        alerts.trigger(
                            event_type="animal",
                            label=det.class_name,
                            frame=display_frame,
                            confidence=det.confidence,
                            zone_name=zone_name,
                            priority=animal_priority,
                        )
                        active_alert_banner = (f"ANIMAL DETECTED: {det.class_name.upper()}", color_key)
                        banner_until = time.time() + 4.0

                # ── Vehicle Detection ──────────────────────────────────────
                elif det.category == "vehicle" and violated_zones:
                    alerts.trigger(
                        event_type="vehicle",
                        label=det.class_name,
                        frame=display_frame,
                        confidence=det.confidence,
                        zone_name=zone_name,
                        priority="MEDIUM",
                    )

                # Draw detection
                dwell_str = ""
                if det.track_id and det.category == "person":
                    dwell_s = loitering.get_dwell_time(det.track_id)
                    if dwell_s > 3:
                        dwell_str = f" {dwell_s:.0f}s"

                draw_detection(
                    display_frame, det.bbox,
                    f"{label}{dwell_str}",
                    color_key=color_key,
                    confidence=det.confidence if det.category != "person" else None,
                    track_id=det.track_id,
                )

        # ── Overlay ───────────────────────────────────────────────────────
        if show_zones:
            draw_zones(display_frame, zones.zones)

        draw_fps(display_frame, current_fps)
        draw_mode_indicator(display_frame, night_vision.active, motion_active)
        add_timestamp(display_frame)

        # Alert banner
        if active_alert_banner and time.time() < banner_until:
            draw_alert_banner(display_frame, active_alert_banner[0], active_alert_banner[1])

        # ── FPS calculation ───────────────────────────────────────────────
        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            current_fps = fps_counter / (time.time() - fps_start)
            fps_counter = 0
            fps_start = time.time()

        # ── Save live frame for Web Dashboard streaming (Atomic write) ────
        try:
            snapshots_dir = Path("database/snapshots")
            snapshots_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = snapshots_dir / "live_stream.tmp"
            final_path = snapshots_dir / "live_stream.jpg"
            cv2.imwrite(str(tmp_path), display_frame)
            os.replace(str(tmp_path), str(final_path))
        except Exception:
            pass

        # ── Display ───────────────────────────────────────────────────────
        if show_display:
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                logger.info("Quit signal received")
                break
            elif key == ord("r"):
                zones.reload()
                logger.info("Zones reloaded")
            elif key == ord("s"):
                from utils.snapshot import save_snapshot
                p = save_snapshot(display_frame, "manual", "screenshot")
                logger.info(f"Manual snapshot: {p}")

    # ── Cleanup ───────────────────────────────────────────────────────────
    cap.release()
    if show_display:
        cv2.destroyAllWindows()
    alerts.telegram.send_system_message("🔴 Intruder Detection System stopped.")
    logger.info("System stopped cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Intruder Detection System")
    parser.add_argument("--headless", action="store_true", help="Run without display window")
    parser.add_argument("--config", default="config/settings.json", help="Path to settings.json")
    args = parser.parse_args()

    config = load_config()
    run_detection(config, args)


if __name__ == "__main__":
    main()

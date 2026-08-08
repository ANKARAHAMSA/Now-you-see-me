"""
enroll_face.py — Face Enrollment CLI Tool

Register known persons into the face recognition database.
Usage:
  python enroll_face.py                               # Interactive mode (asks for name)
  python enroll_face.py --name "John"                 # Auto-capture from webcam
  python enroll_face.py --name "Jane" --images *.jpg  # From image files
  python enroll_face.py --list                        # Show enrolled persons
  python enroll_face.py --remove "John"               # Remove person from DB
"""

from __future__ import annotations

import argparse
import sys
import time
import pickle
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from utils.logger import get_logger
from core.face_recognizer import FaceRecognizer, EMBEDDINGS_DB, KNOWN_FACES_DIR

logger = get_logger("enroll")

KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    import json
    cfg_path = Path("config/settings.json")
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    return {}


def cmd_list(rec: FaceRecognizer):
    """List all enrolled persons."""
    if not rec.known_persons:
        print("\n  No persons enrolled yet.\n  Run: python enroll_face.py\n")
        return

    print(f"\n  ✅ Enrolled Persons ({len(rec.known_persons)} total):")
    print("  " + "─" * 40)
    for name in rec.known_persons:
        count = len(rec._known_db.get(name, []))
        print(f"  • {name:<30} {count} image(s)")
    print()


def cmd_remove(rec: FaceRecognizer, name: str):
    """Remove a person from the database."""
    if name not in rec._known_db:
        print(f"  Person '{name}' not found in database.")
        return

    del rec._known_db[name]
    rec.save_embeddings()
    print(f"  ✓ Removed '{name}' from database.")


def cmd_enroll_images(rec: FaceRecognizer, name: str, image_paths: list[str]):
    """Enroll a person from provided image files."""
    valid_paths = []
    for p in image_paths:
        path = Path(p)
        if not path.exists():
            logger.warning(f"Image not found: {p}")
        else:
            valid_paths.append(str(path))

    if not valid_paths:
        print("  ✗ No valid image files provided.")
        return

    print(f"\n  Enrolling '{name}' from {len(valid_paths)} image(s)...")
    for img_path in tqdm(valid_paths, desc="  Copying"):
        person_dir = KNOWN_FACES_DIR / name
        person_dir.mkdir(exist_ok=True)
        import shutil
        shutil.copy2(img_path, person_dir / Path(img_path).name)

    count = rec.enroll_person(name, valid_paths)
    if count > 0:
        print(f"\n  ✅ Successfully enrolled '{name}' with {count} image(s)!")
    else:
        print(f"\n  ✗ Could not extract face embeddings. Ensure images contain clear, frontal faces.")


def cmd_capture(rec: FaceRecognizer, name: str, n_frames: int = 30):
    """
    AUTO-capture face images directly from webcam for enrollment.
    Automatically captures 30 frames rapidly when a face is detected!
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ✗ Cannot open webcam.")
        return

    person_dir = KNOWN_FACES_DIR / name
    person_dir.mkdir(exist_ok=True)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    print(f"\n  📷 Rapid Auto-capturing {n_frames} frames for '{name}'")
    print("  ✅ Just look at the camera — rapid capturing starts automatically!")
    print("  Press Q to finish early, ESC to cancel\n")

    captured = []
    last_capture_time = 0.0
    capture_interval = 0.15       # Rapid-fire auto-capture every 0.15 seconds (6.6 photos/sec!)
    face_stable_since = 0.0       # Track how long face has been visible
    stable_required = 0.2         # Face must be visible for only 0.2s before capturing

    cv2.namedWindow("Enrollment — Auto Capture", cv2.WINDOW_NORMAL)

    while len(captured) < n_frames:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        now = time.time()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        face_detected = len(faces) > 0

        # ── Track face stability ─────────────────────────────────────────
        if face_detected:
            if face_stable_since == 0.0:
                face_stable_since = now
            stable_duration = now - face_stable_since
        else:
            face_stable_since = 0.0
            stable_duration = 0.0

        # ── Draw face boxes ───────────────────────────────────────────────
        for (x, y, w, h) in faces:
            ready = stable_duration >= stable_required and (now - last_capture_time) >= capture_interval
            color = (0, 255, 0) if ready else (0, 200, 255)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            cx, cy = x + w // 2, y + h // 2
            cv2.line(display, (cx - 10, cy), (cx + 10, cy), color, 1)
            cv2.line(display, (cx, cy - 10), (cx, cy + 10), color, 1)

        # ── Auto-capture logic ────────────────────────────────────────────
        if (
            face_detected
            and stable_duration >= stable_required
            and (now - last_capture_time) >= capture_interval
            and len(captured) < n_frames
        ):
            ts = int(now * 1000)
            img_path = str(person_dir / f"capture_{ts}.jpg")

            # Save CROPPED face (not full frame) so DeepFace can skip detection
            x, y, w, h = faces[0]
            pad = int(max(w, h) * 0.2)   # 20% padding around face
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)
            face_crop = frame[y1:y2, x1:x2]
            # Resize to standard 224x224 for ArcFace
            face_crop = cv2.resize(face_crop, (224, 224))
            cv2.imwrite(img_path, face_crop)

            captured.append(img_path)
            last_capture_time = now
            print(f"  ✓ Auto-captured frame {len(captured)}/{n_frames}")

            # Flash effect
            flash = display.copy()
            cv2.rectangle(flash, (0, 0), (display.shape[1], display.shape[0]), (255, 255, 255), -1)
            cv2.addWeighted(flash, 0.4, display, 0.6, 0, display)

        # ── Progress bar ──────────────────────────────────────────────────
        progress = len(captured) / n_frames
        bar_w = int(frame.shape[1] * progress)
        cv2.rectangle(display, (0, frame.shape[0] - 8), (bar_w, frame.shape[0]), (0, 255, 100), -1)
        cv2.rectangle(display, (0, frame.shape[0] - 8), (frame.shape[1], frame.shape[0]), (60, 60, 60), 1)

        # ── Status overlay ────────────────────────────────────────────────
        h_frame = frame.shape[0]
        cv2.rectangle(display, (0, 0), (display.shape[1], 50), (20, 20, 20), -1)

        cv2.putText(display, f"Capturing: {name}  |  {len(captured)}/{n_frames} frames",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

        if not face_detected:
            cv2.putText(display, "Move your face into frame",
                        (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
        elif stable_duration < stable_required:
            cv2.putText(display, f"Hold still... ({stable_duration:.1f}s)",
                        (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
        else:
            time_to_next = max(0.0, capture_interval - (now - last_capture_time))
            hint = "CAPTURING..." if time_to_next < 0.1 else f"Next capture in {time_to_next:.1f}s"
            color = (0, 255, 100) if time_to_next < 0.1 else (0, 255, 0)
            cv2.putText(display, hint, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        cv2.putText(display, "Q = finish early   ESC = cancel",
                    (10, h_frame - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow("Enrollment — Auto Capture", display)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC
            print("  Enrollment cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured:
        print(f"\n  Processing {len(captured)} captured frames...")
        count = rec.enroll_person(name, captured)
        if count > 0:
            print(f"\n  ✅ Successfully enrolled '{name}' with {count} face embedding(s)!")
            print(f"  Now run: python3 main.py\n")
        else:
            print("  ✗ Could not extract embeddings. Try better lighting or move closer.")
    else:
        print("  No frames captured.")


def main():
    parser = argparse.ArgumentParser(
        description="Face Enrollment Tool — Intruder Detection System",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--name", type=str, help="Person's name (prompted if omitted)")
    parser.add_argument("--images", nargs="+", help="Image file paths")
    parser.add_argument("--list", action="store_true", help="List enrolled persons")
    parser.add_argument("--remove", type=str, help="Remove person from database")
    parser.add_argument("--frames", type=int, default=30, help="Frames to auto-capture (default: 30)")
    args = parser.parse_args()

    config = load_config()
    rec = FaceRecognizer(config)

    if args.list:
        cmd_list(rec)
        return

    if args.remove:
        cmd_remove(rec, args.remove)
        return

    if args.images:
        name = args.name or input("\n  Enter person's name: ").strip()
        if not name:
            print("  ✗ Name cannot be empty.")
            return
        cmd_enroll_images(rec, name, args.images)
        return

    # ── Default: interactive auto-capture ────────────────────────────────────
    if not args.name:
        print("\n" + "═" * 50)
        print("  🛡  Intruder Detection System — Face Enrollment")
        print("═" * 50)
        name = input("\n  Enter your name: ").strip()
        if not name:
            print("  ✗ Name cannot be empty.")
            return
    else:
        name = args.name

    # Show existing embedding count
    existing_count = len(rec._known_db.get(name, []))
    if existing_count > 0:
        print(f"\n  ℹ  '{name}' already has {existing_count} face embedding(s).")
        print("  New captures will be ADDED to existing data (not replaced).\n")

    cmd_capture(rec, name, args.frames)


if __name__ == "__main__":
    main()

from flask import Flask, Response, jsonify, send_file, request, session, redirect
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import atexit
import cv2
import os
import threading
import pymysql
import requests
import time
import signal
import sys
from ultralytics import YOLO
import re
import numpy as np
from detection import MIN_CONFIDENCE
from detection.config import TELEGRAM_TOKEN, CHAT_ID, COOLDOWN_SECONDS
from detection.camera_utils import get_calibrated_indices, get_cameras_with_device_path
from detection.face_utils import (
    recognize_face, save_face_image, delete_face_folder, clear_face_cache,
    list_face_photos, get_first_photo_path, delete_face_photo, FACES_DB
)

app = Flask(__name__)
app.secret_key = 'aquatic_secret_key_2024'

# Database configuration
DB_CONFIG = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '',
    'database': 'alltankdata',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

TANK_DB_CONFIG = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '',
    'database': 'alltankdata',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

VALID_TANKS = {1: 'tank1', 2: 'tank2', 3: 'tank3'}

# Global variables
model = None          # YOLO cam — uses track() (single thread, no lock needed)
surv_model = None     # Surv cams 1,2,3 — uses model() (shared with model_infer_lock)
lock = threading.Lock()
model_infer_lock = threading.Lock()  # only for surv cams (protect surv_model)
last_telegram_sent = 0
telegram_thread_pool = []
max_telegram_threads = 3
running = True

# Detection stats
current_person_count = 0
daily_detection_total = 0
last_detection_timestamp = "-"

# Face recognition globals
face_results             = {}
face_result_lock         = threading.Lock()
face_pending_frame       = None
face_pending_items       = []       # list of (track_id, box) instead of just boxes
face_pending_lock        = threading.Lock()

# Track buffer: per track_id keep N recent face crops + identity
TRACK_BUFFER_SIZE     = 8          # max crops per track
TRACK_CLEANUP_TIMEOUT = 5.0        # seconds without sighting → remove track
UNKNOWN_DEFER_SEC     = 15         # wait N sec before sending "unknown" alert (CAM1 entrance door)
track_crop_buffer     = {}         # {track_id: [crop_img, ...]}
track_identity        = {}         # {track_id: {name, status, confidence}}
track_last_seen       = {}         # {track_id: timestamp}
track_first_seen      = {}         # {track_id: timestamp} — used for deferred unknown alert
track_buffer_lock     = threading.Lock()
face_recognition_ran_once = False   # True after first recognition cycle completes
access_rules_cache        = []
access_rules_cache_time   = 0.0
latest_face_list          = []   # list of {name, status, confidence, detected_at} for all current detections
latest_face_lock          = threading.Lock()
cam2_capture_path         = None   # latest frame captured by Camera 2
cam2_capture_lock         = threading.Lock()
cam2_face_list            = []     # latest recognition results from Camera 2
cam2_face_lock            = threading.Lock()

# Event capture system — store correct camera images per event type
entry_event_capture  = None   # cam2 frame saved when ENTER fires
exit_event_capture   = None   # YOLO cam frame saved when EXIT fires
latest_event_type    = None   # 'ENTER' or 'EXIT' — web picks right image
latest_event_person  = None   # name of person who triggered the event
event_capture_lock   = threading.Lock()
latest_raw_frames    = {1: None, 2: None, 3: None, 'yolo': None}  # latest BGR frame per camera

# ── Event-Based Occupancy Counter ────────────────────────────
room_occupants    = {}   # {person_name: {"entry_time": float, "last_seen": float}}
occupancy_lock    = threading.Lock()
STALE_OCCUPANT_SEC = 300  # ถ้าคนไม่ถูก detect นานเกินนี้ → ถือว่าออกไปแล้ว (safety net)

# ── Unified Access Control ─────────────────────────────────────
ENTRY_WINDOW_SEC       = 12   # both cam1+cam2 must see person within N sec → ENTER
CAM2_SOLO_CONFIDENCE   = 0.80 # if cam2 sees with confidence >= this, ENTER without waiting cam1
PRESENCE_WINDOW_SEC    = 10   # cam2/yolo must see person within N sec → still inside
EXIT_WINDOW_SEC        = 8    # cam3 must see person within N sec → EXIT candidate
EXIT_CONFIRM_DELAY_SEC = 8    # wait N sec for presence to clear before logging EXIT
EXIT_MAX_WAIT_SEC      = 30   # max timeout for exit confirmation
UNKNOWN_COOLDOWN_SEC   = 60   # min sec between unknown-person alerts per camera
SURV_DETECT_INTERVAL   = 0.5  # sec: surv detect loop interval
SURV_FACE_INTERVALS    = {1: 1.0, 2: 1.0, 3: 3.0}  # sec: face rec rate limit per cam (cam2=fast, cam3=no face rec)

# person_state: {face_folder: {name, inside, entered_at, last_seen, exit_pending, exit_started_at}}
person_state      = {}
person_state_lock = threading.Lock()
unknown_last_alert = {}  # {cam_id: last_alert_timestamp}
unknown_alert_lock = threading.Lock()

# Best-frame buffers: keep up to N frames per surv cam, pick sharpest for recognition
BEST_FRAME_BUFFER_SIZE = 5
best_frame_buffers = {1: [], 2: [], 3: []}
best_frame_locks   = {1: threading.Lock(), 2: threading.Lock(), 3: threading.Lock()}

# Voting: cam1 and cam3 require N consistent recognitions before update_sighting()
# cam2 updates immediately (real-time presence tracking)
VOTE_REQUIRED      = 1    # single recognition enough (face rec already has quality gates)
VOTE_WINDOW_SEC    = 12   # votes older than this are discarded
_vote_records      = {1: {}, 3: {}}   # {cam_id: {face_folder: [timestamps]}}
_vote_lock         = threading.Lock()

# ── IoU Tracking ───────────────────────────────────────────────
IOU_THRESHOLD       = 0.15  # primary: IoU threshold for cross-frame matching
CENTER_DIST_LIMIT   = 0.20  # fallback: max center movement as fraction of frame width

# ── Per-camera face-buffer size ────────────────────────────────
SURV_TRACK_BUFFER_SIZES = {1: 12, 2: 6, 3: 4}  # cam1 backup=more buffer, cam3 no face rec

# ── Cameras that skip face recognition (e.g. exit cam sees only legs) ──
CAM_NO_FACE_RECOGNITION = set()

# ── YOLO camera → 即時 - YOLO AI section ──
yolo_cap         = None
yolo_frame       = None
yolo_frame_count = 0
yolo_lock        = threading.Lock()
yolo_online      = False

# ── Surveillance cameras: logical ID → physical index ──────────
#    Dynamically assigned via camera_utils on startup / reconnect
SURV_IDS = [1, 2, 3]

def refresh_camera_indices():
    """Update YOLO_CAM_IDX and SURV_CAM_IDX from live camera enumeration."""
    global YOLO_CAM_IDX, SURV_CAM_IDX
    try:
        mapping, _ = get_calibrated_indices()
        YOLO_CAM_IDX = mapping.get('yolo', 0)
        SURV_CAM_IDX = {
            1: mapping.get('cam1', 1),
            2: mapping.get('cam2', 2),
            3: mapping.get('cam3', 3),
        }
        print(f"📷 Camera indices refreshed: YOLO={YOLO_CAM_IDX}, "
              f"攝影機={SURV_CAM_IDX}")
    except Exception as e:
        print(f"⚠️ Camera index refresh failed: {e}")

refresh_camera_indices()
surv_caps         = {}
surv_frames       = {}
surv_frame_counts = {i: 0 for i in SURV_IDS}
surv_locks        = {i: threading.Lock() for i in SURV_IDS}
surv_online       = {i: False for i in SURV_IDS}

def close_all_cameras():
    global yolo_cap, yolo_online, surv_caps, surv_online
    if yolo_cap is not None:
        try:
            yolo_cap.release()
            yolo_online = False
            print(f"📹 YOLO CAM (index {YOLO_CAM_IDX}): 已關閉")
        except Exception as e:
            print(f"❌ YOLO CAM: 關閉失敗 - {e}")
        yolo_cap = None
    for cam_id, c in list(surv_caps.items()):
        try:
            c.release()
            surv_online[cam_id] = False
            print(f"📹 攝影機 {cam_id}: 已關閉")
        except Exception as e:
            print(f"❌ 攝影機 {cam_id}: 關閉失敗 - {e}")
    surv_caps.clear()

def _flush_camera(cap, n=15):
    """Read and discard N frames to flush DirectShow initial black frames."""
    for _ in range(n):
        cap.read()


def surv_capture_loop(cam_id, phys_idx):
    """cam_id = logical display ID (1/2/3), phys_idx = physical camera index."""
    global surv_frames, surv_frame_counts, surv_online, surv_caps, running
    frame_interval = 1.0 / 5
    surv_idle_loops  = 0
    consecutive_fail = 0   # transient read() failures before real reconnect
    while running:
        try:
            t0 = time.time()
            cap_obj = surv_caps.get(cam_id)
            if not cap_obj:
                surv_idle_loops += 1
                if surv_idle_loops >= 30:
                    surv_idle_loops = 0
                    refresh_camera_indices()
                    cur_idx = SURV_CAM_IDX.get(cam_id, phys_idx)
                    print(f"🔄 攝影機 {cam_id} (index {cur_idx}): 嘗試重新初始化...")
                    c = cv2.VideoCapture(cur_idx, cv2.CAP_DSHOW)
                    if c.isOpened():
                        c.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                        c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        c.set(cv2.CAP_PROP_BUFFERSIZE,   1)
                        _flush_camera(c)
                        surv_caps[cam_id]   = c
                        surv_online[cam_id] = True
                        consecutive_fail    = 0
                        print(f"🔄 攝影機 {cam_id} (index {cur_idx}): 重新初始化成功")
                    else:
                        c.release()
                time.sleep(0.5)
                continue

            success, img = cap_obj.read()

            if not success:
                consecutive_fail += 1
                print(f"⚠️ CAM{cam_id}: read() failed (#{consecutive_fail})")
                if consecutive_fail < 8:
                    # Transient failure — wait briefly and retry without reconnecting
                    time.sleep(0.2)
                    continue
                # Persistent failure → full reconnect
                print(f"⚠️ CAM{cam_id}: {consecutive_fail} consecutive failures — disconnecting")
                consecutive_fail = 0
                surv_online[cam_id] = False
                time.sleep(1)
                old_cap = surv_caps.get(cam_id)
                if old_cap:
                    try:
                        old_cap.release()
                    except Exception:
                        pass
                surv_caps[cam_id] = None
                refresh_camera_indices()
                cur_idx = SURV_CAM_IDX.get(cam_id, phys_idx)
                new_cap = cv2.VideoCapture(cur_idx, cv2.CAP_DSHOW)
                if new_cap.isOpened():
                    new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    _flush_camera(new_cap)
                    surv_caps[cam_id]   = new_cap
                    surv_online[cam_id] = True
                    print(f"🔄 攝影機 {cam_id} (index {cur_idx}): 重新連接成功")
                else:
                    new_cap.release()
                continue

            consecutive_fail = 0

            # Skip obviously-black frames (DirectShow warm-up artifact)
            if img.mean() < 2.0:
                time.sleep(0.05)
                continue

            # Store raw frame for entry/exit event captures
            with event_capture_lock:
                latest_raw_frames[cam_id] = img.copy()

            ret, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 50])
            if ret:
                with surv_locks[cam_id]:
                    surv_frames[cam_id]       = buf.tobytes()
                    surv_frame_counts[cam_id] += 1

            elapsed = time.time() - t0
            sleep_t = frame_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
        except Exception as e:
            print(f"❌ SURV CAM {cam_id} CAPTURE LOOP CRASHED: {e}")
            time.sleep(1)

def yolo_capture_loop():
    """Background thread: capture from YOLO_CAM_IDX, run YOLO, store in yolo_frame."""
    global yolo_cap, yolo_frame, yolo_frame_count, yolo_online
    global model, last_telegram_sent, running
    global current_person_count, daily_detection_total, last_detection_timestamp
    global face_pending_frame, face_pending_items, face_recognition_ran_once

    frame_interval = 1.0 / 10   # 10 fps target
    detect_every   = 3
    local_count    = 0
    last_boxes     = []          # list of boxes (used only for drawing & counting)
    last_track_ids = []          # parallel list of track_id per box
    idle_loops     = 0
    last_frame_time = 0

    while running:
        try:
            t0 = time.time()

            if yolo_cap is None:
                idle_loops += 1
                # Refresh camera indices in case USB re-enumeration happened
                if idle_loops >= 30:
                    idle_loops = 0
                    refresh_camera_indices()
                    print(f"🔄 YOLO CAM (index {YOLO_CAM_IDX}): 嘗試重新初始化...")
                    init_yolo_camera()
                time.sleep(0.5)
                continue

            success, img = yolo_cap.read()
            if not success:
                print(f"⚠️ YOLO CAM (index {YOLO_CAM_IDX}): read() failed — going offline")
                yolo_online = False
                time.sleep(1)
                try:
                    yolo_cap.release()
                except Exception:
                    pass
                yolo_cap = None
                # Refresh indices before reconnect (cameras may have shifted)
                refresh_camera_indices()
                now = time.time()
                if last_frame_time > 0 and (now - last_frame_time) > 3:
                    print(f"🔄 YOLO CAM (index {YOLO_CAM_IDX}): 強制重新連接 (逾時 {now - last_frame_time:.1f}s)")
                new_cap = cv2.VideoCapture(YOLO_CAM_IDX, cv2.CAP_DSHOW)
                if new_cap.isOpened():
                    new_cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                    new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    new_cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
                    yolo_cap    = new_cap
                    yolo_online = True
                    last_frame_time = time.time()
                    print(f"🔄 YOLO CAM (index {YOLO_CAM_IDX}): 重新連接成功")
                else:
                    new_cap.release()
                continue

            last_frame_time = time.time()

            local_count += 1

            if model is not None and local_count % detect_every == 0:
                try:
                    # YOLO cam uses its own model instance — no lock needed
                    results = model.track(img, classes=[0], persist=True, verbose=False)
                    raw_boxes = [b for b in results[0].boxes
                                 if float(b.conf[0]) >= MIN_CONFIDENCE]
                    last_boxes = []
                    last_track_ids = []
                    for b in raw_boxes:
                        tid = int(b.id[0]) if b.id is not None else None
                        last_boxes.append(b)
                        last_track_ids.append(tid)
                except Exception as e:
                    print(f"❌ AI DETECTION: {e}")
                    last_boxes = []
                    last_track_ids = []

                with lock:
                    current_person_count = len(last_boxes)

                if last_boxes:
                    max_c = max(float(b.conf[0]) for b in last_boxes) if last_boxes else 0
                    print(f"🔍 CAM[yolo]: detected {len(last_boxes)} person(s), max_conf={max_c:.2f}")

                # Trigger face recognition every 5 YOLO detections (~1.5 s)
                if last_boxes and local_count % (detect_every * 5) == 0:
                    with face_pending_lock:
                        face_pending_frame = img.copy()
                        face_pending_items = list(zip(last_track_ids, last_boxes))

                if last_boxes:
                    now = time.time()
                    # Save image + log to detection_logs (for history image lookup)
                    if now - last_telegram_sent >= COOLDOWN_SECONDS:
                        count = len(last_boxes)
                        daily_detection_total += count
                        last_detection_timestamp = time.strftime("%Y/%m/%d %H:%M:%S")

                        ts             = time.strftime("%Y%m%d_%H%M%S")
                        image_filename = f"detection_{ts}.jpg"
                        image_path     = os.path.join("detection", "captures", image_filename)
                        cv2.imwrite(image_path, img)

                        with track_buffer_lock:
                            current_tracks = track_identity.copy()
                        primary  = next(iter(current_tracks.values()), {"name": None, "status": "unknown"})
                        p_name   = primary.get("name")
                        p_status = primary.get("status", "unknown")

                        conn = get_db_connection()
                        try:
                            with conn.cursor() as cur:
                                cur.execute(
                                    "INSERT INTO detection_logs "
                                    "(person_count, image_filename, person_name, access_status) "
                                    "VALUES (%s, %s, %s, %s)",
                                    (count, image_filename, p_name, p_status)
                                )
                                conn.commit()
                        except Exception as e:
                            print(f"❌ DB: detection_logs insert failed - {e}")
                        finally:
                            conn.close()

                        last_telegram_sent = now

                    # Feed YOLO-cam sightings into unified state machine (via track_identity)
                    with track_buffer_lock:
                        current_tracks = track_identity.copy()
                    for tid, r in current_tracks.items():
                        r_name   = r.get("name")
                        r_status = r.get("status", "unknown")
                        if r_name and r_status != "unknown":
                            folder = next(
                                (rule["face_folder"] for rule in access_rules_cache
                                 if rule["name"] == r_name),
                                r_name
                            )
                            update_sighting(folder, r_name, 'yolo')
                    # Unknown tracks are handled deferred via _cleanup_stale_tracks
                    # so we don't call update_unknown_sighting here anymore

            # Draw bounding boxes with face recognition labels (via track_id)
            with track_buffer_lock:
                draw_track = track_identity.copy()

            for i, box in enumerate(last_boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                tid   = last_track_ids[i] if i < len(last_track_ids) else None
                info  = draw_track.get(tid) if tid is not None else None
                if info:
                    status = info.get("status", "unknown")
                    name   = info.get("name") or ""
                    if status == "authorized":
                        color = (0, 200, 0)
                        label = f"Authorized: {name}"
                    elif status == "unauthorized":
                        color = (0, 0, 220)
                        label = f"Unauth: {name}"
                    else:
                        color = (0, 140, 255)
                        label = "Unknown"
                else:
                    color = (0, 255, 0)
                    label = f"Person {float(box.conf[0]):.0%}"
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
                cv2.putText(img, label, (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Store raw frame for event captures
            with event_capture_lock:
                latest_raw_frames['yolo'] = img.copy()

            ret, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 50])
            if ret:
                with yolo_lock:
                    yolo_frame        = buf.tobytes()
                    yolo_frame_count += 1

            elapsed = time.time() - t0
            sleep_t = frame_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
        except Exception as e:
            print(f"❌ YOLO CAPTURE LOOP CRASHED: {e}")
            time.sleep(1)


# ── Capture file retention ──────────────────────────────────────
CAPTURE_CAM2_KEEP_HOURS  = 24   # cam2 frames (saved every ~3s — high volume)
CAPTURE_DETECT_KEEP_DAYS = 7    # YOLO detection captures
CAPTURE_UNKNOWN_MAX      = 200  # max unknown-person captures to keep


def _ts_fmt(val):
    """Convert timedelta / str to HH:MM string."""
    if val is None:
        return "00:00"
    if isinstance(val, str):
        return str(val)[:5]
    total = int(val.total_seconds())
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}"


def _ensure_access_rules():
    """Refresh access_rules_cache from DB if older than 60 s. Thread-safe."""
    global access_rules_cache, access_rules_cache_time
    if time.time() - access_rules_cache_time < 60:
        return
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT name, face_folder, access_start, access_end, is_active "
                "FROM authorized_personnel"
            )
            rows = cur.fetchall()
        access_rules_cache = [
            {
                "name":         r["name"],
                "face_folder":  r["face_folder"],
                "access_start": _ts_fmt(r.get("access_start")),
                "access_end":   _ts_fmt(r.get("access_end")),
                "is_active":    bool(r.get("is_active", 1)),
            }
            for r in rows
        ]
        access_rules_cache_time = time.time()
        print(f"🔄 ACCESS RULES: refreshed ({len(access_rules_cache)} personnel)")
    except Exception as e:
        print(f"⚠️ ACCESS RULES: refresh failed - {e}")
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def _restore_person_state():
    """On startup, rebuild person_state from room_occupants_log so EXIT events
    can still fire correctly even after a server restart."""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT face_folder, person_name, entered_at FROM room_occupants_log"
            )
            rows = cur.fetchall()
        with person_state_lock:
            for r in rows:
                ff = r["face_folder"]
                if ff not in person_state:
                    person_state[ff] = {
                        "name":            r["person_name"],
                        "inside":          False,
                        "entered_at":      None,
                        "last_seen":       {1: 0, 2: 0, 3: 0, 'yolo': 0},
                        "last_confidence": {1: 0.0, 2: 0.0, 3: 0.0, 'yolo': 0.0},
                        "exit_pending":    False,
                        "exit_started_at": 0,
                    }
        print(f"✅ STATE RESTORE: {len(rows)} person(s) restored from room_occupants_log")
    except Exception as e:
        print(f"⚠️ STATE RESTORE: failed - {e}")
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def cleanup_old_captures():
    """Background thread: delete stale capture files to prevent disk fill."""
    captures_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "detection", "captures"
    )
    while running:
        try:
            if os.path.isdir(captures_dir):
                now = time.time()
                removed = 0
                for fname in os.listdir(captures_dir):
                    fpath = os.path.join(captures_dir, fname)
                    if not os.path.isfile(fpath):
                        continue
                    age = now - os.path.getmtime(fpath)
                    if fname.startswith("cam2_") and age > CAPTURE_CAM2_KEEP_HOURS * 3600:
                        os.remove(fpath)
                        removed += 1
                    elif fname.startswith("detection_") and age > CAPTURE_DETECT_KEEP_DAYS * 86400:
                        os.remove(fpath)
                        removed += 1

                # Keep only newest CAPTURE_UNKNOWN_MAX unknown captures
                unk_files = sorted(
                    [os.path.join(captures_dir, f) for f in os.listdir(captures_dir)
                     if f.startswith("unknown_cam")],
                    key=os.path.getmtime, reverse=True
                )
                for old in unk_files[CAPTURE_UNKNOWN_MAX:]:
                    try:
                        os.remove(old)
                        removed += 1
                    except Exception:
                        pass

                if removed:
                    print(f"🗑️ CLEANUP: removed {removed} old capture file(s)")

                # Stale occupant cleanup (safety net)
                _cleanup_stale_occupants()
        except Exception as e:
            print(f"⚠️ CLEANUP: {e}")
        time.sleep(3600)  # run once per hour


def _compute_iou(box_a, box_b):
    """IoU between two YOLO boxes (tensor with xyxy format)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (a_area + b_area - inter + 1e-8)


def _match_boxes(prev_boxes, curr_boxes, iou_thresh=None, frame_width=640):
    """
    Match current frame boxes to previous tracks.
    1. Primary: IoU (threshold 0.15) — high overlap = same track
    2. Fallback: center distance (threshold 20% of frame width) — catches fast walkers
    prev_boxes: list of (box_xyxy, track_id)
    curr_boxes: list of box (xyxy tensor)
    Returns: {curr_idx: track_id}
    """
    if iou_thresh is None:
        iou_thresh = IOU_THRESHOLD
    center_limit_px = frame_width * CENTER_DIST_LIMIT  # 640*0.20 = 128px

    matched   = {}
    used_tids = set()
    next_id = (max([tid for _, tid in prev_boxes], default=0) + 1) if prev_boxes else 1

    for ci, cb in enumerate(curr_boxes):
        cb_xyxy = [float(cb[0]), float(cb[1]), float(cb[2]), float(cb[3])]
        cx = (cb_xyxy[0] + cb_xyxy[2]) / 2
        cy = (cb_xyxy[1] + cb_xyxy[3]) / 2

        best_iou     = 0.0
        best_dist    = float("inf")
        best_tid_iou = None
        best_tid_center = None

        for pb, ptid in prev_boxes:
            if ptid in used_tids:
                continue
            pb_xyxy = [float(pb[0]), float(pb[1]), float(pb[2]), float(pb[3])]

            # Primary: IoU
            iou = _compute_iou(cb_xyxy, pb_xyxy)
            if iou > best_iou and iou >= iou_thresh:
                best_iou     = iou
                best_tid_iou = ptid

            # Fallback: center distance
            px = (pb_xyxy[0] + pb_xyxy[2]) / 2
            py = (pb_xyxy[1] + pb_xyxy[3]) / 2
            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if dist < best_dist and dist < center_limit_px:
                best_dist     = dist
                best_tid_center = ptid

        # Prefer IoU match; fall back to center distance
        chosen_tid = best_tid_iou if best_tid_iou is not None else best_tid_center

        if chosen_tid is not None:
            matched[ci] = chosen_tid
            used_tids.add(chosen_tid)

    # New tracks for unmatched current boxes
    for ci in range(len(curr_boxes)):
        if ci not in matched:
            matched[ci] = next_id
            next_id += 1

    return matched


def _crop_face_from_box(img, box, head_ratio=0.30):
    """Crop head region from a YOLO bounding box. Returns (crop, ok).
    head_ratio: fraction of box height to crop from top (lower = more focused on face area)."""
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    box_h   = y2 - y1
    head_y2 = min(img.shape[0], y1 + int(box_h * head_ratio))
    margin  = max(10, int(box_h * 0.05))
    crop_y1 = max(0, y1 - margin)
    crop_x1 = max(0, x1)
    crop_x2 = min(img.shape[1], x2)
    face_img = img[crop_y1:head_y2, crop_x1:crop_x2]
    if face_img.size == 0:
        return None, False
    return face_img, True


def _cleanup_stale_tracks():
    """Remove tracks that haven't been seen for TRACK_CLEANUP_TIMEOUT seconds.
    Also fire deferred unknown alerts for tracks that remained unidentified."""
    global track_first_seen

    now = time.time()

    # Fire deferred unknown alerts for confirmed-unknown tracks
    with track_buffer_lock:
        # ถ้ามี track ไหนถูกจำได้แล้ว → ไม่ต้อง alert unknown
        any_identified = bool(track_identity)
        fire_tids = []
        for tid, first_seen in list(track_first_seen.items()):
            if tid in track_identity:
                track_first_seen.pop(tid, None)  # identified → cancel alert
                continue
            if now - first_seen < UNKNOWN_DEFER_SEC:
                continue
            # Still alive & still unknown after defer period → fire alert
            if not any_identified:
                fire_tids.append(tid)
            track_first_seen.pop(tid, None)

        stale = [tid for tid, t in track_last_seen.items()
                 if now - t > TRACK_CLEANUP_TIMEOUT]
        for tid in stale:
            track_crop_buffer.pop(tid, None)
            track_identity.pop(tid, None)
            track_last_seen.pop(tid, None)
            track_first_seen.pop(tid, None)

    # Fire alerts outside lock
    # YOLO cam unknown alerts disabled — it's an overview camera, faces are always too small.
    # Unknown person detection is handled by CAM1 (entrance door) in surv_detect_loop.
    for tid in fire_tids:
        print(f"🔔 DEFERRED: YOLO track {tid} unknown after {UNKNOWN_DEFER_SEC}s (suppressed — CAM1 handles unknown alerts)")
        with yolo_lock:
            yolo_img_bytes = yolo_frame
        yolo_img = None
        if yolo_img_bytes:
            yolo_img = cv2.imdecode(np.frombuffer(yolo_img_bytes, np.uint8), cv2.IMREAD_COLOR)

    if stale:
        print(f"🧹 TRACK: cleaned {len(stale)} stale track(s)")


def face_recognition_loop():
    """
    Background thread: recognize faces using track-aware multi-frame buffer.
    For each track_id:
      - If already identified → skip (carry forward identity)
      - Else → crop head region, try recognize_face
      - If fail → push crop to buffer (up to TRACK_BUFFER_SIZE)
      - On next frames: try ALL buffered crops (maybe person turned toward camera later)
      - If any crop succeeds → tag track_id in track_identity → update person_state
    """
    global face_pending_frame, face_pending_items, face_results
    global running
    global latest_face_list

    while running:
        img     = None
        items   = []   # list of (track_id, box)

        with face_pending_lock:
            if face_pending_frame is not None and face_pending_items:
                img   = face_pending_frame.copy()
                items = list(face_pending_items)
                face_pending_frame = None
                face_pending_items = []

        if img is None:
            time.sleep(0.05)
            continue

        _ensure_access_rules()

        now = time.time()

        with track_buffer_lock:
            # Mark all current tracks as seen
            for tid, _ in items:
                if tid is not None:
                    track_last_seen[tid] = now

            for tid, box in items:
                if tid is None:
                    continue

                # Already identified → no need to re-recognize
                if tid in track_identity:
                    continue

                # Record first-seen for deferred unknown alert
                if tid not in track_first_seen:
                    track_first_seen[tid] = now

                # Crop head region from current frame
                face_img, ok = _crop_face_from_box(img, box)
                if not ok:
                    continue

                # Try recognize on the current crop
                result = recognize_face(face_img, access_rules_cache, cam_id=0)
                name   = result.get("name")
                status = result.get("status", "unknown")

                if name and status != "unknown":
                    # ✅ Identified!
                    track_identity[tid] = result
                    track_crop_buffer.pop(tid, None)  # clear buffer
                    track_first_seen.pop(tid, None)   # cancel pending unknown alert
                    print(f"✅ TRACK {tid}: identified as {name} ({status}) conf={result.get('confidence',0):.2f}")
                    # Feed into state machine
                    folder = next(
                        (r["face_folder"] for r in access_rules_cache if r["name"] == name),
                        name,
                    )
                    update_sighting(folder, name, 'yolo', confidence=result.get("confidence", 0.0))
                    update_occupancy_last_seen(name)
                else:
                    # ❌ Not identified yet → save crop to buffer for retry later
                    buf = track_crop_buffer.setdefault(tid, [])
                    buf.append(face_img)
                    if len(buf) > TRACK_BUFFER_SIZE:
                        buf.pop(0)

                    # Try ALL buffered crops — maybe an older frame had a better angle
                    identified = False
                    for idx, old_crop in enumerate(buf):
                        result2 = recognize_face(old_crop, access_rules_cache, cam_id=0)
                        name2   = result2.get("name")
                        status2 = result2.get("status", "unknown")
                        if name2 and status2 != "unknown":
                            track_identity[tid] = result2
                            track_crop_buffer.pop(tid, None)
                            track_first_seen.pop(tid, None)  # cancel pending unknown
                            print(f"✅ TRACK {tid}: identified via buffer[{idx}] as {name2} ({status2})")
                            folder = next(
                                (r["face_folder"] for r in access_rules_cache if r["name"] == name2),
                                name2,
                            )
                            update_sighting(folder, name2, 'yolo', confidence=result2.get("confidence", 0.0))
                            update_occupancy_last_seen(name2)
                            identified = True
                            break

                    if not identified:
                        if result["status"] == "unknown":
                            print(f"⏳ TRACK {tid}: not yet identified (buffered {len(buf)} crops)")

        # Build latest_face_list from track_identity for dashboard
        with track_buffer_lock:
            current_tracks = track_identity.copy()
        now_str = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        with latest_face_lock:
            if current_tracks:
                latest_face_list = [
                    {
                        "name":        r.get("name"),
                        "status":      r.get("status", "unknown"),
                        "confidence":  r.get("confidence", 0.0),
                        "detected_at": now_str,
                    }
                    for r in current_tracks.values()
                ]
            else:
                latest_face_list = []

        _cleanup_stale_tracks()

        global face_recognition_ran_once
        face_recognition_ran_once = True




def _save_cam2_capture(frame):
    """Save a Camera-2 frame and update the dashboard capture globals."""
    global cam2_capture_path, cam2_face_list
    captures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "detection", "captures")
    os.makedirs(captures_dir, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(captures_dir, f"cam2_{ts}.jpg")
    cv2.imwrite(path, frame)
    with cam2_capture_lock:
        cam2_capture_path = path


def surv_detect_loop(cam_id):
    """
    Low-fps person detection + face recognition on a surveillance camera.
    Uses YOLO detection with custom IoU-based matching for cross-frame track
    continuity (avoids sharing YOLO's built-in tracker state across cameras).
    Per-track face-crop buffer retries identification across frames.
    cam1 → triggers ENTER   cam3 → triggers EXIT pending   cam2 → updates identity timestamps
    """
    global cam2_face_list

    # Per-camera IoU tracker state
    SURV_TRACK_BUFFER_SIZE = SURV_TRACK_BUFFER_SIZES.get(cam_id, 6)
    prev_tracks            = []  # [(box_xyxy_tensor, track_id), ...]
    surv_track_crops       = {}  # {track_id: [crop_img, ...]}
    surv_track_ident       = {}  # {track_id: {name, status, confidence}}
    surv_track_first_seen  = {}  # {track_id: timestamp} — deferred alert

    last_face_time = 0
    # Stagger: each camera starts face rec at different offset (0, 1, 2 sec)
    time.sleep(cam_id * 0.8)

    while running:
        try:
            _ensure_access_rules()

            with surv_locks[cam_id]:
                frame_bytes = surv_frames.get(cam_id)

            if frame_bytes is None or surv_model is None or not surv_online.get(cam_id):
                time.sleep(SURV_DETECT_INTERVAL)
                continue

            nparr = np.frombuffer(frame_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                time.sleep(SURV_DETECT_INTERVAL)
                continue

            # Person detection — use surv_model (separate from YOLO cam's model)
            # surv_model uses model() not track() to avoid corrupting YOLO's tracker
            with model_infer_lock:
                results = surv_model(img, classes=[0], verbose=False)
            raw_boxes = [b for b in results[0].boxes if float(b.conf[0]) >= MIN_CONFIDENCE]

            # Match current detections to previous tracks via IoU
            now = time.time()
            box_xyxy = [b.xyxy[0] for b in raw_boxes]
            match = _match_boxes(prev_tracks, box_xyxy)
            prev_tracks = [(box_xyxy[ci], match[ci]) for ci in sorted(match.keys())]

            track_ids = [match[i] for i in range(len(raw_boxes))]

            # Always buffer frames that contain people (for best-frame selection)
            if raw_boxes:
                gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                sharpness = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 300.0, 1.0)
                max_conf  = max(float(b.conf[0]) for b in raw_boxes)
                score     = max_conf * 0.6 + sharpness * 0.4
                with best_frame_locks[cam_id]:
                    best_frame_buffers[cam_id].append((score, img.copy(), list(raw_boxes)))
                    if len(best_frame_buffers[cam_id]) > BEST_FRAME_BUFFER_SIZE:
                        best_frame_buffers[cam_id].pop(0)

            if not raw_boxes:
                # Clear stale cam2 face list when no one is detected
                if cam_id == 2:
                    with cam2_face_lock:
                        cam2_face_list = []
                # Cleanup stale tracks
                surv_track_crops = {tid: buf for tid, buf in surv_track_crops.items()
                                    if tid in surv_track_ident}
                time.sleep(SURV_DETECT_INTERVAL)
                continue

            print(f"🔍 CAM{cam_id}: detected {len(raw_boxes)} person(s), max_conf={max_conf:.2f}, sharpness={sharpness:.2f}, score={score:.3f}")

            # ── Select best-scoring frame from buffer for face recognition ──
            rec_img = img
            rec_source = "current"
            with best_frame_locks[cam_id]:
                if best_frame_buffers[cam_id]:
                    best_entry = max(best_frame_buffers[cam_id], key=lambda x: x[0])
                    if best_entry[0] > 0.5:
                        rec_img = best_entry[1]
                        rec_source = f"best({best_entry[0]:.3f})"
                    else:
                        rec_source = f"current(best_in_buffer={best_entry[0]:.3f}<=0.5)"
            print(f"📸 CAM{cam_id}: frame source={rec_source}, buffer_size={len(best_frame_buffers[cam_id])}")

            # ── Per-track face recognition with crop buffer ──
            identified_any = False
            final_result   = None

            for tid, box in zip(track_ids, raw_boxes):
                if tid is None:
                    continue

                if cam_id in CAM_NO_FACE_RECOGNITION:
                    # Camera sees only legs/backs (exit cam) — skip face rec
                    surv_track_ident[tid] = {"name": None, "status": "unknown", "confidence": 0.0}
                    continue

                # Rate-limit face recognition per camera
                face_interval = SURV_FACE_INTERVALS.get(cam_id, 3.0)
                if now - last_face_time < face_interval:
                    continue
                last_face_time = now

                # Already identified in this camera session → skip
                if tid in surv_track_ident:
                    identified_any = True
                    final_result   = surv_track_ident[tid]
                    continue

                # Crop head region (from best available frame)
                face_img, ok = _crop_face_from_box(rec_img, box)
                if not ok:
                    continue

                result = recognize_face(face_img, access_rules_cache, cam_id=cam_id)
                r_name   = result.get("name")
                r_status = result.get("status", "unknown")

                if r_name and r_status != "unknown":
                    surv_track_ident[tid] = result
                    surv_track_crops.pop(tid, None)
                    surv_track_first_seen.pop(tid, None)  # cancel deferred alert
                    identified_any = True
                    final_result   = result
                    print(f"👁️  CAM{cam_id} TRACK {tid}: {r_name} ({r_status}) conf={result.get('confidence',0):.2f}")
                else:
                    if tid not in surv_track_first_seen:
                        surv_track_first_seen[tid] = now

                    # Buffer crop for retry
                    buf = surv_track_crops.setdefault(tid, [])
                    buf.append(face_img)
                    if len(buf) > SURV_TRACK_BUFFER_SIZE:
                        buf.pop(0)

                    # Retry all buffered crops
                    for idx, old_crop in enumerate(buf):
                        result2 = recognize_face(old_crop, access_rules_cache, cam_id=cam_id)
                        n2 = result2.get("name")
                        s2 = result2.get("status", "unknown")
                        if n2 and s2 != "unknown":
                            surv_track_ident[tid] = result2
                            surv_track_crops.pop(tid, None)
                            surv_track_first_seen.pop(tid, None)
                            identified_any = True
                            final_result   = result2
                            print(f"👁️  CAM{cam_id} TRACK {tid}: identified via buffer[{idx}] as {n2} ({s2})")
                            break

            # ── Feed into state machine + deferred unknown alert ──
            name   = final_result.get("name")   if final_result else None
            status = final_result.get("status") if final_result else "unknown"

            # ── CAM3 special: use YOLO body detection as EXIT trigger proxy ──
            if cam_id == 3 and not identified_any and track_ids:
                # CAM3 detected someone via YOLO but can't do face rec
                # → infer identity from the single inside occupant (if any)
                with person_state_lock:
                    inside = [(ff, st["name"]) for ff, st in person_state.items()
                              if st.get("inside")]
                if len(inside) == 1:
                    ff, nm = inside[0]
                    update_sighting(ff, nm, 3, confidence=0.0)
                    print(f"🚪 CAM3: inferred {nm} at exit (YOLO proxy)")

            if not name or status == "unknown":
                # Only fire unknown alert if we actually attempted face recognition
                # and found a face but couldn't identify (NOT when rate-limited or cropless)
                if final_result is not None:
                    fire_alert = False
                    for tid, fs in list(surv_track_first_seen.items()):
                        if now - fs >= UNKNOWN_DEFER_SEC:
                            fire_alert = True
                            surv_track_first_seen.pop(tid, None)
                    if fire_alert and cam_id == 1:
                        update_unknown_sighting(cam_id, img)
            else:
                face_folder = next(
                    (r["face_folder"] for r in access_rules_cache if r["name"] == name),
                    name,
                )
                conf = final_result.get("confidence", 0.0)
                print(f"👁️  CAM{cam_id} sees {name} ({status}) conf={conf:.2f}")

                if cam_id == 2:
                    update_sighting(face_folder, name, cam_id, confidence=conf)
                    update_occupancy_last_seen(name)
                else:
                    if _cast_vote(cam_id, face_folder):
                        update_sighting(face_folder, name, cam_id, confidence=conf)
                        update_occupancy_last_seen(name)

            # Update dashboard — cam_id==2 is the face-level camera (index 2)
            if cam_id == 2 and img is not None:
                threading.Thread(target=_save_cam2_capture,
                                 args=(img.copy(),), daemon=True).start()
                now_str = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                with cam2_face_lock:
                    cam2_face_list = [{
                        "name":        final_result.get("name") if final_result else None,
                        "status":      final_result.get("status", "unknown") if final_result else "unknown",
                        "confidence":  final_result.get("confidence", 0.0) if final_result else 0.0,
                        "detected_at": now_str,
                    }]

        except Exception as e:
            print(f"❌ SURV DETECT cam{cam_id}: {e}")
            import traceback
            traceback.print_exc()

        time.sleep(SURV_DETECT_INTERVAL)


def generate_yolo_stream():
    """Yield cached YOLO frames — one generator per HTTP client."""
    last_count = -1
    try:
        while running:
            count = yolo_frame_count
            if count != last_count:
                last_count = count
                with yolo_lock:
                    f = yolo_frame
                if f:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + f + b'\r\n')
            else:
                time.sleep(0.01)
    except GeneratorExit:
        pass


def init_yolo_camera():
    global yolo_cap, yolo_online
    print(f"\n🎯 初始化 YOLO 攝影機 (index {YOLO_CAM_IDX})...")
    try:
        c = cv2.VideoCapture(YOLO_CAM_IDX, cv2.CAP_DSHOW)
        if not c.isOpened():
            print(f"  ⚠️  YOLO CAM (index {YOLO_CAM_IDX}): 未找到")
            return
        c.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        c.set(cv2.CAP_PROP_FPS,          10)
        c.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        yolo_cap    = c
        yolo_online = True
        t = threading.Thread(target=yolo_capture_loop, daemon=True)
        t.start()
        print(f"  ✅ YOLO CAM (index {YOLO_CAM_IDX}): Online")
    except Exception as e:
        print(f"  ❌ YOLO CAM: Error - {e}")


def init_surveillance_cameras():
    global surv_caps, surv_online
    print("\n📡 初始化監控攝影機...")
    for cam_id in SURV_IDS:
        phys_idx = SURV_CAM_IDX[cam_id]
        try:
            c = cv2.VideoCapture(phys_idx, cv2.CAP_DSHOW)
            if c.isOpened():
                c.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                c.set(cv2.CAP_PROP_FPS,          5)
                c.set(cv2.CAP_PROP_BUFFERSIZE,   1)
                # Flush initial DirectShow frames — A4Tech cameras return black frames
                # for the first ~1 second before AGC/AWB stabilises.
                _flush_camera(c, n=20)
                surv_caps[cam_id]   = c
                surv_online[cam_id] = True
                t = threading.Thread(target=surv_capture_loop,
                                     args=(cam_id, phys_idx), daemon=True)
                t.start()
                print(f"  ✅ 攝影機 {cam_id} (index {phys_idx}): Online")
            else:
                surv_online[cam_id] = False
                c.release()
                print(f"  ⚠️  攝影機 {cam_id} (index {phys_idx}): 未找到 (Offline)")
            # Stagger camera opens — DirectShow can reject rapid sequential opens
            time.sleep(0.5)
        except Exception as e:
            surv_online[cam_id] = False
            print(f"  ❌ 攝影機 {cam_id}: Error - {e}")

def generate_surv_stream(cam_id):
    last_count = -1
    try:
        while running:
            if not surv_online.get(cam_id):
                time.sleep(0.1)
                continue
            count = surv_frame_counts.get(cam_id, 0)
            if count != last_count:
                last_count = count
                with surv_locks[cam_id]:
                    frame_bytes = surv_frames.get(cam_id)
                if frame_bytes:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                time.sleep(0.01)
    except GeneratorExit:
        pass

def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully shutdown the system"""
    global running, telegram_thread_pool
    print("\n🛑 收到停止信號，正在關閉系統...")
    running = False

    close_all_cameras()

    for thread in telegram_thread_pool:
        if thread.is_alive():
            thread.join(timeout=5)
    print("✅ Telegram 線程已清理")

    print("👋 系統已安全關閉")
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

# Ensure cameras are released on normal program exit
atexit.register(close_all_cameras)

def get_db_connection():
    return pymysql.connect(**DB_CONFIG)

def get_tank_db_connection():
    return pymysql.connect(**TANK_DB_CONFIG)

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Unauthorized'}), 401
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated

def init_users_table():
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            connection.commit()
        print("✅ DB: users table ready")
    except Exception as e:
        print(f"❌ DB: Failed to init users table - {e}")
    finally:
        connection.close()

def init_detection_logs():
    os.makedirs(os.path.join("detection", "captures"), exist_ok=True)
    os.makedirs("faces_db", exist_ok=True)
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detection_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    person_count INT DEFAULT 1,
                    image_filename VARCHAR(255),
                    person_name VARCHAR(100) DEFAULT NULL,
                    access_status ENUM('authorized','unauthorized','unknown') DEFAULT 'unknown'
                )
            """)
            # Add columns to existing table if they were created before this update
            for col_sql in [
                "ALTER TABLE detection_logs ADD COLUMN person_name VARCHAR(100) DEFAULT NULL",
                "ALTER TABLE detection_logs ADD COLUMN access_status ENUM('authorized','unauthorized','unknown') DEFAULT 'unknown'",
            ]:
                try:
                    cursor.execute(col_sql)
                except Exception:
                    pass
            connection.commit()
        print("✅ DB: detection_logs table ready")
    except Exception as e:
        print(f"❌ DB: Failed to init detection_logs - {e}")
    finally:
        connection.close()


def init_room_tables():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS room_events (
                    id          INT AUTO_INCREMENT PRIMARY KEY,
                    event_type  ENUM('ENTER','EXIT') NOT NULL,
                    person_name VARCHAR(100),
                    face_folder VARCHAR(100),
                    cam_id      INT,
                    timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS room_occupants_log (
                    face_folder VARCHAR(100) PRIMARY KEY,
                    person_name VARCHAR(100),
                    entered_at  DATETIME
                )
            """)
            conn.commit()
        print("✅ DB: room tables ready")
    except Exception as e:
        print(f"❌ DB: room tables init failed - {e}")
    finally:
        conn.close()


def log_room_event(event_type, name, face_folder, cam_id):
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO room_events (event_type, person_name, face_folder, cam_id) "
                "VALUES (%s, %s, %s, %s)",
                (event_type, name, face_folder, cam_id)
            )
            if event_type == "ENTER":
                cur.execute(
                    "REPLACE INTO room_occupants_log (face_folder, person_name, entered_at) "
                    "VALUES (%s, %s, NOW())",
                    (face_folder, name)
                )
            else:
                cur.execute(
                    "DELETE FROM room_occupants_log WHERE face_folder = %s",
                    (face_folder,)
                )
            conn.commit()
    except Exception as e:
        print(f"❌ DB: log_room_event failed - {e}")
    finally:
        conn.close()


def _save_capture(frame_img, prefix="capture"):
    """Save a frame to detection/captures/ and return the path, or None."""
    if frame_img is None:
        return None
    try:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{ts}.jpg"
        path     = os.path.join("detection", "captures", filename)
        cv2.imwrite(path, frame_img)
        return path
    except Exception:
        return None


def update_sighting(face_folder, name, cam_id, confidence=0.0):
    """Record that cam_id just saw face_folder. Called by all detection loops."""
    with person_state_lock:
        if face_folder not in person_state:
            person_state[face_folder] = {
                "name":            name,
                "inside":          False,
                "entered_at":      None,
                "entered_at_timestamp": None,
                "last_seen":       {1: 0, 2: 0, 3: 0, 'yolo': 0},
                "last_confidence": {1: 0.0, 2: 0.0, 3: 0.0, 'yolo': 0.0},
                "exit_pending":    False,
                "exit_started_at": 0,
            }
        st = person_state[face_folder]
        st['last_seen'][cam_id]       = time.time()
        st['last_confidence'][cam_id] = confidence
        st['name']                    = name


def update_unknown_sighting(cam_id, frame_img=None):
    """Alert once per UNKNOWN_COOLDOWN_SEC per camera when an unknown person is seen."""
    now = time.time()
    with unknown_alert_lock:
        if now - unknown_last_alert.get(cam_id, 0) < UNKNOWN_COOLDOWN_SEC:
            return
        unknown_last_alert[cam_id] = now

    img_path  = _save_capture(frame_img, f"unknown_cam{cam_id}")
    cam_label = {1: "入口", 2: "室內", 3: "出口", 'yolo': "全景"}.get(cam_id, str(cam_id))
    msg = (f"⚠️ 發現不明人士！\n"
           f"📷 攝影機: {cam_label}\n"
           f"🕐 {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}\n"
           f"📍 IoT Lab")
    send_telegram_notification(msg, img_path)
    print(f"⚠️ UNKNOWN: cam{cam_id} alert sent")


def _cast_vote(cam_id, face_folder):
    """
    Add one recognition vote for face_folder at cam_id.
    Returns True when vote count reaches VOTE_REQUIRED within VOTE_WINDOW_SEC.
    Only used for cam1 and cam3 (cam2 updates presence immediately).
    """
    now = time.time()
    with _vote_lock:
        bucket = _vote_records.setdefault(cam_id, {})
        timestamps = [t for t in bucket.get(face_folder, [])
                      if now - t < VOTE_WINDOW_SEC]
        timestamps.append(now)
        bucket[face_folder] = timestamps
        count = len(timestamps)
    reached = count >= VOTE_REQUIRED
    if reached:
        print(f"🗳️  CAM{cam_id} VOTE THRESHOLD REACHED: {face_folder} ({count}/{VOTE_REQUIRED}) — proceeding")
        with _vote_lock:
            bucket[face_folder] = []  # reset votes after threshold
    else:
        print(f"🗳️  CAM{cam_id} vote: {face_folder} = {count}/{VOTE_REQUIRED}")
    return reached


_last_enter_notify = {}
_enter_notify_lock = threading.Lock()
ENTER_NOTIFY_COOLDOWN = 60  # seconds between Telegram ENTER alerts per person

def _do_notify_enter(name, face_folder):
    # Debounce: skip Telegram if notified recently for same person
    with _enter_notify_lock:
        last = _last_enter_notify.get(face_folder, 0)
        if time.time() - last < ENTER_NOTIFY_COOLDOWN:
            log_room_event("ENTER", name, face_folder, 1)
            print(f"🚪 ENTER logged: {name} (Telegram skipped — cooldown)")
            return
        _last_enter_notify[face_folder] = time.time()

    ts  = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    # Check authorization status
    rule = next(
        (r for r in access_rules_cache if r.get("face_folder") == face_folder),
        None,
    )
    now_str = datetime.now().strftime("%H:%M")
    is_active = rule.get("is_active", True) if rule else True
    start = rule.get("access_start", "00:00") if rule else "00:00"
    end   = rule.get("access_end",   "23:59") if rule else "23:59"
    authorized = is_active and (start <= now_str <= end)
    icon = "✅" if authorized else "⚠️"
    status_tag = f" ({'授權' if authorized else '未授權'})"
    msg = f"{icon} {name} 進入實驗室{status_tag}\n🕐 {ts}\n📍 IoT Lab"
    send_telegram_notification(msg)
    log_room_event("ENTER", name, face_folder, 1)
    print(f"🚪 ENTER logged: {name}")


def _do_notify_exit(name, face_folder, duration_sec):
    ts   = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    mins = int(duration_sec // 60)
    msg  = (f"🚶 {name} 離開實驗室\n"
            f"🕐 {ts}\n"
            f"⏱ 停留 {mins} 分鐘\n"
            f"📍 IoT Lab")
    send_telegram_notification(msg)
    log_room_event("EXIT", name, face_folder, 3)
    print(f"🚪 EXIT logged: {name} (duration {mins} min)")


# ── Event-Based Occupancy Functions ────────────────────────────

def record_entry(name):
    """เรียกเมื่อ ENTER event ยืนยันแล้ว — เพิ่มคนใน occupancy"""
    with occupancy_lock:
        if name not in room_occupants:
            room_occupants[name] = {
                "entry_time": time.time(),
                "last_seen":  time.time(),
            }
            print(f"✅ OCCUPANCY: {name} ENTER — ตอนนี้ {len(room_occupants)} คนในห้อง")


def record_exit(name):
    """เรียกเมื่อ EXIT event ยืนยันแล้ว — ลบคนออกจาก occupancy"""
    with occupancy_lock:
        if name in room_occupants:
            del room_occupants[name]
            print(f"🚪 OCCUPANCY: {name} EXIT — ตอนนี้ {len(room_occupants)} คนในห้อง")


def update_occupancy_last_seen(name):
    """อัปเดต last_seen — เรียกทุกครั้งที่กล้องเห็นคนที่อยู่ในห้อง"""
    with occupancy_lock:
        if name in room_occupants:
            room_occupants[name]["last_seen"] = time.time()


def get_occupancy():
    """คืนค่า occupancy ปัจจุบัน (ใช้ใน dashboard route)"""
    with occupancy_lock:
        return dict(room_occupants)


def _cleanup_stale_occupants():
    """
    Safety net: ถ้าคน inside อยู่นานแต่ไม่ถูกกล้องตรวจจับเกิน STALE_OCCUPANT_SEC
    → ถือว่าออกไปแล้ว (ป้องกัน count ค้าง) และบันทึก EXIT event
    """
    now = time.time()
    stale = []
    with person_state_lock:
        for ff, st in list(person_state.items()):
            if not st.get("inside"):
                continue
            # check if ANY camera has seen this person recently
            any_seen = any(
                now - ts < STALE_OCCUPANT_SEC
                for ts in st.get("last_seen", {}).values()
                if ts > 0
            )
            if not any_seen:
                stale.append((ff, st["name"]))
                st["inside"]          = False
                st["entered_at"]      = None
                st["exit_pending"]    = False
                st["exit_started_at"] = 0
    for ff, name in stale:
        record_exit(name)  # also removes from room_occupants
        print(f"⏱️ OCCUPANCY STALE: {name} ถูกลบ (ไม่เห็นนานเกิน)")


def _save_event_capture(event_type):
    """
    Save the latest event capture image from the correct camera.
    ENTER → save cam1's frame (entrance door, index 1 = 攝影機 2 in UI)
    EXIT  → save YOLO cam's frame (door overview, index 0 = 攝影機 1 in UI)
    """
    global entry_event_capture, exit_event_capture, latest_event_type
    captures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "detection", "captures")
    os.makedirs(captures_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with event_capture_lock:
        if event_type == "ENTER":
            frame = latest_raw_frames.get(1)
            if frame is not None:
                path = os.path.join(captures_dir, f"entry_cam1_{ts}.jpg")
                cv2.imwrite(path, frame)
                entry_event_capture = path
        elif event_type == "EXIT":
            frame = latest_raw_frames.get('yolo')
            if frame is not None:
                path = os.path.join(captures_dir, f"exit_yolo_{ts}.jpg")
                cv2.imwrite(path, frame)
                exit_event_capture = path


def unified_state_machine():
    """
    Background thread: sole decision-maker for ENTER/EXIT events.
    2-Stage EXIT: cam3 triggers → wait → cam2 confirms departure.
    """
    global latest_event_type, latest_event_person
    while running:
        try:
            now = time.time()
            with person_state_lock:
                for face_folder, state in list(person_state.items()):
                    cam1      = now - state['last_seen'].get(1,      0) < ENTRY_WINDOW_SEC
                    cam2      = now - state['last_seen'].get(2,      0) < ENTRY_WINDOW_SEC
                    cam3      = now - state['last_seen'].get(3,      0) < EXIT_WINDOW_SEC

                    # ── ENTER: Cam1 จ่อประตู alone, Cam2+Cam1 cross-confirm, or Cam2 solo ──
                    if not state['inside']:
                        cam1_conf  = state.get('last_confidence', {}).get(1, 0.0)
                        cam2_conf  = state.get('last_confidence', {}).get(2, 0.0)
                        cam1_solo  = cam1 and cam1_conf >= 0.40
                        cam2_solo  = cam2 and cam2_conf >= CAM2_SOLO_CONFIDENCE
                        cross_conf = cam1 and cam2
                        print(f"🏠 STATE {face_folder}: inside={state['inside']} cam1={cam1}({cam1_conf:.2f}) cam2={cam2}({cam2_conf:.2f}) cam3={cam3} → cam1_solo={cam1_solo} cam2_solo={cam2_solo} cross={cross_conf}")
                        if cam1_solo or cam2_solo or cross_conf:
                            state['inside']       = True
                            state['entered_at']   = datetime.now()
                            state['entered_at_timestamp'] = time.time()
                            state['exit_pending'] = False
                            name   = state['name']
                            reason = ("cam1 solo" if cam1_solo else
                                      "cam2 solo" if cam2_solo else
                                      "cam1+cam2 cross")
                            print(f"🚪 ENTER trigger: {name} ({reason})")
                            record_entry(name)
                            _save_event_capture("ENTER")
                            with event_capture_lock:
                                latest_event_type   = "ENTER"
                                latest_event_person = name
                            threading.Thread(target=_do_notify_enter,
                                             args=(name, face_folder), daemon=True).start()

                    # ── EXIT: 2-Stage — Cam3 triggers, Cams confirm ──────────
                    else:
                        # Stage 1: Cam3 sees person near exit → start exit watch
                        if cam3 and not state['exit_pending']:
                            state['exit_pending']    = True
                            state['exit_started_at'] = now
                            print(f"🚶 EXIT PENDING: {state['name']} — รอ確認離開")

                        # Stage 2: After delay, check if any cam still sees them
                        if state['exit_pending']:
                            waited      = now - state['exit_started_at']
                            entered_ago = now - state.get('entered_at_timestamp', 0)

                            # Check all cameras (cam1, cam2, yolo) for presence
                            cam1_still_sees = now - state['last_seen'].get(1, 0) < ENTRY_WINDOW_SEC
                            yolo_still_sees = now - state['last_seen'].get('yolo', 0) < ENTRY_WINDOW_SEC
                            cam2_still_sees = cam2
                            still_sees = cam1_still_sees or cam2_still_sees or yolo_still_sees

                            # Grace period: if just entered (< 30s), don't allow EXIT
                            MIN_STAY_SEC = 30
                            just_entered = entered_ago < MIN_STAY_SEC
                            if just_entered and still_sees:
                                # Ignore exit trigger — person probably just walked past cam3
                                state['exit_pending']    = False
                                state['exit_started_at'] = 0
                                print(f"↩️ EXIT IGNORED (grace): {state['name']} entered {entered_ago:.0f}s ago")
                                continue

                            if waited >= EXIT_CONFIRM_DELAY_SEC:
                                if not still_sees:
                                    # no camera sees them → person has left the room
                                    dur  = (datetime.now() - state['entered_at']).total_seconds() \
                                           if state['entered_at'] else 0
                                    name = state['name']
                                    state['inside']          = False
                                    state['entered_at']      = None
                                    state['entered_at_timestamp'] = None
                                    state['exit_pending']    = False
                                    state['exit_started_at'] = 0
                                    _save_event_capture("EXIT")
                                    with event_capture_lock:
                                        latest_event_type   = "EXIT"
                                        latest_event_person = name
                                    record_exit(name)
                                    reason = "所有鏡頭都看不見"
                                    print(f"🚪 EXIT CONFIRMED: {name} ({reason})")
                                    threading.Thread(target=_do_notify_exit,
                                                     args=(name, face_folder, dur), daemon=True).start()
                                else:
                                    # still sees → cancel EXIT
                                    src = "CAM1" if cam1_still_sees else ("CAM2" if cam2_still_sees else "YOLO")
                                    state['exit_pending']    = False
                                    state['exit_started_at'] = 0
                                    print(f"↩️ EXIT CANCELLED: {state['name']} — {src} ยังเห็นคนอยู่")
                            elif waited > EXIT_MAX_WAIT_SEC:
                                # Timeout fallback — force EXIT
                                dur  = (datetime.now() - state['entered_at']).total_seconds() \
                                       if state['entered_at'] else 0
                                name = state['name']
                                state['inside']          = False
                                state['entered_at']      = None
                                state['entered_at_timestamp'] = None
                                state['exit_pending']    = False
                                state['exit_started_at'] = 0
                                _save_event_capture("EXIT")
                                with event_capture_lock:
                                    latest_event_type   = "EXIT"
                                    latest_event_person = name
                                record_exit(name)
                                print(f"⏱️ EXIT TIMEOUT: {name} (รอนานเกิน {EXIT_MAX_WAIT_SEC}s)")
                                threading.Thread(target=_do_notify_exit,
                                                 args=(name, face_folder, dur), daemon=True).start()

        except Exception as e:
            print(f"❌ STATE MACHINE: error - {e}")
        time.sleep(1)


def init_authorized_personnel_table():
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS authorized_personnel (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    employee_id VARCHAR(50) UNIQUE,
                    role VARCHAR(100) DEFAULT '',
                    face_folder VARCHAR(100) DEFAULT '',
                    access_start TIME DEFAULT '08:00:00',
                    access_end TIME DEFAULT '17:00:00',
                    is_active TINYINT(1) DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            connection.commit()
        print("✅ DB: authorized_personnel table ready")
    except Exception as e:
        print(f"❌ DB: Failed to init authorized_personnel table - {e}")
    finally:
        connection.close()


def normalize_sensor_data(row):
    if not row:
        return row
    normalized = row.copy()
    if 'ph' in row:
        normalized['pH'] = row['ph']
    if 'oxygen' in row:
        normalized['DO_mgl'] = row['oxygen']
    if 'temperature' in row:
        normalized['Temperature_C'] = row['temperature']
    # Handle capital-T Timestamp column (actual DB column name)
    if 'Timestamp' in row and 'timestamp' not in row:
        normalized['timestamp'] = row['Timestamp']
    return normalized

def send_telegram_notification(message, image_path=None):
    """Send notification to Telegram in a separate thread with rate limiting"""
    global telegram_thread_pool, max_telegram_threads

    # Clean up finished threads
    telegram_thread_pool = [t for t in telegram_thread_pool if t.is_alive()]

    # Check if we have too many active threads
    if len(telegram_thread_pool) >= max_telegram_threads:
        print("⚠️ TELEGRAM: 太多活躍線程，跳過此次發送")
        return

    def send_async():
        try:
            if image_path:
                url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
                with open(image_path, "rb") as img:
                    response = requests.post(url,
                        data={"chat_id": CHAT_ID, "caption": message},
                        files={"photo": img}, timeout=15)
                if response.status_code == 200:
                    print("✅ TELEGRAM: 圖片通知發送成功")
                else:
                    print(f"❌ TELEGRAM: 發送失敗，狀態碼: {response.status_code}")
            else:
                url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
                response = requests.post(url,
                    data={"chat_id": CHAT_ID, "text": message}, timeout=10)
                if response.status_code == 200:
                    print("✅ TELEGRAM: 文字通知發送成功")
                else:
                    print(f"❌ TELEGRAM: 發送失敗，狀態碼: {response.status_code}")
        except Exception as e:
            print(f"❌ TELEGRAM: 發送失敗 - {e}")
        finally:
            try:
                telegram_thread_pool.remove(threading.current_thread())
            except ValueError:
                pass

    # Send in background thread to avoid blocking
    thread = threading.Thread(target=send_async, daemon=True)
    telegram_thread_pool.append(thread)
    thread.start()

@app.route('/api/send-fault-alert', methods=['POST'])
@login_required
def send_fault_alert():
    try:
        data = request.get_json()
        tank_id = data.get('tank_id', '?')
        sensor  = data.get('sensor', '')
        message = (
            f"🔌 感測器異常！\n"
            f"🐟 魚缸 {tank_id} - {sensor} 持續回傳 0\n"
            f"⚠️ 感測器可能已斷線或損壞，請立即檢查\n"
            f"🕐 {time.strftime('%Y/%m/%d %H:%M:%S')}\n"
            f"📍 魚菜共生監控系統"
        )
        send_telegram_notification(message)
        return jsonify({'ok': True})
    except Exception as e:
        print(f"❌ FAULT ALERT: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/send-trend-alert', methods=['POST'])
@login_required
def send_trend_alert():
    try:
        data = request.get_json()
        tank_id  = data.get('tank_id', '?')
        sensor   = data.get('sensor', '')
        value    = data.get('value', '')
        direction = data.get('direction', '')
        eta_sec  = data.get('eta_sec', 0)
        message = (
            f"⚠️ 趨勢預警！\n"
            f"{direction}\n"
            f"🐟 魚缸 {tank_id} - {sensor}：{value}\n"
            f"⏱ 預計約 {eta_sec} 秒後超出標準範圍\n"
            f"🕐 {time.strftime('%Y/%m/%d %H:%M:%S')}\n"
            f"📍 魚菜共生監控系統"
        )
        send_telegram_notification(message)
        return jsonify({'ok': True})
    except Exception as e:
        print(f"❌ TREND ALERT: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sensor-data')
@login_required
def get_sensor_data():
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM sensordata ORDER BY timestamp DESC LIMIT 1")
            result = cursor.fetchone()
        print("✅ GET DATA: Sensor data retrieved successfully")
        return jsonify(normalize_sensor_data(result) or {})
    except Exception as e:
        print(f"❌ GET DATA: Failed to retrieve sensor data - {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        connection.close()

@app.route('/api/detection-stats')
@login_required
def get_detection_stats():
    try:
        with lock:
            stats = {
                'current_people': current_person_count,
                'daily_total': daily_detection_total,
                'last_detection': last_detection_timestamp
            }
        return jsonify(stats)
    except Exception as e:
        print(f"❌ GET DATA: Failed to retrieve detection stats - {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/room/status')
@login_required
def get_room_status():
    with person_state_lock:
        occupants = [
            {
                "face_folder": ff,
                "name":        st["name"],
                "entered_at":  st["entered_at"].strftime("%Y-%m-%d %H:%M:%S")
                               if st.get("entered_at") else None,
            }
            for ff, st in person_state.items()
            if st.get("inside")
        ]
    return jsonify({"count": len(occupants), "occupants": occupants})


@app.route('/api/occupancy')
@login_required
def api_occupancy():
    """Event-based occupancy (from ENTER/EXIT events, more reliable than camera detection)"""
    occ = get_occupancy()
    now = time.time()
    return jsonify({
        "count": len(occ),
        "persons": [
            {
                "name":       name,
                "entry_time": datetime.fromtimestamp(info["entry_time"]).strftime("%Y/%m/%d %H:%M:%S"),
                "last_seen":  datetime.fromtimestamp(info["last_seen"]).strftime("%Y/%m/%d %H:%M:%S"),
                "seconds_since_seen": int(now - info["last_seen"]),
            }
            for name, info in occ.items()
        ]
    })

@app.route('/api/room/events')
@login_required
def get_room_events():
    conn = None
    try:
        limit = min(int(request.args.get("limit", 50)), 200)
        conn  = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, event_type, person_name, face_folder, cam_id, "
                "DATE_FORMAT(timestamp, '%%Y/%%m/%%d %%H:%%i:%%S') AS timestamp "
                "FROM room_events ORDER BY timestamp DESC LIMIT %s",
                (limit,)
            )
            events = cur.fetchall()
        return jsonify({"events": events})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


@app.route('/api/face-photo/<face_folder>')
@login_required
def get_face_photo_by_folder(face_folder):
    """Serve the first available face photo for a given face_folder slug."""
    import re
    if not re.match(r'^[\w\-]+$', face_folder):
        return jsonify({"error": "invalid"}), 400
    path = get_first_photo_path(face_folder)
    if path and os.path.isfile(path):
        return send_file(path, mimetype='image/jpeg')
    return jsonify({"error": "no photo"}), 404


@app.route('/api/unknown-alerts')
@login_required
def get_unknown_alerts():
    """Return list of unknown-person captures from 攝影機 2 (cam1), newest first."""
    import glob as _glob
    captures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "detection", "captures")
    files = sorted(
        _glob.glob(os.path.join(captures_dir, "unknown_cam1_*.jpg")),
        key=os.path.getmtime, reverse=True
    )[:50]
    results = []
    for f in files:
        fname = os.path.basename(f)
        mtime = os.path.getmtime(f)
        results.append({
            "filename": fname,
            "url":      f"/detection-image/{fname}",
            "time":     datetime.fromtimestamp(mtime).strftime("%Y/%m/%d %H:%M:%S"),
        })
    return jsonify({"alerts": results})


@app.route('/api/room/history')
@login_required
def get_room_history():
    conn = None
    try:
        page     = max(1, int(request.args.get('page', 1)))
        per_page = min(int(request.args.get('per_page', 20)), 50)
        offset   = (page - 1) * per_page
        search   = request.args.get('search', '').strip()
        date_from = request.args.get('date_from', '')
        date_to   = request.args.get('date_to', '')

        where_parts = ["e.event_type = 'ENTER'"]
        params      = []
        if search:
            where_parts.append("e.person_name LIKE %s")
            params.append(f'%{search}%')
        if date_from:
            where_parts.append("DATE(e.timestamp) >= %s")
            params.append(date_from)
        if date_to:
            where_parts.append("DATE(e.timestamp) <= %s")
            params.append(date_to)
        where_sql = " AND ".join(where_parts)

        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) AS total FROM room_events e WHERE {where_sql}", params)
            total = cur.fetchone()['total']

            cur.execute(f"""
                SELECT
                    e.person_name,
                    e.face_folder,
                    e.cam_id                                                     AS entry_cam,
                    DATE_FORMAT(e.timestamp, '%%Y/%%m/%%d %%H:%%i:%%S')         AS entered_at,
                    x.cam_id                                                     AS exit_cam,
                    DATE_FORMAT(x.timestamp, '%%Y/%%m/%%d %%H:%%i:%%S')         AS exited_at,
                    TIMESTAMPDIFF(MINUTE, e.timestamp, x.timestamp)              AS duration_min,
                    (
                        SELECT dl.image_filename FROM detection_logs dl
                        WHERE dl.timestamp >= e.timestamp - INTERVAL 5 SECOND
                          AND dl.timestamp <= e.timestamp + INTERVAL 15 SECOND
                        ORDER BY ABS(TIMESTAMPDIFF(SECOND, e.timestamp, dl.timestamp))
                        LIMIT 1
                    )                                                            AS image_filename
                FROM room_events e
                LEFT JOIN room_events x
                    ON  x.face_folder  = e.face_folder
                    AND x.event_type   = 'EXIT'
                    AND x.timestamp    = (
                        SELECT MIN(r.timestamp) FROM room_events r
                        WHERE  r.face_folder = e.face_folder
                        AND    r.event_type  = 'EXIT'
                        AND    r.timestamp   > e.timestamp
                    )
                WHERE {where_sql}
                ORDER BY e.timestamp DESC
                LIMIT %s OFFSET %s
            """, params + [per_page, offset])
            rows = cur.fetchall()
            for r in rows:
                fn = r.get("image_filename")
                r["image_url"] = f"/detection-image/{fn}" if fn else None

        return jsonify({
            'records':     rows,
            'total':       total,
            'page':        page,
            'per_page':    per_page,
            'total_pages': max(1, (total + per_page - 1) // per_page),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


@app.route('/api/face/latest')
@login_required
def get_latest_face():
    # Prefer identified results; fall back to any available
    with cam2_face_lock:
        cam2 = list(cam2_face_list) if cam2_face_list else []
    with latest_face_lock:
        yolo = list(latest_face_list) if latest_face_list else []

    # CAM2 always has "unknown" even when it can't identify → check YOLO too
    cam2_has_known = any(p.get("status") != "unknown" for p in cam2)
    yolo_has_known = any(p.get("status") != "unknown" for p in yolo)

    if cam2_has_known:
        people = cam2
    elif yolo_has_known:
        people = yolo
    else:
        people = cam2 or yolo  # fallback to whatever exists

    enriched = []
    for p in people:
        item = p.copy()
        if p.get("name"):
            conn = None
            try:
                conn = get_db_connection()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT employee_id, role, access_start, access_end "
                        "FROM authorized_personnel WHERE name=%s",
                        (p["name"],)
                    )
                    row = cur.fetchone()
                if row:
                    item["employee_id"]  = row.get("employee_id") or "-"
                    item["role"]         = row.get("role") or "-"
                    item["access_start"] = _ts_fmt(row.get("access_start"))
                    item["access_end"]   = _ts_fmt(row.get("access_end"))
            except Exception:
                pass
            finally:
                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass
        enriched.append(item)

    # Filter: ถ้ามีคนที่ identify ได้ → แสดงเฉพาะคนรู้จัก
    known = [p for p in enriched if p.get("status") != "unknown"]
    if known:
        enriched = known

    # ถ้าไม่มีคนที่ตรวจจับได้ → ไม่ส่ง capture_url (ให้หน้าเว็บแสดง placeholder)
    if not enriched:
        return jsonify({
            "people":       [],
            "capture_url":  None,
            "event_type":   None,
            "event_person": None,
        })

    # Return capture URL based on latest event type
    with event_capture_lock:
        ev_type   = latest_event_type
        ev_person = latest_event_person
    if ev_type == "ENTER":
        capture_url = "/api/face/latest/capture/entry"
    elif ev_type == "EXIT":
        capture_url = "/api/face/latest/capture/exit"
    else:
        # Fallback to cam2 live capture
        with cam2_capture_lock:
            capture_url = "/api/face/latest/capture" if cam2_capture_path else None
    return jsonify({
        "people":       enriched,
        "capture_url":  capture_url,
        "event_type":   ev_type,
        "event_person": ev_person,
    })

@app.route('/api/face/latest/capture')
@login_required
def get_latest_capture():
    with cam2_capture_lock:
        path = cam2_capture_path
    if path and os.path.isfile(path):
        return send_file(path, mimetype='image/jpeg')
    return jsonify({"error": "No cam2 capture yet"}), 404

@app.route('/api/face/latest/capture/entry')
@login_required
def get_entry_capture():
    with event_capture_lock:
        path = entry_event_capture
    if path and os.path.isfile(path):
        return send_file(path, mimetype='image/jpeg')
    return jsonify({"error": "No entry capture yet"}), 404

@app.route('/api/face/latest/capture/exit')
@login_required
def get_exit_capture():
    with event_capture_lock:
        path = exit_event_capture
    if path and os.path.isfile(path):
        return send_file(path, mimetype='image/jpeg')
    return jsonify({"error": "No exit capture yet"}), 404

@app.route('/api/face/capture-entry')
@login_required
def capture_entry_frame():
    """Return a live JPEG from the requested camera source.
    Query param: ?cam=yolo|cam1|cam2|cam3  (default yolo)
    Uses per-camera frame buffers (surv_frames / yolo_frame) directly.
    """
    source = request.args.get('cam', 'yolo')
    buf = None
    if source == 'yolo':
        with yolo_lock:
            buf = yolo_frame
    elif source == 'cam1':
        with surv_locks[1]:
            buf = surv_frames.get(1)
    elif source == 'cam2':
        with surv_locks[2]:
            buf = surv_frames.get(2)
    elif source == 'cam3':
        with surv_locks[3]:
            buf = surv_frames.get(3)
    if buf is None:
        return jsonify({"error": f"Camera '{source}' not ready"}), 503
    return Response(buf, mimetype='image/jpeg')


@app.route('/api/face/capture-guide')
@login_required
def capture_guide():
    """Return a JPEG with face bounding box and position guidance drawn on it,
    or JSON with analysis data when ?fmt=json is set."""
    source = request.args.get('cam', 'yolo')
    fmt    = request.args.get('fmt')
    img = None
    if source == 'yolo':
        with yolo_lock:
            if yolo_frame is not None:
                img = cv2.imdecode(np.frombuffer(yolo_frame, np.uint8), cv2.IMREAD_COLOR)
    elif source in ('cam1', 'cam2', 'cam3'):
        cid = int(source[-1])
        with surv_locks[cid]:
            buf = surv_frames.get(cid)
        if buf is not None:
            img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        if fmt == 'json':
            return jsonify({"detected": False, "guidance": "相機尚未就緒", "ready": False})
        return jsonify({"error": f"Camera '{source}' not ready"}), 503

    from detection.face_utils import analyze_face_for_guide
    guide = analyze_face_for_guide(img)

    if fmt == 'json':
        return jsonify(guide)

    bbox = guide.get("bbox")
    if bbox:
        x1, y1, x2, y2 = bbox
        color = (0, 200, 0) if guide["ready"] else (0, 0, 220)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cx, cy = guide.get("center", (0, 0))
        cv2.circle(img, (cx, cy), 5, color, -1)

    text = guide["guidance"]
    cv2.rectangle(img, (0, img.shape[0] - 40), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    cv2.putText(img, text, (20, img.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    ret, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ret:
        return jsonify({"error": "Encoding failed"}), 500
    return Response(buf.tobytes(), mimetype='image/jpeg')


@app.route('/api/sensor-data/tank/<int:tank_id>')
@login_required
def get_tank_sensor_data(tank_id):
    if tank_id not in VALID_TANKS:
        return jsonify({'error': 'Invalid tank ID'}), 400
    table = VALID_TANKS[tank_id]
    connection = get_tank_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT * FROM `{table}` ORDER BY timestamp DESC LIMIT 1")
            result = cursor.fetchone()
        normalized = normalize_sensor_data(result) or {}
        if result and result.get('Timestamp'):
            age = (datetime.now() - result['Timestamp']).total_seconds()
            if age > 15:
                normalized['stale'] = True
        print(f"✅ GET DATA: {table} data retrieved successfully")
        return jsonify(normalized)
    except Exception as e:
        print(f"❌ GET DATA: Failed to retrieve {table} data - {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        connection.close()

@app.route('/api/sensor-history/<int:hours>')
def get_sensor_history(hours):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT * FROM sensordata
                WHERE timestamp >= DATE_SUB(NOW(), INTERVAL %s HOUR)
                ORDER BY timestamp ASC
            """, (hours,))
            results = cursor.fetchall()
        normalized_results = [normalize_sensor_data(row) for row in results]
        print(f"✅ GET DATA: Retrieved {len(normalized_results)} records for last {hours} hours")
        return jsonify(normalized_results)
    except Exception as e:
        print(f"❌ GET DATA: Failed to retrieve history data - {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        connection.close()


@app.after_request
def add_no_cache_headers(response):
    if 'multipart/x-mixed-replace' in response.content_type:
        return response
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


def init_system():
    global model, surv_model
    print("🤖 加載 YOLO 模型...")
    try:
        model = YOLO("yolov8n.pt")
        surv_model = YOLO("yolov8n.pt")
        print("✅ AI DETECTION: YOLO 模型加載成功 (YOLO cam + Surv cams แยก instance)")
        init_detection_logs()
        init_users_table()
        init_authorized_personnel_table()
        init_room_tables()
        _ensure_access_rules()
        _restore_person_state()
        init_yolo_camera()
        init_surveillance_cameras()
        # Prebuild FACE DB cache before starting recognition threads
        from detection.face_utils import _build_db
        _build_db()
        print("✅ FACE DB: prebuilt at startup")

        t = threading.Thread(target=face_recognition_loop, daemon=True)
        t.start()
        print("✅ FACE: Recognition thread started")
        for cid in [1, 2, 3]:
            t2 = threading.Thread(target=surv_detect_loop, args=(cid,), daemon=True)
            t2.start()
            print(f"✅ ROOM DETECT: cam{cid} thread started")
        t3 = threading.Thread(target=unified_state_machine, daemon=True)
        t3.start()
        print("✅ STATE MACHINE: unified access control thread started")
        t4 = threading.Thread(target=cleanup_old_captures, daemon=True)
        t4.start()
        print("✅ CLEANUP: capture cleanup thread started")
        return True
    except Exception as e:
        print(f"❌ SYSTEM: 初始化失敗 - {e}")
        return False

@app.route('/personnel')
@login_required
def personnel_page():
    return send_file('personnel.html')

@app.route('/api/personnel', methods=['GET'])
@login_required
def get_personnel():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, name, employee_id, role, face_folder, "
                "access_start, access_end, is_active, created_at "
                "FROM authorized_personnel ORDER BY created_at DESC"
            )
            rows = cur.fetchall()

        result = []
        for r in rows:
            folder = r.get("face_folder", "")
            r["access_start"] = _ts_fmt(r.get("access_start"))
            r["access_end"]   = _ts_fmt(r.get("access_end"))
            r["is_active"]    = bool(r.get("is_active"))
            photos            = list_face_photos(folder) if folder else []
            r["has_face"]     = len(photos) > 0
            r["photo_count"]  = len(photos)
            if r.get("created_at"):
                r["created_at"] = r["created_at"].strftime("%Y-%m-%d %H:%M:%S")
            result.append(r)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/personnel', methods=['POST'])
@login_required
def add_personnel():
    name        = (request.form.get("name") or "").strip()
    employee_id = (request.form.get("employee_id") or "").strip()
    role        = (request.form.get("role") or "").strip()
    acc_start   = (request.form.get("access_start") or "08:00").strip()
    acc_end     = (request.form.get("access_end") or "17:00").strip()

    if not name:
        return jsonify({"error": "請填寫姓名"}), 400

    face_folder = re.sub(r'[\\/:*?"<>| ]', "_", employee_id or name).strip("_") or "person"

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if employee_id:
                cur.execute(
                    "SELECT id FROM authorized_personnel WHERE employee_id = %s",
                    (employee_id,)
                )
                if cur.fetchone():
                    return jsonify({"error": "員工編號已存在"}), 409
            cur.execute(
                "INSERT INTO authorized_personnel "
                "(name, employee_id, role, face_folder, access_start, access_end) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (name, employee_id or None, role, face_folder, acc_start, acc_end)
            )
            conn.commit()
            new_id = cur.lastrowid

        files = request.files.getlist("face_images")
        print(f"📤 ADD PERSONNEL: received {len(files)} file(s) for folder '{face_folder}'")
        for fi in files:
            print(f"   → filename='{fi.filename}' content_type='{fi.content_type}'")
            if fi and fi.filename:
                save_face_image(face_folder, fi)

        global access_rules_cache_time
        access_rules_cache_time = 0.0
        return jsonify({"message": "新增成功", "id": new_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/personnel/<int:person_id>', methods=['PUT'])
@login_required
def update_personnel(person_id):
    name       = (request.form.get("name") or "").strip()
    role       = (request.form.get("role") or "").strip()
    acc_start  = (request.form.get("access_start") or "08:00").strip()
    acc_end    = (request.form.get("access_end") or "17:00").strip()

    if not name:
        return jsonify({"error": "請填寫姓名"}), 400

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT face_folder FROM authorized_personnel WHERE id = %s", (person_id,)
            )
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "找不到此人員"}), 404
            face_folder = row["face_folder"]
            cur.execute(
                "UPDATE authorized_personnel "
                "SET name=%s, role=%s, access_start=%s, access_end=%s WHERE id=%s",
                (name, role, acc_start, acc_end, person_id)
            )
            conn.commit()

        files = request.files.getlist("face_images")
        print(f"📤 UPDATE PERSONNEL: received {len(files)} file(s) for folder '{face_folder}'")
        for fi in files:
            print(f"   → filename='{fi.filename}' content_type='{fi.content_type}'")
            if fi and fi.filename:
                save_face_image(face_folder, fi)

        global access_rules_cache_time
        access_rules_cache_time = 0.0
        return jsonify({"message": "更新成功"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/personnel/<int:person_id>', methods=['DELETE'])
@login_required
def remove_personnel(person_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT face_folder FROM authorized_personnel WHERE id = %s", (person_id,)
            )
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "找不到此人員"}), 404
            face_folder = row["face_folder"]
            cur.execute("DELETE FROM authorized_personnel WHERE id = %s", (person_id,))
            conn.commit()

        delete_face_folder(face_folder)
        global access_rules_cache_time
        access_rules_cache_time = 0.0
        return jsonify({"message": "刪除成功"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/personnel/<int:person_id>/toggle', methods=['POST'])
@login_required
def toggle_personnel(person_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT is_active FROM authorized_personnel WHERE id = %s", (person_id,)
            )
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "找不到此人員"}), 404
            new_status = 0 if row["is_active"] else 1
            cur.execute(
                "UPDATE authorized_personnel SET is_active=%s WHERE id=%s",
                (new_status, person_id)
            )
            conn.commit()

        global access_rules_cache_time
        access_rules_cache_time = 0.0
        return jsonify({"message": "狀態更新成功", "is_active": bool(new_status)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/personnel/<int:person_id>/photo')
@login_required
def get_personnel_photo(person_id):
    """Return the first available photo (used as thumbnail)."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT face_folder FROM authorized_personnel WHERE id = %s", (person_id,)
            )
            row = cur.fetchone()
        if not row:
            return jsonify({"error": "Not found"}), 404
        photo_path = get_first_photo_path(row["face_folder"])
        if photo_path and os.path.exists(photo_path):
            return send_file(photo_path)
        return jsonify({"error": "No photo"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/personnel/<int:person_id>/photos')
@login_required
def get_personnel_photos(person_id):
    """Return list of all photo filenames for a person."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT face_folder FROM authorized_personnel WHERE id = %s", (person_id,)
            )
            row = cur.fetchone()
        if not row:
            return jsonify({"error": "Not found"}), 404
        photos = list_face_photos(row["face_folder"])
        return jsonify({"photos": photos})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/personnel/<int:person_id>/photo/<filename>')
@login_required
def get_personnel_photo_by_name(person_id, filename):
    """Serve a specific photo by filename."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT face_folder FROM authorized_personnel WHERE id = %s", (person_id,)
            )
            row = cur.fetchone()
        if not row:
            return jsonify({"error": "Not found"}), 404
        safe = os.path.basename(filename)
        photo_path = os.path.join("faces_db", row["face_folder"], safe)
        if os.path.exists(photo_path):
            return send_file(photo_path)
        return jsonify({"error": "Photo not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/personnel/<int:person_id>/photo/<filename>', methods=['DELETE'])
@login_required
def delete_personnel_photo(person_id, filename):
    """Delete one specific photo from a person's folder."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT face_folder FROM authorized_personnel WHERE id = %s", (person_id,)
            )
            row = cur.fetchone()
        if not row:
            return jsonify({"error": "Not found"}), 404
        safe = os.path.basename(filename)
        if delete_face_photo(row["face_folder"], safe):
            global access_rules_cache_time
            access_rules_cache_time = 0.0
            return jsonify({"message": "刪除成功"})
        return jsonify({"error": "檔案不存在"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


@app.route('/login')
def login_page():
    if 'user_id' in session:
        return redirect('/')
    return send_file('login.html')

@app.route('/register')
def register_page():
    if 'user_id' in session:
        return redirect('/')
    return send_file('register.html')

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json()
    username = (data.get('username') or '').strip()
    password = (data.get('password') or '').strip()
    if not username or not password:
        return jsonify({'error': '請填寫所有欄位'}), 400
    if len(username) < 3:
        return jsonify({'error': '帳號至少需要 3 個字元'}), 400
    if len(password) < 6:
        return jsonify({'error': '密碼至少需要 6 個字元'}), 400
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
            if cursor.fetchone():
                return jsonify({'error': '此帳號已被使用'}), 409
            pw_hash = generate_password_hash(password)
            cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", (username, pw_hash))
            connection.commit()
        print(f"✅ AUTH: 新用戶註冊 - {username}")
        return jsonify({'message': '註冊成功'})
    except Exception as e:
        print(f"❌ AUTH: 註冊失敗 - {e}")
        return jsonify({'error': '伺服器錯誤，請稍後再試'}), 500
    finally:
        connection.close()

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    username = (data.get('username') or '').strip()
    password = (data.get('password') or '').strip()
    if not username or not password:
        return jsonify({'error': '請填寫所有欄位'}), 400
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT id, password_hash FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()
        if not user or not check_password_hash(user['password_hash'], password):
            return jsonify({'error': '帳號或密碼錯誤'}), 401
        session['user_id']  = user['id']
        session['username'] = username
        print(f"✅ AUTH: 用戶登入 - {username}")
        return jsonify({'message': '登入成功'})
    except Exception as e:
        print(f"❌ AUTH: 登入失敗 - {e}")
        return jsonify({'error': '伺服器錯誤，請稍後再試'}), 500
    finally:
        connection.close()

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

@app.route('/tank/<int:tank_id>')
@login_required
def tank_page(tank_id):
    if tank_id not in VALID_TANKS:
        return jsonify({'error': 'Invalid tank ID'}), 400
    return send_file('tank.html')

@app.route('/api/sensor-history/tank/<int:tank_id>/<int:hours>')
@login_required
def get_tank_sensor_history(tank_id, hours):
    if tank_id not in VALID_TANKS:
        return jsonify({'error': 'Invalid tank ID'}), 400
    table = VALID_TANKS[tank_id]
    connection = get_tank_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"""
                SELECT * FROM `{table}`
                WHERE `Timestamp` >= DATE_SUB(NOW(), INTERVAL %s HOUR)
                ORDER BY `Timestamp` ASC
            """, (hours,))
            results = cursor.fetchall()
        normalized = []
        for r in results:
            nr = normalize_sensor_data(r)
            ts = nr.get('timestamp')
            if ts and hasattr(ts, 'strftime'):
                nr['timestamp'] = ts.strftime('%Y-%m-%dT%H:%M:%S')
            normalized.append(nr)
        return jsonify(normalized)
    except Exception as e:
        print(f"❌ GET DATA: Failed to retrieve {table} history - {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        connection.close()

@app.route('/history')
@login_required
def history_page():
    return send_file('history.html')

@app.route('/detection-image/<filename>')
@login_required
def serve_detection_image(filename):
    path = os.path.join("detection", "captures", filename)
    if os.path.exists(path):
        return send_file(path)
    return jsonify({'error': 'Image not found'}), 404

@app.route('/api/detection-history')
@login_required
def get_detection_history():
    page  = request.args.get('page',  1,  type=int)
    limit = request.args.get('limit', 12, type=int)
    offset = (page - 1) * limit
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT * FROM detection_logs
                ORDER BY timestamp DESC
                LIMIT %s OFFSET %s
            """, (limit, offset))
            results = cursor.fetchall()
            cursor.execute("SELECT COUNT(*) as total FROM detection_logs")
            total = cursor.fetchone()['total']
            cursor.execute("""
                SELECT COUNT(*) as today
                FROM detection_logs
                WHERE DATE(timestamp) = CURDATE()
            """)
            today = cursor.fetchone()['today']
        for r in results:
            if r.get('timestamp'):
                r['timestamp'] = r['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        return jsonify({'data': results, 'total': total, 'today': today, 'page': page})
    except Exception as e:
        print(f"❌ GET DATA: Failed to retrieve detection history - {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        connection.close()

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_yolo_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/<int:cam_id>')
@login_required
def video_feed_surv(cam_id):
    if cam_id not in SURV_IDS:
        return jsonify({'error': 'Invalid camera ID'}), 400
    return Response(generate_surv_stream(cam_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/cameras/status')
@login_required
def cameras_status():
    return jsonify({str(i): surv_online.get(i, False) for i in SURV_IDS})

@app.route('/')
@login_required
def index():
    print("🌐 Serving web dashboard...")
    return send_file('index.html')

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🐟 AQUAPONICS MONITORING SYSTEM")
    print("="*60)
    if init_system():
        print("\n🚀 Starting server...")
        print("📍 http://0.0.0.0:5000")
        print("✅ System ready! Press Ctrl+C to stop\n")
        try:
            from waitress import serve
            print("🚀 使用 Waitress production server (threads=32)")
            serve(app, host='0.0.0.0', port=5000, threads=32, channel_timeout=300)
        except ImportError:
            print("⚠️  Waitress 未安裝，使用 Flask dev server (pip install waitress)")
            app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    else:
        print("❌ Failed to initialize system")
        print("⚠️ Shutting down...")
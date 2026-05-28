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
model = None
lock = threading.Lock()
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
face_pending_boxes       = None
face_pending_lock        = threading.Lock()
face_recognition_ran_once = False   # True after first recognition cycle completes
access_rules_cache        = []
access_rules_cache_time   = 0.0
latest_face_list          = []   # list of {name, status, confidence, detected_at} for all current detections
latest_face_lock          = threading.Lock()

# ── Room Entry/Exit Detection ─────────────────────────────────
EXIT_TRIGGER_DELAY   = 5    # sec: wait after cam3 sees person before checking cam2
CAM2_PERSON_TIMEOUT  = 8    # sec: if cam2 hasn't seen person in N sec → left room
ENTRY_COOLDOWN_SEC   = 30   # sec: min gap between ENTER events per person
SURV_DETECT_INTERVAL = 0.5  # sec: how often surv detect loop runs (~2 fps)
SURV_FACE_INTERVAL   = 3.0  # sec: face recognition rate limit per surv camera

cam_identities     = {1: {}, 2: {}, 3: {}}  # {cam_id: {face_folder: last_seen_ts}}
cam_ident_lock     = threading.Lock()
room_occupants     = {}                       # {face_folder: {name, entered_at}}
room_lock          = threading.Lock()
pending_exit       = {}                       # {face_folder: threading.Timer}
pending_exit_lock  = threading.Lock()
entry_cooldown_ts  = {}                       # {face_folder: last_entry_time}
entry_cd_lock      = threading.Lock()

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

def surv_capture_loop(cam_id, phys_idx):
    """cam_id = logical display ID (1/2/3), phys_idx = physical camera index."""
    global surv_frames, surv_frame_counts, surv_online, surv_caps, running
    frame_interval = 1.0 / 5
    surv_idle_loops = 0
    while running:
        try:
            t0 = time.time()
            cap_obj = surv_caps.get(cam_id)
            if not cap_obj:
                surv_idle_loops += 1
                # Try to reinitialize camera every ~15 seconds when offline
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
                        surv_caps[cam_id]   = c
                        surv_online[cam_id] = True
                        print(f"🔄 攝影機 {cam_id} (index {cur_idx}): 重新初始化成功")
                    else:
                        c.release()
                time.sleep(0.5)
                continue
            success, img = cap_obj.read()
            if not success:
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
                    surv_caps[cam_id] = new_cap
                    surv_online[cam_id] = True
                    print(f"🔄 攝影機 {cam_id} (index {cur_idx}): 重新連接成功")
                else:
                    new_cap.release()
                continue
            ret, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 50])
            if ret:
                with surv_locks[cam_id]:
                    surv_frames[cam_id] = buf.tobytes()
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
    global face_pending_frame, face_pending_boxes, face_recognition_ran_once

    frame_interval = 1.0 / 10   # 10 fps target
    detect_every   = 3
    local_count    = 0
    last_boxes     = []
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
                    results    = model(img, classes=[0], verbose=False)
                    last_boxes = [b for b in results[0].boxes
                                  if float(b.conf[0]) >= MIN_CONFIDENCE]
                except Exception as e:
                    print(f"❌ AI DETECTION: {e}")
                    last_boxes = []

                with lock:
                    current_person_count = len(last_boxes)

                # Trigger face recognition every 5 YOLO detections (~1.5 s)
                if last_boxes and local_count % (detect_every * 5) == 0:
                    with face_pending_lock:
                        face_pending_frame = img.copy()
                        face_pending_boxes = list(last_boxes)

                if last_boxes:
                    now = time.time()
                    if now - last_telegram_sent >= COOLDOWN_SECONDS:
                        count = len(last_boxes)
                        daily_detection_total += count
                        last_detection_timestamp = time.strftime("%Y/%m/%d %H:%M:%S")

                        ts             = time.strftime("%Y%m%d_%H%M%S")
                        image_filename = f"detection_{ts}.jpg"
                        image_path     = os.path.join("detection", "captures", image_filename)
                        cv2.imwrite(image_path, img)

                        # Read latest face recognition results
                        with face_result_lock:
                            current_face = face_results.copy()

                        primary      = current_face.get(0, {"name": None, "status": "unknown"})
                        p_name       = primary.get("name")
                        p_status     = primary.get("status", "unknown")
                        all_authorized = bool(current_face) and all(
                            r.get("status") == "authorized" for r in current_face.values()
                        )

                        conn = get_db_connection()
                        try:
                            with conn.cursor() as cur:
                                cur.execute(
                                    "INSERT INTO detection_logs (person_count, image_filename, person_name, access_status) VALUES (%s, %s, %s, %s)",
                                    (count, image_filename, p_name, p_status)
                                )
                                conn.commit()
                        except Exception as e:
                            print(f"❌ DB: 記錄偵測失敗 - {e}")
                        finally:
                            conn.close()

                        if all_authorized:
                            names = ", ".join(r.get("name", "") for r in current_face.values())
                            print(f"✅ 授權人員進入: {names}")
                        elif not face_recognition_ran_once:
                            # Recognition hasn't completed its first cycle yet — skip alert
                            print(f"⏳ 偵測到 {count} 人，等待人臉識別初始化...")
                        else:
                            has_unauth = any(
                                r.get("status") == "unauthorized" for r in current_face.values()
                            )
                            if has_unauth:
                                bad_names = ", ".join(
                                    r.get("name") or "未知"
                                    for r in current_face.values()
                                    if r.get("status") == "unauthorized"
                                )
                                msg = (f"🚨 未授權人員入侵！\n👤 {bad_names}\n"
                                       f"👥 偵測到: {count} 人\n"
                                       f"🕐 {time.strftime('%Y/%m/%d %H:%M:%S')}\n"
                                       f"📍 智慧物聯實驗室")
                            else:
                                msg = (f"⚠️ 未知人員偵測！\n👥 偵測到: {count} 人\n"
                                       f"🕐 {time.strftime('%Y/%m/%d %H:%M:%S')}\n"
                                       f"📍 智慧物聯實驗室")
                            send_telegram_notification(msg, image_path)

                        last_telegram_sent = now
                    else:
                        print(f"✅ AI DETECTION: 偵測到 {len(last_boxes)} 人 (冷卻中)")

            # Draw bounding boxes with face recognition labels
            with face_result_lock:
                draw_face = face_results.copy()

            for i, box in enumerate(last_boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                info   = draw_face.get(i)
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


def face_recognition_loop():
    """Background thread: run DeepFace on frames queued by yolo_capture_loop."""
    global face_pending_frame, face_pending_boxes, face_results
    global access_rules_cache, access_rules_cache_time, running
    global latest_face_list

    while running:
        img   = None
        boxes = None

        with face_pending_lock:
            if face_pending_frame is not None and face_pending_boxes is not None:
                img   = face_pending_frame.copy()
                boxes = list(face_pending_boxes)
                face_pending_frame = None
                face_pending_boxes = None

        if img is None:
            time.sleep(0.05)
            continue

        # Refresh access rules from DB every 60 seconds
        now_ts = time.time()
        if now_ts - access_rules_cache_time > 60:
            try:
                conn = get_db_connection()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT name, face_folder, access_start, access_end, is_active "
                        "FROM authorized_personnel"
                    )
                    rows = cur.fetchall()
                conn.close()

                def _ts(val):
                    if val is None:
                        return "00:00"
                    if isinstance(val, str):
                        return str(val)[:5]
                    total = int(val.total_seconds())
                    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}"

                access_rules_cache = [
                    {
                        "name":         r["name"],
                        "face_folder":  r["face_folder"],
                        "access_start": _ts(r.get("access_start")),
                        "access_end":   _ts(r.get("access_end")),
                        "is_active":    bool(r.get("is_active", 1)),
                    }
                    for r in rows
                ]
                access_rules_cache_time = now_ts
                print(f"🔄 FACE: Access rules refreshed ({len(access_rules_cache)} personnel)")
            except Exception as e:
                print(f"❌ FACE: Failed to refresh access rules - {e}")

        new_results = {}
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Pass full bounding box — DeepFace's internal detector finds the face
            # Add a small top margin so the full head is captured
            margin   = max(10, int((y2 - y1) * 0.08))
            crop_y1  = max(0, y1 - margin)
            crop_y2  = min(img.shape[0], y2)
            crop_x1  = max(0, x1)
            crop_x2  = min(img.shape[1], x2)
            face_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
            if face_img.size == 0:
                new_results[i] = {"name": None, "status": "unknown", "confidence": 0.0}
                continue
            result = recognize_face(face_img, access_rules_cache)
            new_results[i] = result
            print(f"🔍 FACE [{i}]: {result}")

        with face_result_lock:
            face_results = new_results

        # Save all detected persons for dashboard display
        if new_results:
            now_str = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            with latest_face_lock:
                latest_face_list = [
                    {
                        "name":        r.get("name"),
                        "status":      r.get("status", "unknown"),
                        "confidence":  r.get("confidence", 0.0),
                        "detected_at": now_str,
                    }
                    for r in new_results.values()
                ]

        global face_recognition_ran_once
        face_recognition_ran_once = True


def handle_entry(name, face_folder):
    """Called when cam1 recognizes a person — logs ENTER if not in cooldown."""
    now = time.time()
    with entry_cd_lock:
        if now - entry_cooldown_ts.get(face_folder, 0) < ENTRY_COOLDOWN_SEC:
            return
        entry_cooldown_ts[face_folder] = now

    with room_lock:
        if face_folder in room_occupants:
            return  # already inside
        room_occupants[face_folder] = {
            "name": name,
            "entered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    threading.Thread(target=log_room_event,
                     args=("ENTER", name, face_folder, 1), daemon=True).start()
    print(f"🚪 ENTER: {name} ({face_folder})")


def handle_exit_trigger(name, face_folder):
    """Called when cam3 recognizes a person — starts exit confirmation timer."""
    with pending_exit_lock:
        if face_folder in pending_exit:
            return  # already counting down

    def confirm_exit():
        now = time.time()
        with cam_ident_lock:
            last_seen = cam_identities[2].get(face_folder, 0)
        if now - last_seen > CAM2_PERSON_TIMEOUT:
            with room_lock:
                room_occupants.pop(face_folder, None)
            threading.Thread(target=log_room_event,
                             args=("EXIT", name, face_folder, 3), daemon=True).start()
            print(f"🚪 EXIT confirmed: {name} ({face_folder})")
        else:
            print(f"🚪 EXIT cancelled: {name} still in room (cam2 saw them {now - last_seen:.1f}s ago)")
        with pending_exit_lock:
            pending_exit.pop(face_folder, None)

    t = threading.Timer(EXIT_TRIGGER_DELAY, confirm_exit)
    with pending_exit_lock:
        pending_exit[face_folder] = t
    t.start()
    print(f"🚪 EXIT pending: {name} ({face_folder}), confirming in {EXIT_TRIGGER_DELAY}s")


def surv_detect_loop(cam_id):
    """
    Low-fps person detection + face recognition on a surveillance camera.
    cam1 → triggers ENTER   cam3 → triggers EXIT pending   cam2 → updates identity timestamps
    """
    last_face_time = 0

    while running:
        try:
            with surv_locks[cam_id]:
                frame_bytes = surv_frames.get(cam_id)

            if frame_bytes is None or model is None or not surv_online.get(cam_id):
                time.sleep(SURV_DETECT_INTERVAL)
                continue

            nparr = np.frombuffer(frame_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                time.sleep(SURV_DETECT_INTERVAL)
                continue

            # Person detection (fast)
            results = model(img, classes=[0], verbose=False)
            boxes = [b for b in results[0].boxes if float(b.conf[0]) >= MIN_CONFIDENCE]

            if not boxes:
                time.sleep(SURV_DETECT_INTERVAL)
                continue

            # Rate-limit face recognition
            now = time.time()
            if now - last_face_time < SURV_FACE_INTERVAL:
                time.sleep(SURV_DETECT_INTERVAL)
                continue
            last_face_time = now

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                margin   = max(10, int((y2 - y1) * 0.08))
                face_img = img[max(0, y1 - margin):min(img.shape[0], y2),
                               max(0, x1):min(img.shape[1], x2)]
                if face_img.size == 0:
                    continue

                result = recognize_face(face_img, access_rules_cache)
                name   = result.get("name")
                status = result.get("status", "unknown")

                if not name or status == "unknown":
                    continue

                # face_folder lookup from access_rules_cache
                face_folder = next(
                    (r["face_folder"] for r in access_rules_cache if r["name"] == name),
                    name,
                )

                # Update presence timestamp for cam2 confirmation
                with cam_ident_lock:
                    cam_identities[cam_id][face_folder] = time.time()

                print(f"👁️  CAM{cam_id} sees {name} ({status})")

                if cam_id == 1:
                    handle_entry(name, face_folder)
                elif cam_id == 3:
                    handle_exit_trigger(name, face_folder)

        except Exception as e:
            print(f"❌ SURV DETECT cam{cam_id}: {e}")

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
    with room_lock:
        occupants = [
            {"face_folder": k, "name": v["name"], "entered_at": v["entered_at"]}
            for k, v in room_occupants.items()
        ]
    return jsonify({"count": len(occupants), "occupants": occupants})


@app.route('/api/room/events')
@login_required
def get_room_events():
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
        conn.close()
        return jsonify({"events": events})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/room/history')
@login_required
def get_room_history():
    try:
        page     = max(1, int(request.args.get('page', 1)))
        per_page = 20
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
                    TIMESTAMPDIFF(MINUTE, e.timestamp, x.timestamp)              AS duration_min
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
        conn.close()

        return jsonify({
            'records':     rows,
            'total':       total,
            'page':        page,
            'per_page':    per_page,
            'total_pages': max(1, (total + per_page - 1) // per_page),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/face/latest')
@login_required
def get_latest_face():
    import glob as _glob
    with latest_face_lock:
        people = list(latest_face_list)

    def _ts(v):
        if v is None: return "-"
        if isinstance(v, str): return str(v)[:5]
        total = int(v.total_seconds())
        return f"{total//3600:02d}:{(total%3600)//60:02d}"

    enriched = []
    for p in people:
        item = p.copy()
        if p.get("name"):
            try:
                conn = get_db_connection()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT employee_id, role, access_start, access_end "
                        "FROM authorized_personnel WHERE name=%s",
                        (p["name"],)
                    )
                    row = cur.fetchone()
                conn.close()
                if row:
                    item["employee_id"]  = row.get("employee_id") or "-"
                    item["role"]         = row.get("role") or "-"
                    item["access_start"] = _ts(row.get("access_start"))
                    item["access_end"]   = _ts(row.get("access_end"))
            except Exception:
                pass
        enriched.append(item)

    captures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection", "captures")
    captures = sorted(_glob.glob(os.path.join(captures_dir, "*.jpg")), key=os.path.getmtime, reverse=True)
    return jsonify({
        "people":      enriched,
        "capture_url": "/api/face/latest/capture" if captures else None,
    })

@app.route('/api/face/latest/capture')
@login_required
def get_latest_capture():
    import glob as _glob
    captures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection", "captures")
    captures = sorted(_glob.glob(os.path.join(captures_dir, "*.jpg")), key=os.path.getmtime, reverse=True)
    if captures:
        return send_file(captures[0])
    return jsonify({"error": "No captures"}), 404

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
    global model
    print("🤖 加載 YOLO 模型...")
    try:
        model = YOLO("yolov8n.pt")
        print("✅ AI DETECTION: YOLO 模型加載成功")
        init_detection_logs()
        init_users_table()
        init_authorized_personnel_table()
        init_room_tables()
        init_yolo_camera()
        init_surveillance_cameras()
        t = threading.Thread(target=face_recognition_loop, daemon=True)
        t.start()
        print("✅ FACE: Recognition thread started")
        for cid in [1, 2, 3]:
            t2 = threading.Thread(target=surv_detect_loop, args=(cid,), daemon=True)
            t2.start()
            print(f"✅ ROOM DETECT: cam{cid} thread started")
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

        def _ts(val):
            if val is None: return "00:00"
            if isinstance(val, str): return str(val)[:5]
            total = int(val.total_seconds())
            return f"{total // 3600:02d}:{(total % 3600) // 60:02d}"

        result = []
        for r in rows:
            folder = r.get("face_folder", "")
            r["access_start"] = _ts(r.get("access_start"))
            r["access_end"]   = _ts(r.get("access_end"))
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
import cv2
import requests
import time
import os
import threading
from datetime import datetime
from ultralytics import YOLO
from config import (TELEGRAM_TOKEN, CHAT_ID, COOLDOWN_SECONDS,
                    MIN_CONFIDENCE, ACTIVE_START, ACTIVE_END,
                    MIN_PERSON_COUNT, ENABLE_LOG, LOG_FILE)

# ── Setup ──────────────────────────────────────────
os.makedirs("captures", exist_ok=True)
model = YOLO("yolov8n.pt")
last_sent = 0

# ── ตัวแปรควบคุมระบบ ──────────────────────────────
notify_enabled = True
running        = True
current_frame  = None

# ── ฟังก์ชันเช็คเวลาทำงาน ──────────────────────────
def is_active_time():
    now = datetime.now().strftime("%H:%M")
    return ACTIVE_START <= now <= ACTIVE_END

# ── ฟังก์ชันบันทึก Log ────────────────────────────
def write_log(message):
    if ENABLE_LOG:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | {message}\n")
        print(f"[LOG] {message}")

# ── ฟังก์ชันส่ง Telegram ──────────────────────────
def send_telegram(message, image_path=None):
    try:
        if image_path:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
            with open(image_path, "rb") as img:
                requests.post(url,
                    data={"chat_id": CHAT_ID, "caption": message},
                    files={"photo": img}, timeout=10)
        else:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            requests.post(url,
                data={"chat_id": CHAT_ID, "text": message}, timeout=10)
        print(f"[Telegram] sent ok")
        write_log(f"sent: {message[:50]}")
    except Exception as e:
        print(f"[Telegram] error: {e}")
        write_log(f"send failed: {e}")

# ── ล้าง update เก่าทิ้งก่อนเริ่ม ────────────────
def clear_old_updates():
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
        res = requests.get(url, timeout=10)
        updates = res.json().get("result", [])
        if updates:
            last_id = updates[-1]["update_id"] + 1
            requests.get(url, params={"offset": last_id}, timeout=10)
            print(f"[Bot] cleared {len(updates)} old updates")
    except Exception as e:
        print(f"[Bot] clear error: {e}")

# ── ฟังก์ชันรับคำสั่งจาก Telegram ────────────────
def listen_commands():
    global notify_enabled, running
    last_update_id = None

    print("[Bot] listening for commands...")

    while running:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
            params = {"timeout": 5, "offset": last_update_id}
            res = requests.get(url, params=params, timeout=10)
            updates = res.json().get("result", [])

            for update in updates:
                last_update_id = update["update_id"] + 1
                msg     = update.get("message", {})
                chat_id = str(msg.get("chat", {}).get("id", ""))
                text    = msg.get("text", "").strip().lower()

                if not text:
                    continue

                if chat_id != str(CHAT_ID):
                    print(f"[Bot] ignored chat_id: {chat_id}")
                    continue

                print(f"[Bot] command: {text}")

                if text == "/start":
                    notify_enabled = True
                    send_telegram("✅ เปิดการแจ้งเตือนแล้ว!")
                    write_log("/start — notify on")

                elif text == "/stop":
                    send_telegram("🛑 ปิดโปรแกรมแล้ว! กดรัน mode1.py ใหม่เพื่อเริ่มใหม่")
                    write_log("/stop — shutdown")
                    running = False
                    break

                elif text == "/status":
                    active        = is_active_time()
                    status_notify = "✅ เปิดอยู่" if notify_enabled else "🔕 ปิดอยู่"
                    status_time   = "🟢 ในเวลาทำงาน" if active else "🔴 นอกเวลาทำงาน"
                    msg_out = (f"📊 สถานะระบบ\n"
                               f"🔔 การแจ้งเตือน: {status_notify}\n"
                               f"⏰ เวลาทำงาน: {ACTIVE_START} - {ACTIVE_END} น.\n"
                               f"🕐 เวลาปัจจุบัน: {datetime.now().strftime('%H:%M')}\n"
                               f"📅 สถานะเวลา: {status_time}")
                    send_telegram(msg_out)
                    write_log("/status")

                elif text == "/photo":
                    if current_frame is not None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        img_path  = f"captures/manual_{timestamp}.jpg"
                        cv2.imwrite(img_path, current_frame)
                        send_telegram(
                            f"📸 ภาพจากกล้องตอนนี้\n🕐 {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
                            img_path)
                        write_log("/photo — sent")
                    else:
                        send_telegram("❌ ยังไม่มีภาพจากกล้อง")

                else:
                    send_telegram(
                        "❓ คำสั่งที่ใช้ได้:\n"
                        "/status — ดูสถานะ\n"
                        "/stop — ปิดโปรแกรม\n"
                        "/start — เปิดแจ้งเตือน\n"
                        "/photo — ถ่ายรูปส่งมาเลย")

        except Exception as e:
            print(f"[Bot] error: {e}")

        time.sleep(1)

# ── ล้าง update เก่า + เริ่ม Thread ──────────────
clear_old_updates()
bot_thread = threading.Thread(target=listen_commands, daemon=True)
bot_thread.start()

# ── ข้อมูลส่งต่อให้ระบบอื่น ──────────────────────
detection_data = {"name": None, "time": None, "image_path": None}

# ── Main Loop ─────────────────────────────────────
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("cannot open camera!")
    exit()

print("[ Mode 1 ] running — press Q to quit")
print(f"work hours: {ACTIVE_START} - {ACTIVE_END}")
send_telegram(
    f"🟢 ระบบเริ่มทำงานแล้ว!\n"
    f"⏰ เวลาทำงาน: {ACTIVE_START} - {ACTIVE_END} น.\n\n"
    f"คำสั่งที่ใช้ได้:\n"
    f"/status — ดูสถานะ\n"
    f"/stop — ปิดโปรแกรม\n"
    f"/start — เปิดแจ้งเตือน\n"
    f"/photo — ถ่ายรูปส่งมาเลย")
write_log("=== system started ===")

while running:
    ret, frame = cap.read()

    if not ret:
        print("[camera] disconnected, restarting...")
        write_log("camera disconnected")
        cap.release()
        time.sleep(2)
        cap = cv2.VideoCapture(0)
        continue

    current_frame = frame.copy()

    results    = model(frame, classes=[0], verbose=False)
    detections = results[0].boxes
    valid      = [b for b in detections if float(b.conf[0]) >= MIN_CONFIDENCE]

    now    = time.time()
    active = is_active_time()

    for box in valid:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf  = float(box.conf[0])
        color = (0, 255, 0) if active else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Person {conf:.0%}",
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    status_color = (0, 255, 0) if active else (0, 0, 255)
    status_text  = f"ACTIVE {ACTIVE_START}-{ACTIVE_END}" if active else "INACTIVE"
    notify_color = (0, 255, 0) if notify_enabled else (0, 0, 255)
    notify_text  = "NOTIFY: ON" if notify_enabled else "NOTIFY: OFF"

    cv2.putText(frame, f"Status: {status_text}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    cv2.putText(frame, notify_text,
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, notify_color, 2)
    cv2.putText(frame, f"Person: {len(valid)}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    enough_people = len(valid) >= MIN_PERSON_COUNT
    cooldown_ok   = (now - last_sent) > COOLDOWN_SECONDS

    if enough_people and cooldown_ok and notify_enabled:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path  = f"captures/person_{timestamp}.jpg"
        cv2.imwrite(img_path, frame)

        detection_data = {
            "name": "Unknown Person",
            "time": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "image_path": img_path
        }

        if active:
            msg = (f"🚨 ตรวจพบคน!\n"
                   f"🕐 {detection_data['time']}\n"
                   f"👤 จำนวน: {len(valid)} คน\n"
                   f"🎯 ความแม่น: {max(float(b.conf[0]) for b in valid):.0%}")
            write_log(f"detected {len(valid)} (in hours) | {img_path}")
        else:
            msg = (f"⚠️ พบคนนอกเวลาทำงาน!\n"
                   f"🕐 {detection_data['time']}\n"
                   f"👤 จำนวน: {len(valid)} คน\n"
                   f"🎯 ความแม่น: {max(float(b.conf[0]) for b in valid):.0%}\n"
                   f"⏰ เวลาทำงาน: {ACTIVE_START} - {ACTIVE_END} น.")
            write_log(f"detected {len(valid)} (out of hours!) | {img_path}")

        send_telegram(msg, img_path)
        last_sent = now

    cv2.imshow("Mode 1 - Person Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

write_log("=== system stopped ===")
send_telegram("🔴 ระบบหยุดทำงานแล้ว!")
cap.release()
cv2.destroyAllWindows()
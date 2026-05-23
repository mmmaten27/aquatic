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
from detection import MIN_CONFIDENCE
from detection.config import TELEGRAM_TOKEN, CHAT_ID, COOLDOWN_SECONDS

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

# ── YOLO camera (physical index 3) → 即時 - YOLO AI section ──
YOLO_CAM_IDX    = 2
yolo_cap         = None
yolo_frame       = None
yolo_frame_count = 0
yolo_lock        = threading.Lock()
yolo_online      = False

# ── Surveillance cameras: logical ID → physical index ──────────
#    攝影機 1 = cam 3 │ 攝影機 2 = cam 0 │ 攝影機 3 = cam 1
SURV_IDS     = [1, 2, 3]
SURV_CAM_IDX = {1: 3, 2: 0, 3: 1}
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
    surv_caps.clear()

def surv_capture_loop(cam_id, phys_idx):
    """cam_id = logical display ID (1/2/3), phys_idx = physical camera index."""
    global surv_frames, surv_frame_counts, surv_online, surv_caps, running
    frame_interval = 1.0 / 15
    while running:
        try:
            t0 = time.time()
            cap_obj = surv_caps.get(cam_id)
            if not cap_obj:
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
                new_cap = cv2.VideoCapture(phys_idx, cv2.CAP_DSHOW)
                if new_cap.isOpened():
                    new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    surv_caps[cam_id] = new_cap
                    surv_online[cam_id] = True
                    print(f"🔄 攝影機 {cam_id} (index {phys_idx}): 重新連接成功")
                else:
                    new_cap.release()
                continue
            ret, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 65])
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

    frame_interval = 1.0 / 20   # 20 fps target
    detect_every   = 3
    local_count    = 0
    last_boxes     = []

    while running:
        try:
            t0 = time.time()

            if yolo_cap is None:
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
                new_cap = cv2.VideoCapture(YOLO_CAM_IDX, cv2.CAP_DSHOW)
                if new_cap.isOpened():
                    new_cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                    new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    new_cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
                    yolo_cap    = new_cap
                    yolo_online = True
                    print(f"🔄 YOLO CAM (index {YOLO_CAM_IDX}): 重新連接成功")
                else:
                    new_cap.release()
                continue

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

                        try:
                            conn = get_db_connection()
                            with conn.cursor() as cur:
                                cur.execute(
                                    "INSERT INTO detection_logs (person_count, image_filename) VALUES (%s, %s)",
                                    (count, image_filename)
                                )
                                conn.commit()
                            conn.close()
                        except Exception as e:
                            print(f"❌ DB: 記錄偵測失敗 - {e}")

                        msg = (f"🚨 偵測到人員！\n👥 偵測到: {count} 人\n"
                               f"🕐 {time.strftime('%Y/%m/%d %H:%M:%S')}\n📍 魚菜共生監控系統")
                        send_telegram_notification(msg, image_path)
                        last_telegram_sent = now
                    else:
                        print(f"✅ AI DETECTION: 偵測到 {len(last_boxes)} 人 (冷卻中)")

            for box in last_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"Person {float(box.conf[0]):.0%}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            ret, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
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


def generate_yolo_stream():
    """Yield cached YOLO frames — one generator per HTTP client."""
    last_count = -1
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
        c.set(cv2.CAP_PROP_FPS,          20)
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
                c.set(cv2.CAP_PROP_FPS,          15)
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
    try:
        connection = get_db_connection()
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
        connection.close()
        print("✅ DB: users table ready")
    except Exception as e:
        print(f"❌ DB: Failed to init users table - {e}")

def init_detection_logs():
    os.makedirs(os.path.join("detection", "captures"), exist_ok=True)
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detection_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    person_count INT DEFAULT 1,
                    image_filename VARCHAR(255)
                )
            """)
            connection.commit()
        connection.close()
        print("✅ DB: detection_logs table ready")
    except Exception as e:
        print(f"❌ DB: Failed to init detection_logs - {e}")


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
            # Remove this thread from pool when done
            if threading.current_thread() in telegram_thread_pool:
                telegram_thread_pool.remove(threading.current_thread())

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
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM sensordata ORDER BY timestamp DESC LIMIT 1")
            result = cursor.fetchone()
        connection.close()
        print("✅ GET DATA: Sensor data retrieved successfully")
        return jsonify(normalize_sensor_data(result) or {})
    except Exception as e:
        print(f"❌ GET DATA: Failed to retrieve sensor data - {e}")
        return jsonify({'error': str(e)}), 500

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

@app.route('/api/sensor-data/tank/<int:tank_id>')
@login_required
def get_tank_sensor_data(tank_id):
    if tank_id not in VALID_TANKS:
        return jsonify({'error': 'Invalid tank ID'}), 400
    table = VALID_TANKS[tank_id]
    try:
        connection = get_tank_db_connection()
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT * FROM `{table}` ORDER BY timestamp DESC LIMIT 1")
            result = cursor.fetchone()
        connection.close()
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

@app.route('/api/sensor-history/<int:hours>')
def get_sensor_history(hours):
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT * FROM sensordata
                WHERE timestamp >= DATE_SUB(NOW(), INTERVAL %s HOUR)
                ORDER BY timestamp ASC
            """, (hours,))
            results = cursor.fetchall()
        connection.close()
        normalized_results = [normalize_sensor_data(row) for row in results]
        print(f"✅ GET DATA: Retrieved {len(normalized_results)} records for last {hours} hours")
        return jsonify(normalized_results)
    except Exception as e:
        print(f"❌ GET DATA: Failed to retrieve history data - {e}")
        return jsonify({'error': str(e)}), 500


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
        init_yolo_camera()
        init_surveillance_cameras()
        return True
    except Exception as e:
        print(f"❌ SYSTEM: 初始化失敗 - {e}")
        return False

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
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
            if cursor.fetchone():
                connection.close()
                return jsonify({'error': '此帳號已被使用'}), 409
            pw_hash = generate_password_hash(password)
            cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", (username, pw_hash))
            connection.commit()
        connection.close()
        print(f"✅ AUTH: 新用戶註冊 - {username}")
        return jsonify({'message': '註冊成功'})
    except Exception as e:
        print(f"❌ AUTH: 註冊失敗 - {e}")
        return jsonify({'error': '伺服器錯誤，請稍後再試'}), 500

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    username = (data.get('username') or '').strip()
    password = (data.get('password') or '').strip()
    if not username or not password:
        return jsonify({'error': '請填寫所有欄位'}), 400
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT id, password_hash FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()
        connection.close()
        if not user or not check_password_hash(user['password_hash'], password):
            return jsonify({'error': '帳號或密碼錯誤'}), 401
        session['user_id']  = user['id']
        session['username'] = username
        print(f"✅ AUTH: 用戶登入 - {username}")
        return jsonify({'message': '登入成功'})
    except Exception as e:
        print(f"❌ AUTH: 登入失敗 - {e}")
        return jsonify({'error': '伺服器錯誤，請稍後再試'}), 500

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
    try:
        connection = get_tank_db_connection()
        with connection.cursor() as cursor:
            cursor.execute(f"""
                SELECT * FROM `{table}`
                WHERE `Timestamp` >= DATE_SUB(NOW(), INTERVAL %s HOUR)
                ORDER BY `Timestamp` ASC
            """, (hours,))
            results = cursor.fetchall()
        connection.close()
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
    try:
        connection = get_db_connection()
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
        connection.close()
        for r in results:
            if r.get('timestamp'):
                r['timestamp'] = r['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        return jsonify({'data': results, 'total': total, 'today': today, 'page': page})
    except Exception as e:
        print(f"❌ GET DATA: Failed to retrieve detection history - {e}")
        return jsonify({'error': str(e)}), 500

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
            print("🚀 使用 Waitress production server (threads=16)")
            serve(app, host='0.0.0.0', port=5000, threads=16, channel_timeout=300)
        except ImportError:
            print("⚠️  Waitress 未安裝，使用 Flask dev server (pip install waitress)")
            app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    else:
        print("❌ Failed to initialize system")
        print("⚠️ Shutting down...")
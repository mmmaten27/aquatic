"""
ทดสอบส่ง Telegram แจ้งเตือนทุกรูปแบบที่ระบบใช้งาน
ข้อมูลเป็นการสมมติทั้งหมด ไม่กระทบระบบจริง
"""

import os
import time
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")

if not TOKEN or not CHAT_ID:
    print("❌ ไม่พบ TELEGRAM_TOKEN หรือ CHAT_ID ใน .env")
    exit(1)


def send_text(msg: str) -> bool:
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    r = requests.post(url, data={"chat_id": CHAT_ID, "text": msg}, timeout=10)
    return r.status_code == 200


def send_photo(img_path: str, caption: str) -> bool:
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    with open(img_path, "rb") as f:
        r = requests.post(url,
                          data={"chat_id": CHAT_ID, "caption": caption},
                          files={"photo": f}, timeout=15)
    return r.status_code == 200


def ts() -> str:
    return datetime.now().strftime("%Y/%m/%d %H:%M:%S")


def create_dummy_image(path: str):
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (320, 240), color=(50, 50, 50))
        draw = ImageDraw.Draw(img)
        draw.rectangle([80, 40, 240, 180], outline="red", width=3)
        draw.text((100, 200), "UNKNOWN PERSON", fill="red")
        img.save(path)
        return True
    except ImportError:
        # Pillow ไม่ได้ติดตั้ง → สร้าง JPEG ขนาดเล็กด้วย bytes ตรงๆ
        jpeg_bytes = (
            b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
            b'\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t'
            b'\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a'
            b'\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\x1e'
            b'\xff\xc0\x00\x0b\x08\x00\x10\x00\x10\x01\x01\x11\x00'
            b'\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00'
            b'\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b'
            b'\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xff\xd9'
        )
        with open(path, "wb") as f:
            f.write(jpeg_bytes)
        return True


def run_test(label: str, fn):
    ok = fn()
    status = "✅" if ok else "❌"
    print(f"  {status}  {label}")
    time.sleep(1.5)  # หน่วงเล็กน้อยไม่ให้ Telegram rate-limit


print("=" * 55)
print("  ทดสอบการแจ้งเตือน Telegram ทุกรูปแบบ")
print("=" * 55)

# ─── 1. Access Control Alerts ──────────────────────────────────

print("\n[1/3] 人員管制通知 (Access Control Alerts)")

# 1-A : UNKNOWN detected (พร้อมรูป)
dummy_img = "test_unknown_capture.jpg"
create_dummy_image(dummy_img)
msg_unknown = (
    f"⚠️ 發現不明人士！\n"
    f"📷 攝影機: 入口\n"
    f"🕐 {ts()}\n"
    f"📍 IoT Lab"
)
run_test("1-A  UNKNOWN 發現不明人士 (พร้อมรูป)",
         lambda: send_photo(dummy_img, msg_unknown))

# 1-B : UNKNOWN left
msg_unknown_left = (
    f"✅ 不明人士已離開\n"
    f"📷 攝影機: 入口\n"
    f"🕐 {ts()}\n"
    f"📍 IoT Lab"
)
run_test("1-B  UNKNOWN 不明人士已離開",
         lambda: send_text(msg_unknown_left))

# 1-C : Authorized enter — outside working hours
msg_after_hours = (
    f"⚠️ Tongen 進入實驗室 (非上班時間)\n"
    f"🕐 {ts()}\n"
    f"📍 IoT Lab"
)
run_test("1-C  授權人員 非上班時間 進入",
         lambda: send_text(msg_after_hours))

# 1-D : Unauthorized entry (deactivated account / wrong time slot)
msg_unauth = (
    f"🚨 jiabao 進入實驗室 (未授權進入)\n"
    f"🕐 {ts()}\n"
    f"📍 IoT Lab"
)
run_test("1-D  未授權進入",
         lambda: send_text(msg_unauth))

# 1-E : Person exits with duration
msg_exit = (
    f"🚶 TT 離開實驗室\n"
    f"🕐 {ts()}\n"
    f"⏱ 停留 42 分鐘\n"
    f"📍 IoT Lab"
)
run_test("1-E  人員離開 (含停留時間)",
         lambda: send_text(msg_exit))

# ─── 2. Sensor Alerts ──────────────────────────────────────────

print("\n[2/3] 感測器監測通知 (Sensor Monitoring Alerts)")

# 2-A : Sustained alert level 1 (>30 min)
msg_alert1 = (
    f"🟡 魚缸 1 - pH = 8.2\n"
    f"📌 數值超標 > 30 分鐘\n"
    f"🕐 {ts()}\n"
    f"📍 魚菜共生監控系統"
)
run_test("2-A  數值超標 > 30 分鐘 (Level 1)",
         lambda: send_text(msg_alert1))

# 2-B : Sustained alert level 2 (>2 hr)
msg_alert2 = (
    f"🔴 魚缸 2 - 溫度 = 29.5\n"
    f"📌 數值超標 > 2 小時\n"
    f"🕐 {ts()}\n"
    f"📍 魚菜共生監控系統"
)
run_test("2-B  數值超標 > 2 小時 (Level 2)",
         lambda: send_text(msg_alert2))

# 2-C : Sensor fault (value stuck at 0)
msg_fault = (
    f"🔌 感測器異常！\n"
    f"🐟 魚缸 3 - DO 持續回傳 0\n"
    f"⚠️ 感測器可能已斷線或損壞，請立即檢查\n"
    f"🕐 {ts()}\n"
    f"📍 魚菜共生監控系統"
)
run_test("2-C  感測器故障 (持續回傳 0)",
         lambda: send_text(msg_fault))

# 2-D : Recovery
msg_recovery = (
    f"✅ 魚缸 1 - pH 已恢復正常 (7.1)\n"
    f"🕐 {ts()}\n"
    f"📍 魚菜共生監控系統"
)
run_test("2-D  數值已恢復正常",
         lambda: send_text(msg_recovery))

# 2-E : Trend alert (กำลังจะเกินเกณฑ์)
msg_trend = (
    f"⚠️ 趨勢預警！\n"
    f"📉 pH 持續下降中\n"
    f"🐟 魚缸 1 - pH：6.8\n"
    f"⏱ 預計約 300 秒後超出標準範圍\n"
    f"🕐 {ts()}\n"
    f"📍 魚菜共生監控系統"
)
run_test("2-E  趨勢預警 (กำลังจะเกินเกณฑ์)",
         lambda: send_text(msg_trend))

# ─── 3. ล้างไฟล์ทดสอบ ──────────────────────────────────────────

print("\n[3/3] ล้างไฟล์ทดสอบ")
if os.path.exists(dummy_img):
    os.remove(dummy_img)
    print(f"  ✅  ลบ {dummy_img} แล้ว")

print("\n" + "=" * 55)
print("  เสร็จสิ้น — ตรวจสอบ Telegram ได้เลยครับ")
print("=" * 55)

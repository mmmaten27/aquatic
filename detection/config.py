import os
from dotenv import load_dotenv
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID        = os.getenv("CHAT_ID", "")

# ── Cooldown ──
COOLDOWN_SECONDS = 10

# ── Confidence Threshold ──
MIN_CONFIDENCE = 0.70

# ── เวลาทำงาน 08:00 - 17:00 ──
ACTIVE_START = "08:00"
ACTIVE_END   = "17:00"

# ── จำนวนคน ──
MIN_PERSON_COUNT = 1

# ── Log ──
ENABLE_LOG = True
LOG_FILE = "detection_log.txt"
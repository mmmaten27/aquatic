import logging
import os
import sys
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, "detailed_log.txt")

# ── Debug logger (existing) ───────────────────────────────────────
detailed_logger = logging.getLogger("aquaponics_detailed")
detailed_logger.setLevel(logging.DEBUG)

if not detailed_logger.handlers:
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8", mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    detailed_logger.addHandler(fh)

# ── Event logger — key events only (ENTER/EXIT/camera/face/error) ─
EVENT_LOG = os.path.join(os.path.dirname(LOG_DIR), "system_events.log")
event_logger = logging.getLogger("aquaponics_events")
event_logger.setLevel(logging.INFO)

if not event_logger.handlers:
    efh = RotatingFileHandler(
        EVENT_LOG, encoding="utf-8", mode="a",
        maxBytes=2 * 1024 * 1024,   # 2 MB max
        backupCount=3,               # keep 3 old files
    )
    efh.setFormatter(logging.Formatter(
        "%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    event_logger.addHandler(efh)


VERBOSE = False


def dprint(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    detailed_logger.debug(msg)
    if VERBOSE:
        print(*args, **kwargs, file=sys.stderr)


def iprint(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    detailed_logger.info(msg)
    print(*args, **kwargs)


def elog(msg: str):
    """Write a key event to system_events.log and stdout."""
    event_logger.info(msg)
    print(msg)

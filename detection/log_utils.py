import logging
import os
import sys

LOG_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, "detailed_log.txt")

detailed_logger = logging.getLogger("aquaponics_detailed")
detailed_logger.setLevel(logging.DEBUG)

if not detailed_logger.handlers:
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    detailed_logger.addHandler(fh)

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

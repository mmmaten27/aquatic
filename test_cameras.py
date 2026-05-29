"""
Quick camera diagnostic.
Run: python test_cameras.py
Shows each camera index, whether it opens, and whether it returns a real (non-black) frame.
"""
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from detection.camera_utils import get_cameras_with_device_path, load_calibration

MAX_INDEX = 8

def test_camera(idx, label=""):
    tag = f"[{idx}] {label}".strip()
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"  {tag} → CANNOT OPEN")
        cap.release()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    # Flush stale DirectShow buffer frames
    for _ in range(5):
        cap.read()

    ok, frame = cap.read()
    if not ok or frame is None:
        print(f"  {tag} → OPENED but read() FAILED")
    else:
        mean = float(np.mean(frame))
        status = "OK" if mean > 3.0 else "BLACK FRAME (mean={:.2f})".format(mean)
        h, w = frame.shape[:2]
        print(f"  {tag} → {status}  ({w}x{h})")
    cap.release()


print("=" * 60)
print("  Camera Diagnostic Tool")
print("=" * 60)

# Show OS-level camera list
print("\n[DirectShow] Detected cameras:")
try:
    devices = get_cameras_with_device_path()
    for d in devices:
        print(f"  index {d['index']}: {d['name']}")
        if d['device_path']:
            print(f"            path: {d['device_path'][:80]}")
except Exception as e:
    print(f"  (enumeration failed: {e})")
    devices = []

# Show calibration mapping
print("\n[Calibration] Saved device-path → role mapping:")
cal = load_calibration()
if cal:
    for role, path in cal.items():
        matched = [d for d in devices if d['device_path'] == path]
        idx_str = str(matched[0]['index']) if matched else "NOT FOUND"
        print(f"  {role}: index {idx_str}  path={path[:60]}")
else:
    print("  (no calibration file found)")

# Test every index
print("\n[Frame Test] Testing indices 0 to {}:".format(MAX_INDEX - 1))
for i in range(MAX_INDEX):
    label = next((d['name'] for d in devices if d['index'] == i), "")
    test_camera(i, label)

print("\n" + "=" * 60)
print("Done. Compare 'index' values above with your SURV_CAM_IDX mapping.")
print("=" * 60)

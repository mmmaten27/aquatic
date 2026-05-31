"""
Face capture & recognition diagnostic tool.
Run: python test_face_capture.py [camera_index]
If no index given, shows available cameras and lets you pick.

Opens a live window showing YOLO detections + face recognition results.
Press 's' to save debug frame, 'q' to quit.
"""
import os, sys, glob, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
os.environ["YOLO_VERBOSE"] = "False"

from ultralytics import YOLO
from insightface.app import FaceAnalysis
from detection.face_utils import (
    _normalize_lighting, recognize_face, _ensure_db, CAM_PROFILE
)
from detection.camera_utils import get_cameras_with_device_path
from detection.config import MIN_CONFIDENCE

DEBUG_DIR = os.path.join(os.path.dirname(__file__), "detection", "debug")
os.makedirs(DEBUG_DIR, exist_ok=True)

# ── Overridable per-camera profile (tune here without touching main code) ──
TEST_PROFILE = {
    0: {"min_size": 24, "sim_thresh": 0.55, "quality": 0.40},
    1: {"min_size": 22, "sim_thresh": 0.55, "quality": 0.38},
    2: {"min_size": 20, "sim_thresh": 0.55, "quality": 0.38},
    3: {"min_size": 20, "sim_thresh": 0.60, "quality": 0.35},
}

def apply_test_profile():
    import detection.face_utils as fu
    for k, v in TEST_PROFILE.items():
        if k in fu.CAM_PROFILE:
            fu.CAM_PROFILE[k].update(v)

apply_test_profile()

print("Loading YOLO model...")
yolo = YOLO("yolov8n.pt")
print("YOLO ready.")

print("Loading InsightFace buffalo_l...")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(320, 320))
print("InsightFace ready.\n")

_ensure_db()

def crop_head_region(img, box, expand=0.30):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    box_h = y2 - y1
    head_y2 = min(img.shape[0], y1 + int(box_h * expand))
    margin = max(10, int(box_h * 0.05))
    crop_y1 = max(0, y1 - margin)
    crop_x1 = max(0, x1)
    crop_x2 = min(img.shape[1], x2)
    face_img = img[crop_y1:head_y2, crop_x1:crop_x2]
    if face_img.size == 0:
        return None, None
    return face_img, (crop_x1, crop_y1, crop_x2, head_y2)

def draw_debug(frame, results, cam_id):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for r in results:
        box = r["box"]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = r.get("name", "?")
        status = r.get("status", "unknown")
        conf = r.get("confidence", 0.0)
        color = (0, 255, 0) if status == "authorized" else (0, 165, 255) if status == "unknown" else (0, 0, 255)
        text = f"{label} ({status}) conf={conf:.2f}"
        cv2.putText(frame, text, (x1, y1 - 10), font, 0.5, color, 2)
        cx, cy = r.get("crop_center", (0, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"cam_id={cam_id}", (10, 30), font, 0.7, (255, 255, 255), 2)
    stats = [f"min_size={TEST_PROFILE.get(cam_id,{}).get('min_size','?')}",
             f"quality={TEST_PROFILE.get(cam_id,{}).get('quality','?')}",
             f"thresh={TEST_PROFILE.get(cam_id,{}).get('sim_thresh','?')}"]
    for i, s in enumerate(stats):
        cv2.putText(frame, s, (10, 60 + i * 25), font, 0.5, (200, 200, 200), 1)
    return frame

def select_camera():
    devices = get_cameras_with_device_path()
    print("Available cameras:")
    for d in devices:
        print(f"  [{d['index']}] {d['name']}")
    print(f"\nCurrent CAM_PROFILE:")
    for cid, prof in CAM_PROFILE.items():
        print(f"  cam_id={cid}: {prof}")
    print(f"\nTEST_PROFILE (active):")
    for cid, prof in TEST_PROFILE.items():
        print(f"  cam_id={cid}: {prof}")
    while True:
        inp = input("\nEnter camera index (or cam_id like 'cam1'/'cam2'/'cam3'/'yolo'): ").strip()
        if inp.isdigit():
            return int(inp), int(inp)
        if inp.startswith("cam"):
            try:
                cid = int(inp.replace("cam", ""))
                mapping = {1: 1, 2: 2, 3: 3}
                if cid in mapping:
                    return mapping[cid], cid
            except: pass
        if inp == "yolo" or inp == "0":
            return 0, 0
        print("?? Try again.")

def main():
    phys_idx, cam_id = select_camera()
    print(f"\nOpening camera index {phys_idx} as cam_id={cam_id}...")
    cap = cv2.VideoCapture(phys_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("FAILED to open camera!")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(10):
        cap.read()
    print("Camera OK. Press 's'=save debug, 'q'=quit.\n")

    save_counter = 0
    frame_count = 0
    last_face_time = 0
    last_auto_save = 0
    face_interval = 0.5
    AUTO_SAVE_COOLDOWN = 2.0  # seconds between auto-saves

    print("Auto-save enabled: saves automatically when person detected + face found.")
    print("Press 's'=manual save, 'q'=quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Read failed")
            break
        frame_count += 1
        display = frame.copy()
        now = time.time()

        results = []
        yolo_results = yolo(frame, classes=[0], verbose=False)
        boxes = [b for b in yolo_results[0].boxes if float(b.conf[0]) >= MIN_CONFIDENCE]

        any_person = False
        any_face = False
        save_this = False

        for box in boxes:
            conf = float(box.conf[0])
            any_person = True
            crop_img, crop_box = crop_head_region(frame, box, expand=0.55)
            if crop_img is None:
                continue

            # Draw crop region
            if crop_box:
                cx1, cy1, cx2, cy2 = crop_box
                cv2.rectangle(display, (cx1, cy1), (cx2, cy2), (255, 0, 0), 1)

            entry = {
                "box": box.xyxy[0].tolist(),
                "det_conf": conf,
                "name": "?",
                "status": "unknown",
                "confidence": 0.0,
                "crop_center": ((cx1 + cx2)//2, (cy1 + cy2)//2) if crop_box else (0, 0),
                "details": []
            }

            if now - last_face_time >= face_interval:
                last_face_time = now

                # Step-by-step: try InsightFace directly on crop
                pre = _normalize_lighting(crop_img)
                raw_faces = app.get(pre)
                if not raw_faces:
                    entry["status"] = "no_face"
                    entry["details"].append("no face in crop")
                else:
                    any_face = True
                    raw_face = max(raw_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                    fw = raw_face.bbox[2] - raw_face.bbox[0]
                    fh = raw_face.bbox[3] - raw_face.bbox[1]
                    ds = raw_face.det_score
                    prof = TEST_PROFILE.get(cam_id, {})
                    min_sz = prof.get("min_size", 25)
                    min_q = prof.get("quality", 0.40)

                    entry["details"].append(f"face={fw:.0f}x{fh:.0f} score={ds:.3f}")
                    entry["details"].append(f"min_size={min_sz} quality={min_q}")

                    if ds < min_q:
                        entry["status"] = "low_quality"
                        entry["details"].append(f"LOW_QUALITY ({ds:.3f}<{min_q})")
                    elif fw < min_sz or fh < min_sz:
                        entry["status"] = "too_small"
                        entry["details"].append(f"TOO_SMALL ({fw:.0f}x{fh:.0f}<{min_sz})")
                    else:
                        # Full recognition
                        result = recognize_face(crop_img, cam_id=cam_id)
                        entry["name"] = result.get("name", "?")
                        entry["status"] = result.get("status", "unknown")
                        entry["confidence"] = result.get("confidence", 0.0)
                        entry["details"].append(f"rec={result.get('name','?')}({result.get('status','?')})")

            results.append(entry)

            # Print detailed info
            det = entry["details"]
            tag = f"[cam{cam_id}]"
            print(f"  {tag} person conf={conf:.2f} | {' | '.join(det)}")

        # Auto-save when person detected + cooldown expired
        if any_person and now - last_auto_save > AUTO_SAVE_COOLDOWN:
            save_this = True
            last_auto_save = now

        display = draw_debug(display, results, cam_id)

        # Show mini crop previews in corner
        for i, entry in enumerate(results[:3]):
            if "box" in entry:
                x1, y1, x2, y2 = map(int, entry["box"])
                crop_h = min(80, (y2-y1)//2)
                crop_w = int(crop_h * (x2-x1)/(y2-y1)) if (y2-y1) > 0 else 80
                try:
                    mini = cv2.resize(frame[y1:y1+crop_h, x1:x1+crop_w], (80, 80))
                    display[10 + i*90: 90 + i*90, 10:90] = mini
                except: pass

        cv2.imshow("Face Capture Test (auto-save when person detected)", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if save_this or key == ord('s'):
            save_counter += 1
            path = os.path.join(DEBUG_DIR, f"debug_cam{cam_id}_{save_counter}.jpg")
            cv2.imwrite(path, display)
            # Also save raw crop if available
            if results and crop_img is not None:
                raw_path = os.path.join(DEBUG_DIR, f"raw_cam{cam_id}_{save_counter}.jpg")
                cv2.imwrite(raw_path, crop_img)
            src = "auto" if save_this else "manual"
            print(f"  [{src}] Saved debug: debug_cam{cam_id}_{save_counter}.jpg")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()

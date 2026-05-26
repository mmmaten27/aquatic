"""
Camera Setup & Verification Tool
---------------------------------
STEP 1 — Scans all camera indices 0-15 (using DirectShow) and shows
         each one so you can identify which index belongs to which role.

STEP 2 — After you set CAMERA_CONFIG below, re-run to verify role mapping.

Role mapping (edit CAMERA_CONFIG after running STEP 1):
  YOLO_IDX      = physical index for YOLO / 即時 AI display
  SURV_IDX_MAP  = {logical_id: physical_index} for 攝影機 1-3 tiles
"""

import cv2
import sys

sys.stdout.reconfigure(encoding='utf-8')

# ─── CONFIGURATION  (edit after running STEP 1) ────────────────
YOLO_IDX     = 2          # physical camera index for YOLO
SURV_IDX_MAP = {          # logical display ID → physical camera index
    1: 3,                 # 攝影機 1  →  camera index 3
    2: 0,                 # 攝影機 2  →  camera index 0
    3: 1,                 # 攝影機 3  →  camera index 1
}
# ───────────────────────────────────────────────────────────────


def scan_cameras(max_idx=16):
    """Scan using DirectShow (same backend as check_cameras.py)."""
    print(f"  Scanning index 0-{max_idx-1} with DirectShow backend...")
    found = []
    for i in range(max_idx):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                found.append((i, w, h))
                print(f"    ✅  index {i}  —  {w}x{h}")
            else:
                print(f"    ⚠️  index {i}  —  opened but no frame")
        cap.release()
    return found


def show_camera(phys_idx, title_line1, title_line2="SPACE/ENTER=next   Q=quit",
                color=(0, 255, 100)):
    """Show live preview. Returns False if user pressed Q."""
    cap = cv2.VideoCapture(phys_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"  [!] Cannot open camera index {phys_idx}")
        return True

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"  Showing: {title_line1}")
    print(f"  {title_line2}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"  [!] Frame read failed for index {phys_idx}")
            break

        _, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 70), (20, 20, 20), -1)
        cv2.putText(frame, title_line1, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, title_line2, (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

        cv2.imshow("Camera Setup", frame)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord(' '), 13):   # SPACE or ENTER → next
            break
        if key == ord('q'):
            cap.release()
            return False

    cap.release()
    return True


def step1_discover(found):
    """Show every found camera in order — helps identify unknown indices."""
    print("\n─── STEP 1: Identify all cameras ─────────────────────")
    print("  กด SPACE/ENTER ดูกล้องถัดไป  |  Q ออก\n")

    for idx, w, h in found:
        role = "YOLO (即時 AI)" if idx == YOLO_IDX else ""
        for lid, pidx in SURV_IDX_MAP.items():
            if idx == pidx:
                role = f"攝影機 {lid}"
        label = f"index {idx}  ({w}x{h})"
        if role:
            label += f"  →  {role}"
        else:
            label += "  →  ??? (unknown)"

        ok = show_camera(idx, label, "SPACE/ENTER=next   Q=quit",
                         color=(0, 200, 255) if "YOLO" in role else (0, 255, 130))
        if not ok:
            return False
    return True


def step2_verify():
    """Show cameras in their assigned roles — confirms config is correct."""
    print("\n─── STEP 2: Verify role mapping ───────────────────────")
    print("  กด SPACE/ENTER ดูกล้องถัดไป  |  Q ออก\n")

    ok = show_camera(
        YOLO_IDX,
        f"YOLO Camera  (index {YOLO_IDX})",
        "Role: 即時 - YOLO AI section   SPACE/ENTER=next   Q=quit",
        color=(0, 200, 255),
    )
    if not ok:
        return

    for lid, pidx in sorted(SURV_IDX_MAP.items()):
        ok = show_camera(
            pidx,
            f"攝影機 {lid}  (index {pidx})",
            f"Role: /video_feed/{lid}  (監控)   SPACE/ENTER=next   Q=quit",
            color=(0, 255, 130),
        )
        if not ok:
            break


def main():
    print("=" * 56)
    print("  AQUATIC  —  Camera Setup & Verification")
    print("=" * 56)

    found = scan_cameras()
    if not found:
        print("\nไม่พบกล้องใด ปิดโปรแกรม")
        return

    indices = [i for i, w, h in found]
    print(f"\n  พบกล้องทั้งหมด {len(found)} ตัว: {indices}\n")

    # Check config completeness
    required = {YOLO_IDX} | set(SURV_IDX_MAP.values())
    missing  = [i for i in required if i not in indices]
    if missing:
        print(f"  ⚠️  config ยังไม่ครบ — ไม่พบ index: {missing}")
        print("      กรุณาดู STEP 1 เพื่อหา index ที่ถูกต้อง\n")
    else:
        print("  ✅  ทุก index ใน config พร้อมใช้งาน\n")

    # Role table
    print("─" * 56)
    known_roles = {YOLO_IDX: "YOLO / 即時 AI"}
    for lid, pidx in SURV_IDX_MAP.items():
        known_roles[pidx] = f"攝影機 {lid}"
    for idx, w, h in found:
        role = known_roles.get(idx, "??? not in config")
        print(f"  index {idx}  {w}x{h:<6}  →  {role}")
    print("─" * 56)

    # Choose mode
    print("\n  [1] STEP 1 — แสดงทุกกล้อง (ระบุ index ที่ไม่รู้จัก)")
    print("  [2] STEP 2 — ยืนยัน role mapping ตาม config")
    print("  [Q] ออก")
    choice = input("\n  เลือก (1/2/Q): ").strip().lower()

    if choice == '1':
        ok = step1_discover(found)
        cv2.destroyAllWindows()
        if ok:
            print("\n  ─── บันทึก index ที่เห็นแล้วแก้ CAMERA_CONFIG ───")
            print(f"  YOLO_IDX     = {YOLO_IDX}")
            for lid, pidx in sorted(SURV_IDX_MAP.items()):
                print(f"  攝影機 {lid}     = index {pidx}")
            print("\n  แก้ไขค่าในไฟล์นี้และใน app.py แล้วรัน STEP 2 เพื่อยืนยัน")

    elif choice == '2':
        step2_verify()
        cv2.destroyAllWindows()
        print("\n" + "=" * 56)
        print("  สรุป Config ที่ใช้งาน")
        print("=" * 56)
        print(f"  YOLO_IDX     = {YOLO_IDX}   → 即時 - YOLO AI")
        for lid, pidx in sorted(SURV_IDX_MAP.items()):
            print(f"  攝影機 {lid}     = index {pidx} → /video_feed/{lid}")
        print()
        print("  ✅ ถ้าภาพถูกต้องทุกตัว → รัน python app.py ได้เลย")
        print("  ❌ ถ้าต้องเปลี่ยน → แก้ CAMERA_CONFIG ในไฟล์นี้ และ app.py")
        print("=" * 56)
    else:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

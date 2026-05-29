# แนวทางแก้ไขและพัฒนาระบบ Face Recognition Access Control
## อ้างอิงจากการวิเคราะห์ DETECTION_SYSTEM.txt, face_utils.py และ Layout ห้องจริง

---

## สารบัญ
1. [สรุปปัญหาและสาเหตุ](#1-สรุปปัญหาและสาเหตุ)
2. [แผนที่กล้องและหน้าที่ที่ถูกต้อง](#2-แผนที่กล้องและหน้าที่ที่ถูกต้อง)
3. [แก้ไข face_utils.py](#3-แก้ไข-face_utilspy)
4. [แก้ไข EXIT Logic (app.py)](#4-แก้ไข-exit-logic-apppy)
5. [แก้ไข Occupancy Counter](#5-แก้ไข-occupancy-counter)
6. [แก้ไข IoU Tracker](#6-แก้ไข-iou-tracker)
7. [แก้ไข Camera Role Assignment](#7-แก้ไข-camera-role-assignment)
8. [ลำดับการ implement](#8-ลำดับการ-implement)

---

## 1. สรุปปัญหาและสาเหตุ

| ปัญหา | สาเหตุในโค้ด | ไฟล์ที่แก้ |
|---|---|---|
| False Reject สูง | `MIN_FACE_SIZE_PX=50` ตัดทิ้งหน้าที่ระยะ >3m | face_utils.py |
| False Positive | `MIN_INTERCLASS_MARGIN=0.05` เล็กเกินไป | face_utils.py |
| EXIT ไม่ stable | cam3 trigger EXIT เดี่ยวโดยไม่มี cam2 confirm | app.py |
| Track หลุดบ่อย | IoU threshold 0.25 สูงเกินสำหรับคนเดินเร็ว | app.py |
| Latency สูง | `det_size=(640,640)` ใหญ่เกินสำหรับ input 640×480 | face_utils.py |
| Occupancy drift | นับจาก camera detection ไม่ใช่ event | app.py |

---

## 2. แผนที่กล้องและหน้าที่ที่ถูกต้อง

```
ประตู (Door)
  │
  ├─ 攝影機 4 / cam0 (YOLO)
  │    ตำแหน่ง : บนซ้าย ใกล้ประตู
  │    หน้าที่  : Person detection หลัก, ENTER trigger
  │    พิเศษ   : ใช้ model.track() — thread เดียว
  │
  ├─ 攝影機 2 / cam2
  │    ตำแหน่ง : ซ้าย จ่อประตูตรงๆ
  │    หน้าที่  : Face recognition หลัก (เห็นหน้าทันทีที่เข้า)
  │              ยืนยันคนยังอยู่ในห้อง (EXIT confirmation)
  │
  │              [พื้นที่ในห้อง — โต๊ะ aquaponics]
  │
  ├─ 攝影機 1 / cam1
  │    ตำแหน่ง : กลางห้อง
  │    หน้าที่  : Backup identifier (ถ้า cam2 หลุดเฟรม)
  │              ไม่ใช้ trigger ENTER/EXIT
  │
  └─ 攝影機 3 / cam3
       ตำแหน่ง : มุมห้อง ถ่ายออกไปทางประตู
       หน้าที่  : Pre-EXIT signal เท่านั้น (เห็นแค่ขา)
                ไม่รัน face recognition
                ไม่ trigger EXIT โดยตรง
```

**Logic ENTER/EXIT ที่ถูกต้อง:**
```
ENTER:
  cam4 หรือ cam2 detect person → recognize face → log ENTRY → เพิ่ม occupancy

IN ROOM:
  cam2 เห็น → คนอยู่แถวประตู
  cam2 ไม่เห็น + cam1 เห็น → คนเดินเข้าไปกลางห้อง

EXIT (2-stage):
  Stage 1 — cam3 detect person (pre-signal: คนเดินไปทางประตู)
  Stage 2 — รอ EXIT_CONFIRM_WINDOW วินาที แล้วตรวจ cam2
    ├─ cam2 ไม่เห็นคนแล้ว → EXIT จริง → log EXIT → ลด occupancy
    └─ cam2 ยังเห็นคน    → เดินผ่านเฉยๆ → ยกเลิก EXIT pending
```

---

## 3. แก้ไข face_utils.py

### 3.1 ลด det_size (Quick Win — ลด latency ~40%)

```python
# เดิม
app.prepare(ctx_id=-1, det_size=(640, 640))

# แก้เป็น
# input จริงคือ 640×480 ไม่ใช่ 1280×720
# det_size=(320,320) เพียงพอและเร็วกว่ามาก
app.prepare(ctx_id=-1, det_size=(320, 320))
```

### 3.2 Soft Size Penalty แทน Hard Cutoff

**ปัญหา:** `MIN_FACE_SIZE_PX=50` ตัดทิ้งทันที ทำให้คนที่อยู่ไกล (cam1, cam3, ระยะ >3m)
ไม่ถูก recognize เลย ทั้งที่ยังพอ identify ได้

```python
# เพิ่ม constants ใหม่
MIN_FACE_SIZE_PX      = 30     # hard cutoff (เล็กมากๆ จริงๆ ไม่มีประโยชน์)
SOFT_SIZE_THRESHOLD   = 50     # ขนาดสมบูรณ์ ไม่มี penalty
HIGH_CONF_THRESHOLD   = 0.35   # distance ต่ำกว่านี้ = match ชัด ข้าม margin check ได้
MIN_INTERCLASS_MARGIN = 0.08   # เพิ่มจาก 0.05 → 0.08

# ใน recognize_face() แทนที่บล็อก size gate เดิม:

# --- Size Gate (soft) ---
fw = face.bbox[2] - face.bbox[0]
fh = face.bbox[3] - face.bbox[1]

if fw < MIN_FACE_SIZE_PX or fh < MIN_FACE_SIZE_PX:
    # เล็กเกินกว่าจะใช้งานได้จริง
    print(f"⚠️ FACE: too small ({fw:.0f}×{fh:.0f}px < {MIN_FACE_SIZE_PX}px)")
    return _unknown()

# คำนวณ penalty สำหรับหน้าที่เล็กกว่า SOFT_SIZE_THRESHOLD
size_penalty = 0.0
if fw < SOFT_SIZE_THRESHOLD or fh < SOFT_SIZE_THRESHOLD:
    ratio = min(fw, fh) / SOFT_SIZE_THRESHOLD          # 0.6–1.0
    size_penalty = (1.0 - ratio) * 0.08                # penalty สูงสุด +0.08
    print(f"📐 FACE: small face penalty={size_penalty:.3f} ({fw:.0f}×{fh:.0f}px)")

effective_threshold = SIMILARITY_THRESHOLD + size_penalty
```

### 3.3 Inter-class Margin — ยกเว้นถ้า confident มากพอ

```python
# แทนที่บล็อก margin check เดิม:

if best_dist > effective_threshold or best_folder is None:
    return _unknown()

# ถ้า distance ต่ำมากพอ (match ชัดเจน) ไม่ต้องตรวจ margin
if best_dist < HIGH_CONF_THRESHOLD:
    print(f"✅ FACE: high-confidence match dist={best_dist:.3f}, skip margin check")
else:
    if second_dist < float("inf"):
        margin = second_dist - best_dist
        if margin < MIN_INTERCLASS_MARGIN:
            print(f"⚠️ FACE: ambiguous (margin={margin:.3f} < {MIN_INTERCLASS_MARGIN})")
            return _unknown()
```

### 3.4 รับ cam_id เพื่อ tune per-camera (optional แต่แนะนำ)

```python
# เพิ่ม profile แยกตามกล้อง
CAM_PROFILE = {
    # cam0 (YOLO, ใกล้ประตู): threshold มาตรฐาน
    0: {"min_size": 35, "sim_thresh": 0.57, "quality": 0.45},
    # cam1 (กลางห้อง): relax เพราะเป็น backup
    1: {"min_size": 28, "sim_thresh": 0.60, "quality": 0.42},
    # cam2 (จ่อประตู, ใกล้สุด): เข้มขึ้น เพราะภาพควรดี
    2: {"min_size": 45, "sim_thresh": 0.52, "quality": 0.55},
    # cam3 (มุมห้อง, ไม่รัน face recognition — ดูหัวข้อ 7)
    3: {"min_size": 28, "sim_thresh": 0.60, "quality": 0.40},
}

def recognize_face(img_array, access_rules=None, cam_id=None):
    profile = CAM_PROFILE.get(cam_id, {
        "min_size": MIN_FACE_SIZE_PX,
        "sim_thresh": SIMILARITY_THRESHOLD,
        "quality":   MIN_FACE_QUALITY,
    })
    # ใช้ profile["min_size"], profile["sim_thresh"], profile["quality"]
    # แทนค่า hardcoded ในฟังก์ชัน
    ...
```

### 3.5 สรุป face_utils.py — diff สั้นๆ

```
MIN_FACE_SIZE_PX      : 50   → 30   (hard cutoff)
SOFT_SIZE_THRESHOLD   : ใหม่  = 50   (soft penalty zone)
HIGH_CONF_THRESHOLD   : ใหม่  = 0.35
MIN_INTERCLASS_MARGIN : 0.05 → 0.08
det_size              : (640,640) → (320,320)
recognize_face()      : รับ cam_id เพิ่ม, ใช้ CAM_PROFILE
```

---

## 4. แก้ไข EXIT Logic (app.py)

### 4.1 ปัญหาเดิม

```
cam3 detect person
    ↓
_cast_vote()
    ↓ (ถ้า vote ผ่าน)
update_sighting(EXIT)  ← trigger ทันทีโดยไม่ยืนยัน cam2
```

cam3 เห็นแค่ขา → YOLO confidence ต่ำ → vote ผ่านบ้างไม่ผ่านบ้าง → EXIT ไม่ stable

### 4.2 โครงสร้างใหม่ — 2-Stage EXIT

```python
# ── Constants ────────────────────────────────────────────────────────────────
EXIT_CONFIRM_WINDOW   = 8.0   # วินาที: หลังจาก cam3 trigger ให้รอนานเท่านี้
CAM2_RECENT_WINDOW    = 3.0   # วินาที: ถ้า cam2 เห็นคนใน 3 วินาที = ยังอยู่

# ── State ─────────────────────────────────────────────────────────────────────
# เพิ่ม dict นี้ใน global scope ของ app.py
exit_pending = {}
# โครงสร้าง: { person_id: {"triggered_at": float, "name": str} }

cam2_last_detection = {}
# โครงสร้าง: { person_id: float (timestamp ล่าสุดที่ cam2 เห็น) }
# person_id ใช้ track_id หรือ identity name ก็ได้ — ควร consistent กับที่ใช้ใน surv_track_ident


# ── ฟังก์ชันใหม่ ──────────────────────────────────────────────────────────────

def _cam2_still_present(person_id: str) -> bool:
    """
    ตรวจว่า cam2 เห็นคนคนนี้ใน CAM2_RECENT_WINDOW วินาทีที่ผ่านมาไหม
    ใช้ surv_track_ident ของ cam2 ที่มีอยู่แล้ว
    """
    now = time.time()
    last_seen = cam2_last_detection.get(person_id, 0.0)
    return (now - last_seen) < CAM2_RECENT_WINDOW


def _on_cam3_person_detected(person_id: str, name: str):
    """
    เรียกจาก surv_detect_loop เมื่อ cam3 detect person
    แทนที่การเรียก _cast_vote() → update_sighting(EXIT) โดยตรง
    """
    if person_id in exit_pending:
        return  # มี pending อยู่แล้ว ไม่ต้อง reset

    exit_pending[person_id] = {
        "triggered_at": time.time(),
        "name": name,
    }
    print(f"🚶 EXIT PENDING: {name} ({person_id}) — รอยืนยัน {EXIT_CONFIRM_WINDOW}s")


def _process_exit_confirmations():
    """
    เรียกใน loop หลัก หรือใน thread แยก ทุก ~1 วินาที
    ตรวจ exit_pending ทั้งหมดว่าครบ window แล้วหรือยัง
    """
    now = time.time()
    to_remove = []

    for person_id, state in exit_pending.items():
        elapsed = now - state["triggered_at"]

        if elapsed < EXIT_CONFIRM_WINDOW:
            continue  # ยังไม่ครบ window

        name = state["name"]

        if _cam2_still_present(person_id):
            # cam2 ยังเห็นคน → เดินผ่านเฉยๆ ไม่ได้ออก
            print(f"↩️ EXIT CANCELLED: {name} ยังอยู่ใน cam2")
        else:
            # cam2 ไม่เห็นแล้ว → EXIT จริง
            print(f"🚪 EXIT CONFIRMED: {name} ออกจากห้องแล้ว")
            update_sighting("EXIT", name, person_id)  # เรียก function เดิม

        to_remove.append(person_id)

    for pid in to_remove:
        del exit_pending[pid]


# ── แก้ใน surv_detect_loop() สำหรับ cam_id == 2 ──────────────────────────────
# ทุกครั้งที่ cam2 recognize ได้ ให้อัปเดต timestamp
def _update_cam2_seen(person_id: str):
    cam2_last_detection[person_id] = time.time()

# เรียกใน surv_detect_loop เมื่อ cam_id == 2 และ recognition สำเร็จ:
# if cam_id == 2 and result["name"]:
#     _update_cam2_seen(result["name"])


# ── แก้ใน surv_detect_loop() สำหรับ cam_id == 3 ──────────────────────────────
# แทนที่:
#   _cast_vote(cam_id=3, ...) → update_sighting(EXIT)
# ด้วย:
#   _on_cam3_person_detected(person_id, name)
```

### 4.3 เพิ่ม _process_exit_confirmations() ใน loop

```python
# ใน yolo_detect_loop() หรือ thread หลัก เพิ่ม call นี้ทุก ~1 วินาที:

last_exit_check = 0.0

while True:
    # ... detection code เดิม ...

    now = time.time()
    if now - last_exit_check >= 1.0:
        _process_exit_confirmations()
        last_exit_check = now
```

---

## 5. แก้ไข Occupancy Counter

### 5.1 ปัญหาเดิม

ถ้า occupancy ถูกคำนวณจาก live camera detection จะ drift ได้ เพราะ:
- กล้องแต่ละตัวครอบคลุมพื้นที่ต่างกัน
- คนที่อยู่ใน blind spot ทุกกล้องจะ "หายไป" จาก count

### 5.2 โครงสร้างใหม่ — Event-Based Occupancy

```python
# ── State ─────────────────────────────────────────────────────────────────────
# Source of truth: คนที่อยู่ในห้องตอนนี้
room_occupants = {}
# โครงสร้าง: { person_name: {"entry_time": float, "last_seen": float} }

occupancy_lock = threading.Lock()


# ── ฟังก์ชัน ──────────────────────────────────────────────────────────────────

def record_entry(name: str):
    """เรียกเมื่อ ENTER event ยืนยันแล้ว"""
    with occupancy_lock:
        if name not in room_occupants:
            room_occupants[name] = {
                "entry_time": time.time(),
                "last_seen":  time.time(),
            }
            print(f"✅ ENTER: {name} — occupancy={len(room_occupants)}")


def record_exit(name: str):
    """เรียกเมื่อ EXIT event ยืนยันแล้ว (จาก _process_exit_confirmations)"""
    with occupancy_lock:
        if name in room_occupants:
            del room_occupants[name]
            print(f"🚪 EXIT: {name} — occupancy={len(room_occupants)}")


def update_last_seen(name: str):
    """
    เรียกทุกครั้งที่กล้องใดก็ตาม recognize คนได้
    ใช้ป้องกัน stale occupancy (คนอยู่แต่ไม่ถูก detect นานๆ)
    """
    with occupancy_lock:
        if name in room_occupants:
            room_occupants[name]["last_seen"] = time.time()


def get_current_occupancy() -> dict:
    """ใช้ใน web dashboard route"""
    with occupancy_lock:
        return dict(room_occupants)


def _cleanup_stale_occupants(timeout_sec: float = 300.0):
    """
    Safety net: ถ้าคนไม่ถูกเห็นนานกว่า timeout → ถือว่าออกไปแล้ว
    ป้องกัน count ค้างถ้า EXIT event พลาด
    เรียกทุก 60 วินาที
    """
    now = time.time()
    with occupancy_lock:
        stale = [
            name for name, info in room_occupants.items()
            if now - info["last_seen"] > timeout_sec
        ]
        for name in stale:
            del room_occupants[name]
            print(f"⏱️ STALE CLEANUP: {name} ถูกลบออกจาก occupancy")
```

### 5.3 Dashboard route

```python
@app.route("/api/occupancy")
def api_occupancy():
    occ = get_current_occupancy()
    return jsonify({
        "count": len(occ),
        "persons": [
            {
                "name": name,
                "entry_time": info["entry_time"],
                "last_seen":  info["last_seen"],
            }
            for name, info in occ.items()
        ]
    })
```

---

## 6. แก้ไข IoU Tracker

### 6.1 ปัญหาเดิม

`_match_boxes()` ใช้ IoU threshold 0.25 เป็น hard cutoff ถ้าคนเดินเร็วและ
box ขยับมากกว่า 25% overlap ในเฟรมเดียว → track หลุด → ได้ track_id ใหม่
→ identity buffer ที่สะสมมาหายหมด

### 6.2 แก้ด้วย Center Distance Fallback

```python
# แทนที่ _match_boxes() เดิม ทั้งฟังก์ชัน

IOU_THRESHOLD       = 0.15    # ลดจาก 0.25 → 0.15
CENTER_DIST_LIMIT   = 0.20    # fallback: ยอมรับถ้า center ขยับ < 20% ของ frame width

def _match_boxes(prev_tracks, curr_boxes, frame_width=640):
    """
    จับคู่ bounding boxes ระหว่างเฟรม
    1. ลอง IoU ก่อน (threshold ต่ำกว่าเดิม)
    2. ถ้าไม่ผ่าน → fallback ด้วย center distance
    Returns: { curr_idx: track_id หรือ None }
    """
    if not prev_tracks or not curr_boxes:
        return {i: None for i in range(len(curr_boxes))}

    center_limit_px = frame_width * CENTER_DIST_LIMIT  # default = 128px

    matched   = {}
    used_tids = set()

    for curr_idx, curr_box in enumerate(curr_boxes):
        cx = (curr_box[0] + curr_box[2]) / 2
        cy = (curr_box[1] + curr_box[3]) / 2

        best_iou  = 0.0
        best_dist = float("inf")
        best_tid_iou  = None
        best_tid_dist = None

        for prev_box, tid in prev_tracks:
            if tid in used_tids:
                continue

            # --- Primary: IoU ---
            iou = _compute_iou(prev_box, curr_box)
            if iou > best_iou and iou >= IOU_THRESHOLD:
                best_iou     = iou
                best_tid_iou = tid

            # --- Fallback: Center Distance ---
            px = (prev_box[0] + prev_box[2]) / 2
            py = (prev_box[1] + prev_box[3]) / 2
            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if dist < best_dist and dist < center_limit_px:
                best_dist     = dist
                best_tid_dist = tid

        # ใช้ IoU ก่อน ถ้าไม่มีค่อยใช้ center distance
        chosen_tid = best_tid_iou if best_tid_iou is not None else best_tid_dist

        matched[curr_idx] = chosen_tid
        if chosen_tid is not None:
            used_tids.add(chosen_tid)

    return matched
```

### 6.3 เพิ่ม Buffer Size แยกตามกล้อง

```python
# ใน app.py แทนที่ค่า hardcoded ด้วย dict
SURV_TRACK_BUFFER_SIZE = {
    1: 12,   # cam1 กลางห้อง — backup, ต้องการ buffer มากกว่า
    2:  6,   # cam2 จ่อประตู — ใกล้, รู้ผลเร็ว ไม่ต้องการ buffer ใหญ่
    3:  4,   # cam3 มุมห้อง — ไม่รัน face recognition ใช้แค่ detect
}

# ใช้งาน:
buf_size = SURV_TRACK_BUFFER_SIZE.get(cam_id, 6)
```

---

## 7. แก้ไข Camera Role Assignment

### 7.1 ปิด Face Recognition บน cam3

cam3 เห็นแค่ขา การรัน ArcFace ไม่มีประโยชน์และกิน CPU เปล่า

```python
# ใน config.py หรือ app.py
CAM_NO_FACE_RECOGNITION = {3}   # set ของ cam_id ที่ไม่รัน face recognition

# ใน surv_detect_loop() เพิ่มเงื่อนไข:
if cam_id not in CAM_NO_FACE_RECOGNITION:
    result = recognize_face(best_frame, access_rules, cam_id=cam_id)
    # ... process result ...
else:
    # cam3: ใช้แค่ person detection เป็น pre-EXIT signal
    if len(curr_boxes) > 0:
        _on_cam3_person_detected(person_id="motion", name="unknown")
```

### 7.2 cam2 เป็น Primary Identifier

```python
# ใน surv_detect_loop() สำหรับ cam_id == 2
# เพิ่ม priority: ถ้า cam2 identify ได้ → lock identity 10 วินาที

CAM2_IDENTITY_LOCK_SEC = 10.0
cam2_identity_lock = {}
# โครงสร้าง: { person_id: {"name": str, "locked_until": float} }

def _try_lock_identity_from_cam2(person_id: str, name: str):
    cam2_identity_lock[person_id] = {
        "name":         name,
        "locked_until": time.time() + CAM2_IDENTITY_LOCK_SEC,
    }

def _get_locked_identity(person_id: str):
    """
    ถ้า cam2 เคย identify แล้วและยังอยู่ใน lock window
    → ให้กล้องอื่น reuse identity นี้ ไม่ต้อง recognize ใหม่
    """
    lock = cam2_identity_lock.get(person_id)
    if lock and time.time() < lock["locked_until"]:
        return lock["name"]
    return None
```

### 7.3 cam1 เป็น Backup Identifier

```python
# cam1 ควร recognize เฉพาะเมื่อ cam2 ยังไม่ identify track นั้นได้

# ใน surv_detect_loop() สำหรับ cam_id == 1:
locked_name = _get_locked_identity(track_id)
if locked_name:
    # cam2 รู้แล้ว ไม่ต้อง recognize ซ้ำ ประหยัด CPU
    result = {"name": locked_name, "status": "authorized", "confidence": 0.9}
else:
    # cam2 ยังไม่รู้ → cam1 ช่วย identify
    result = recognize_face(best_frame, access_rules, cam_id=1)
    if result["name"]:
        print(f"🔄 cam1 backup identified: {result['name']}")
```

---

## 8. ลำดับการ Implement

### Phase 1 — Quick Wins (ทำได้ทันที ไม่กระทบ logic หลัก)

```
✅ 1. ลด det_size=(320,320) ใน face_utils.py
      → ลด latency ~40%, ไม่กระทบ accuracy
      → แก้ 1 บรรทัด

✅ 2. เพิ่มรูปใน DB ต่อคน (5-8 รูป, หลายมุม, หลายแสง)
      → ลด False Reject โดยไม่แก้โค้ดเลย
      → สำคัญมาก เพราะ mean embedding จาก 1-2 รูปไม่เสถียร

✅ 3. เพิ่ม MIN_INTERCLASS_MARGIN = 0.08
      → ลด False Positive
      → แก้ 1 บรรทัดใน face_utils.py
```

### Phase 2 — Core Fixes (สำคัญ แก้ปัญหาหลัก)

```
🔧 4. Soft Size Penalty (หัวข้อ 3.2-3.3)
      → ลด False Reject สำหรับกล้องไกล (cam1)
      → แก้ ~15 บรรทัดใน recognize_face()

🔧 5. 2-Stage EXIT Logic (หัวข้อ 4)
      → EXIT stable ขึ้นมาก
      → เพิ่ม ~60 บรรทัดใน app.py

🔧 6. ปิด Face Recognition บน cam3 (หัวข้อ 7.1)
      → ลด CPU + ลด false EXIT trigger
      → แก้ ~5 บรรทัดใน surv_detect_loop
```

### Phase 3 — Stability (ทำหลัง phase 2 stable แล้ว)

```
🔩 7. Event-Based Occupancy Counter (หัวข้อ 5)
      → occupancy ไม่ drift อีก
      → refactor ~30 บรรทัด

🔩 8. IoU Center Distance Fallback (หัวข้อ 6)
      → track หลุดน้อยลง
      → แทนที่ _match_boxes() ทั้งฟังก์ชัน

🔩 9. CAM_PROFILE แยกตามกล้อง (หัวข้อ 3.4)
      → tune accuracy ต่อกล้องได้อิสระ
      → เพิ่ม ~15 บรรทัด + แก้ signature ของ recognize_face()
```

### Phase 4 — Optional Enhancements

```
💡 10. cam2 Identity Lock (หัวข้อ 7.2-7.3)
       → ลด CPU เพราะไม่ recognize ซ้ำซ้อน
       → ทำเมื่อ CPU เป็น bottleneck จริงๆ

💡 11. _cleanup_stale_occupants() (หัวข้อ 5.2)
       → safety net ถ้า EXIT event พลาด
       → เรียกทุก 60 วินาทีใน background thread
```

---

## ตาราง Parameters สรุป

| Parameter | ค่าเดิม | ค่าแนะนำ | เหตุผล |
|---|---|---|---|
| `det_size` | (640,640) | **(320,320)** | input จริง 640×480 |
| `MIN_FACE_SIZE_PX` | 50 | **30** (hard) + soft zone ถึง 50 | ระยะ >3m หน้า ~35px |
| `MIN_INTERCLASS_MARGIN` | 0.05 | **0.08** | ลด false positive |
| `HIGH_CONF_THRESHOLD` | ไม่มี | **0.35** | ข้าม margin check เมื่อ match ชัด |
| `SIMILARITY_THRESHOLD` | 0.55 | **0.55** (คงเดิม) | เหมาะสำหรับ DB เล็ก 2-5 คน |
| `IOU_THRESHOLD` | 0.25 | **0.15** + center fallback | ลด track หลุด |
| `EXIT_CONFIRM_WINDOW` | ไม่มี | **8.0 วินาที** | รอ cam2 ยืนยัน |
| `CAM2_RECENT_WINDOW` | ไม่มี | **3.0 วินาที** | นิยาม "ยังอยู่ใน cam2" |
| Buffer size cam1 | 6 | **12** | backup identifier ต้องการ buffer มาก |
| Buffer size cam3 | 6 | **4** | ไม่รัน face recognition |

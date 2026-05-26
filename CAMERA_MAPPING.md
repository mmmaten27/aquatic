# Camera Mapping Reference
## Aquaponics Monitoring System

### Physical Camera → Index (Current)
| Role | DShow Index | Device Name | USB Instance ID |
|------|-------------|-------------|-----------------|
| **YOLO AI** (即時) | **0** | Logi C270 HD WebCam | `...6&1bea6145...` |
| **攝影機 1** | **2** | A4tech FHD 1080P PC Camera | `...6&33ca29c9...` |
| **攝影機 2** | **3** | A4tech FHD 1080P PC Camera | `...6&19364c92...` |
| **攝影機 3** | **1** | A4tech FHD 1080P PC Camera | `...6&26daa0e0...` |
| — | 4 | OBS Virtual Camera | (ignore) |

### Files to check/modify

#### `app.py`
Lines where indices are set:
```python
# refresh_camera_indices()  ← auto from calibration
YOLO_CAM_IDX  # auto
SURV_CAM_IDX = {1: ?, 2: ?, 3: ?}  # auto
```

#### `detection/camera_calibration.json`
Persistent mapping by USB device path (auto-generated, edit if cameras moved between USB ports):
```json
{
  "yolo": "\\\\?\\usb#vid_046d&pid_0825&mi_00#6&1bea6145&0&0000...",
  "cam1": "\\\\?\\usb#vid_09da&pid_2695&mi_00#6&33ca29c9&0&0000...",
  "cam2": "\\\\?\\usb#vid_09da&pid_2695&mi_00#6&19364c92&0&0000...",
  "cam3": "\\\\?\\usb#vid_09da&pid_2695&mi_00#6&26daa0e0&0&0000..."
}
```

#### `detection/camera_utils.py`
Auto-detection logic when calibration file is missing.
Uses `pygrabber` to enumerate cameras → Logitech = cam2, A4tech sorted = yolo/cam1/cam3.

### Troubleshooting

**กล้องสลับกัน / เปิดผิดตัว:**
1. รัน: `python check_cameras.py` → ดู index 0-4 ว่าตัวไหน OK
2. รัน: `python detection/camera_utils.py` → ดู device path ปัจจุบัน
3. เปิด `detection/camera_calibration.json` → แก้ device path ให้ตรงกับ index ที่ถูกต้อง
4. หรือลบไฟล์ `detection/camera_calibration.json` ทิ้ง → restart `app.py` → ระบบ auto-calibrate ใหม่

**วิธีหา device path ของแต่ละ index:**
```python
from detection.camera_utils import get_cameras_with_device_path
for d in get_cameras_with_device_path():
    print(f"[{d['index']}] {d['name']}  →  {d['device_path']}")
```

หรือรัน: `python -c "from detection.camera_utils import get_cameras_with_device_path as g; [print(f'[{d[\"index\"]}] {d[\"name\"]}') for d in g()]"`

### Emergency steps ถ้ากล้องไม่ขึ้นเลย
```powershell
# 1. เช็ค index
python check_cameras.py

# 2. kill python processes (ถ้ากล้องค้าง)
taskkill /f /im python.exe

# 3. รัน setup เพื่อหา index ที่ถูกต้อง
python camera_setup.py

# 4. ลบ calibration แล้ว restart
del detection\camera_calibration.json
python app.py
```

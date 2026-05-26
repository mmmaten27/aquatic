import pythoncom
import json
import os
from pygrabber.dshow_graph import (
    SystemDeviceEnum, DeviceCategories, IPropertyBag, GUID
)

CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "camera_calibration.json")


def get_cameras_with_device_path():
    """Enumerate DirectShow cameras with index, name, and device path."""
    pythoncom.CoInitialize()
    result = []
    
    enum = SystemDeviceEnum()
    filter_enumerator = enum.system_device_enum.CreateClassEnumerator(
        GUID(DeviceCategories.VideoInputDevice), dwFlags=0
    )
    
    idx = 0
    try:
        moniker, count = filter_enumerator.Next(1)
        while count > 0:
            try:
                prop_bag = moniker.BindToStorage(
                    0, 0, IPropertyBag._iid_
                ).QueryInterface(IPropertyBag)
                
                friendly_name = prop_bag.Read("FriendlyName", pErrorLog=None)
                
                device_path = ""
                try:
                    device_path = prop_bag.Read("DevicePath", pErrorLog=None)
                except Exception:
                    pass
                
                result.append({
                    'index': idx,
                    'name': friendly_name,
                    'device_path': device_path,
                })
            except Exception:
                result.append({
                    'index': idx,
                    'name': f"Camera {idx}",
                    'device_path': "",
                })
            
            idx += 1
            moniker, count = filter_enumerator.Next(1)
    except ValueError:
        pass
    
    return result


def save_calibration(role_to_index):
    """
    Save the current camera mapping by device path.
    role_to_index: {'yolo': idx, 'cam1': idx, ...}
    """
    devices = get_cameras_with_device_path()
    index_to_dev = {d['index']: d for d in devices}
    
    calibration = {}
    for role, idx in role_to_index.items():
        dev = index_to_dev.get(idx)
        if dev and dev['device_path']:
            calibration[role] = dev['device_path']
    
    if calibration:
        with open(CALIBRATION_FILE, 'w') as f:
            json.dump(calibration, f, indent=2)
        print(f"[camera_utils] Saved calibration for {len(calibration)} cameras")
    return calibration


def load_calibration():
    """Load saved calibration if it exists."""
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, 'r') as f:
            return json.load(f)
    return {}


def get_calibrated_indices(auto_calibrate=True):
    """
    Get camera indices using calibration (device path matching).
    Falls back to name-based sorting if calibration is missing/stale.
    
    If auto_calibrate is True and no calibration exists, creates one
    using name-based sorting.
    
    Returns: {'yolo': idx, 'cam1': idx, 'cam2': idx, 'cam3': idx}
    and the full device list.
    """
    devices = get_cameras_with_device_path()
    calibration = load_calibration()
    result = {}
    
    if calibration:
        # Try to match each role by device path
        for role, saved_path in calibration.items():
            found = [d for d in devices if d['device_path'] == saved_path]
            if found:
                result[role] = found[0]['index']
    
    # Fill any missing roles using name-based fallback
    if len(result) < 4:
        logitech = [d for d in devices if 'logi' in d['name'].lower() or 'c270' in d['name'].lower()]
        a4tech = [d for d in devices if 'a4tech' in d['name'].lower()]
        a4tech.sort(key=lambda d: d['index'])
        logitech.sort(key=lambda d: d['index'])
        
        a4tech_idx = 0
        for role in ['yolo', 'cam1', 'cam2', 'cam3']:
            if role in result:
                continue
            if role == 'cam2' and logitech:
                result[role] = logitech[0]['index']
            elif role == 'yolo' and a4tech_idx < len(a4tech):
                result[role] = a4tech[a4tech_idx]['index']
                a4tech_idx += 1
            elif role == 'cam1' and a4tech_idx < len(a4tech):
                result[role] = a4tech[a4tech_idx]['index']
                a4tech_idx += 1
            elif role == 'cam3' and a4tech_idx < len(a4tech):
                result[role] = a4tech[a4tech_idx]['index']
                a4tech_idx += 1
    
    # Auto-save calibration on first run
    if not calibration and auto_calibrate and len(result) == 4:
        save_calibration(result)
        print(f"[camera_utils] Auto-saved calibration (first run)")
    
    return result, devices


def calibrate_from_current(role_to_index):
    """One-time calibration: saves device paths for current role→index mapping."""
    calibration = save_calibration(role_to_index)
    return calibration


if __name__ == '__main__':
    import sys
    print("=" * 60)
    print("  Camera Calibration Tool")
    print("=" * 60)
    
    devices = get_cameras_with_device_path()
    print(f"\nDetected {len(devices)} cameras:")
    for d in devices:
        print(f"  [{d['index']}] {d['name']}")
        if d['device_path']:
            print(f"       DevicePath: {d['device_path'][:70]}...")
    
    print("\n" + "-" * 60)
    print("  Current configuration (from app.py):")
    print("    yolo → index 2")
    print("    cam1 → index 3")
    print("    cam2 → index 0 (Logitech)")
    print("    cam3 → index 1")
    print()
    
    answer = input("  Save this as calibration? (y/N): ").strip().lower()
    if answer == 'y':
        calibrate_from_current({
            'yolo': 2,
            'cam1': 3,
            'cam2': 0,
            'cam3': 1,
        })
        print("  ✅ Calibration saved!")
    
    print("\n" + "-" * 60)
    mapping, _ = get_calibrated_indices()
    print("  Active mapping (after calibration):")
    for role, idx in sorted(mapping.items()):
        name = next((d['name'] for d in devices if d['index'] == idx), '?')
        print(f"    {role} → index {idx} ({name})")
    print("=" * 60)

if __name__ != '__main__':
    # When imported, get_calibrated_indices is the main entry point
    pass

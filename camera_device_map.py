import subprocess
import re

def get_pnp_camera_details():
    """Get PnP camera details including FriendlyName and InstanceId."""
    result = subprocess.run([
        'powershell', '-Command',
        'Get-PnpDevice -Class Camera | Select-Object FriendlyName, InstanceId | ConvertTo-Csv -NoTypeInformation'
    ], capture_output=True, text=True, encoding='utf-8')
    cameras = []
    lines = result.stdout.strip().split('\n')
    for line in lines[1:]:
        line = line.strip().strip('"')
        parts = line.split('","')
        if len(parts) >= 2:
            cameras.append({'name': parts[0], 'instance_id': parts[1]})
    return cameras

def get_dshow_camera_indices():
    """Get DirectShow camera indices using pygrabber."""
    from pygrabber.dshow_graph import FilterGraph
    graph = FilterGraph()
    return graph.get_input_devices()

def map_cameras():
    """Map DirectShow cameras to PnP devices by matching instance IDs."""
    print("=" * 60)
    print("  Camera Device Mapping")
    print("=" * 60)
    
    dshow_names = get_dshow_camera_indices()
    print(f"\nDirectShow cameras ({len(dshow_names)} found):")
    for i, name in enumerate(dshow_names):
        print(f"  Index {i}: {name}")
    
    pnp_cameras = get_pnp_camera_details()
    print(f"\nPnP Camera devices ({len(pnp_cameras)} found):")
    for cam in pnp_cameras:
        print(f"  {cam['name']}")
        print(f"    InstanceId: {cam['instance_id']}")
    
    print("\n" + "=" * 60)
    print("  Current mapping in app.py:")
    print("    YOLO CAM  → index 2")
    print("    攝影機 1  → index 3")
    print("    攝影機 2  → index 0")
    print("    攝影機 3  → index 1")
    print("=" * 60)
    print("\nRecommendation:")
    print("  Run camera_setup.py STEP 1 to verify each camera's position.")
    print("  If indices keep changing, connect cameras to different USB ports")
    print("  or label each USB cable to ensure consistent connection order.")
    
    # Show which DShow names correlate with PnP names
    dshow_set = set(dshow_names)
    pnp_set = set(c['name'] for c in pnp_cameras)
    print(f"\nUnique DShow devices: {dshow_set}")
    print(f"Unique PnP devices:   {pnp_set}")

if __name__ == '__main__':
    map_cameras()

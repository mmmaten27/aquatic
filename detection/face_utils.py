import os
import glob
import threading
from datetime import datetime

face_lock = threading.Lock()
# Absolute path derived from this file's location so it works regardless of CWD
FACES_DB           = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "faces_db")
MODEL_NAME         = "Facenet512"  # more robust than VGG-Face for webcam conditions
DISTANCE_THRESHOLD = 0.40          # cosine threshold for Facenet512 real-world webcam


def _time_to_str(val):
    """Convert MySQL timedelta or string to 'HH:MM'."""
    if val is None:
        return "00:00"
    if isinstance(val, str):
        return str(val)[:5]
    total = int(getattr(val, "total_seconds", lambda: 0)())
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}"


def _unknown():
    return {"name": None, "status": "unknown", "confidence": 0.0}


def recognize_face(img_array, access_rules=None):
    """
    Compare img_array against registered faces in faces_db/.
    access_rules: list of dicts {name, face_folder, access_start, access_end, is_active}
    Returns: {name, status, confidence}
      status: 'authorized' | 'unauthorized' | 'unknown'
    """
    try:
        from deepface import DeepFace

        if not os.path.exists(FACES_DB):
            print(f"⚠️ FACE: faces_db not found → {os.path.abspath(FACES_DB)}")
            return _unknown()

        subdirs = [
            d for d in os.listdir(FACES_DB)
            if os.path.isdir(os.path.join(FACES_DB, d)) and not d.startswith(".")
        ]
        print(f"🗂️ FACE: faces_db subdirs = {subdirs}")
        if not subdirs:
            print("⚠️ FACE: no person folders found")
            return _unknown()

        with face_lock:
            results = DeepFace.find(
                img_path=img_array,
                db_path=FACES_DB,
                model_name=MODEL_NAME,
                distance_metric="cosine",
                enforce_detection=False,
                threshold=1.0,   # disable internal filter, we check distance ourselves
                silent=True,
            )

        print(f"🔎 FACE: DeepFace results count = {len(results) if results else 0}")

        if not results or len(results) == 0:
            print("⚠️ FACE: DeepFace returned empty list")
            return _unknown()

        df = results[0]
        print(f"🔎 FACE: df shape = {df.shape}, columns = {list(df.columns)}")
        if df.empty:
            print("⚠️ FACE: DataFrame is empty")
            return _unknown()

        top = df.iloc[0]

        # Distance column name varies by DeepFace version and model name
        dist_col = next(
            (c for c in df.columns if "cosine" in c.lower() or c == "distance"),
            None,
        )
        distance = float(top[dist_col]) if dist_col else 1.0
        print(f"🔬 FACE DIST: {distance:.3f} (threshold={DISTANCE_THRESHOLD}) col={dist_col}")

        if distance > DISTANCE_THRESHOLD:
            return _unknown()

        # Extract folder name (person identity) from path
        identity_path = str(top.get("identity", "")).replace("\\", "/")
        parts = identity_path.split("/")
        folder_name = parts[-2] if len(parts) >= 2 else ""
        confidence = round(1.0 - distance, 2)

        if not folder_name:
            return _unknown()

        # Match against access rules
        if access_rules:
            rule = next(
                (r for r in access_rules if r.get("face_folder") == folder_name), None
            )
            if rule:
                display_name = rule.get("name", folder_name)
                if not rule.get("is_active", True):
                    return {"name": display_name, "status": "unauthorized", "confidence": confidence}
                now_str = datetime.now().strftime("%H:%M")
                start = _time_to_str(rule.get("access_start", "00:00"))
                end   = _time_to_str(rule.get("access_end",   "23:59"))
                status = "authorized" if start <= now_str <= end else "unauthorized"
                return {"name": display_name, "status": status, "confidence": confidence}

        # No rules → report name only, mark authorized
        return {"name": folder_name, "status": "authorized", "confidence": confidence}

    except Exception as e:
        print(f"❌ FACE RECOGNITION: {e}")
        return _unknown()


def clear_face_cache():
    """Delete DeepFace pkl representations so the DB rebuilds on next call."""
    for pkl in glob.glob(os.path.join(FACES_DB, "**", "*.pkl"), recursive=True):
        try:
            os.remove(pkl)
        except Exception:
            pass
    print("🗑️ FACE DB: representation cache cleared")


def save_face_image(face_folder, file_storage):
    """
    Save one uploaded photo to faces_db/<face_folder>/photo<N>.jpg,
    auto-numbered so existing photos are never overwritten.
    """
    dest_dir = os.path.join(FACES_DB, face_folder)
    print(f"📁 SAVE FACE: dest_dir = {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)
    i = 1
    while os.path.exists(os.path.join(dest_dir, f"photo{i}.jpg")):
        i += 1
    dest_path = os.path.join(dest_dir, f"photo{i}.jpg")
    print(f"💾 SAVE FACE: saving → {dest_path}")
    file_storage.save(dest_path)
    print(f"✅ SAVE FACE: saved OK → {dest_path}")
    clear_face_cache()
    return dest_path


def list_face_photos(face_folder):
    """Return sorted list of image filenames in faces_db/<face_folder>/."""
    dest_dir = os.path.join(FACES_DB, face_folder)
    if not os.path.exists(dest_dir):
        return []
    return sorted([
        f for f in os.listdir(dest_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])


def get_first_photo_path(face_folder):
    """Return the path to the first available photo, or None."""
    photos = list_face_photos(face_folder)
    return os.path.join(FACES_DB, face_folder, photos[0]) if photos else None


def delete_face_photo(face_folder, filename):
    """Delete one specific photo and rebuild cache. Returns True on success."""
    safe = os.path.basename(filename)
    photo_path = os.path.join(FACES_DB, face_folder, safe)
    if os.path.isfile(photo_path):
        os.remove(photo_path)
        clear_face_cache()
        return True
    return False


def delete_face_folder(face_folder):
    """Remove a person's entire face folder and clear the cache."""
    import shutil
    folder_path = os.path.join(FACES_DB, face_folder)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    clear_face_cache()

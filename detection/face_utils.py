import os
import glob
import shutil
import threading
import numpy as np
import cv2
from datetime import datetime

face_lock = threading.Lock()
FACES_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "faces_db")

# ── Recognition thresholds ────────────────────────────────────────────────────
SIMILARITY_THRESHOLD  = 0.55   # cosine distance: lower = stricter (same person)
MIN_FACE_QUALITY      = 0.50   # InsightFace det_score: reject blurry/partial faces
MIN_FACE_SIZE_PX      = 30     # hard cutoff: face smaller than this → reject immediately
SOFT_SIZE_THRESHOLD   = 50     # face between 30-50 px → apply size penalty to threshold
HIGH_CONF_THRESHOLD   = 0.35   # distance below this → skip margin check (confident match)
MIN_INTERCLASS_MARGIN = 0.08   # 2nd-best must be this much worse → reject ambiguous

_app = None
_app_lock = threading.Lock()
_db = {}        # {folder_name: [L2-normalized embedding, ...]}
_db_lock = threading.Lock()
_db_ready = False


def _time_to_str(val):
    if val is None:
        return "00:00"
    if isinstance(val, str):
        return str(val)[:5]
    total = int(getattr(val, "total_seconds", lambda: 0)())
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}"


def _unknown():
    return {"name": None, "status": "unknown", "confidence": 0.0}


def _normalize_lighting(img):
    """CLAHE on L channel — equalises uneven lighting before embedding extraction."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _get_app():
    """Lazy-load InsightFace FaceAnalysis (downloads ~300 MB buffalo_l on first run)."""
    global _app
    if _app is None:
        with _app_lock:
            if _app is None:
                from insightface.app import FaceAnalysis
                app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
                app.prepare(ctx_id=-1, det_size=(320, 320))  # input 640×480 → (320,320) is enough, ~40% faster
                _app = app
    return _app


def _build_db():
    """Read every photo in faces_db, extract ArcFace embeddings, cache in memory."""
    global _db, _db_ready
    import cv2

    new_db = {}
    if not os.path.exists(FACES_DB):
        print(f"⚠️ FACE DB: {FACES_DB} not found")
        return

    app = _get_app()

    for folder in os.listdir(FACES_DB):
        folder_path = os.path.join(FACES_DB, folder)
        if not os.path.isdir(folder_path) or folder.startswith("."):
            continue

        embeddings = []
        for img_path in (
            glob.glob(os.path.join(folder_path, "*.jpg"))
            + glob.glob(os.path.join(folder_path, "*.jpeg"))
            + glob.glob(os.path.join(folder_path, "*.png"))
        ):
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = _normalize_lighting(img)   # same CLAHE as live recognition
            with face_lock:
                faces = app.get(img)
            if not faces:
                print(f"⚠️ FACE DB: no face in {img_path}")
                continue
            # Largest detected face is the subject
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            emb = face.embedding.copy().astype(np.float32)
            emb /= np.linalg.norm(emb)
            embeddings.append(emb)

        if embeddings:
            # Average all photo embeddings → one stable canonical vector per person
            mean_emb = np.mean(embeddings, axis=0).astype(np.float32)
            mean_emb /= np.linalg.norm(mean_emb)
            # Keep individual embeddings alongside mean for better coverage
            new_db[folder] = embeddings + [mean_emb]
            print(f"✅ FACE DB: '{folder}' → {len(embeddings)} photos + 1 mean embedding")
        else:
            print(f"⚠️ FACE DB: '{folder}' has no usable photos")

    with _db_lock:
        _db = new_db
        _db_ready = True
    print(f"✅ FACE DB: ready — {len(new_db)} person(s)")


def _ensure_db():
    if not _db_ready:
        _build_db()


def recognize_face(img_array, access_rules=None):
    """
    Identify the person in img_array (BGR numpy array) using ArcFace embeddings.
    Returns: {name, status, confidence}
      status: 'authorized' | 'unauthorized' | 'unknown'
    """
    try:
        _ensure_db()
        app = _get_app()

        # Layer 1: lighting normalisation before detection
        preprocessed = _normalize_lighting(img_array)

        with face_lock:
            faces = app.get(preprocessed)

        if not faces:
            print("⚠️ FACE: no face detected")
            return _unknown()

        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        # Layer 2: face quality gate (det_score = InsightFace confidence in detection)
        if face.det_score < MIN_FACE_QUALITY:
            print(f"⚠️ FACE: low quality det_score={face.det_score:.2f} < {MIN_FACE_QUALITY}")
            return _unknown()

        # Layer 3: face size gate (soft penalty zone)
        fw = face.bbox[2] - face.bbox[0]
        fh = face.bbox[3] - face.bbox[1]
        if fw < MIN_FACE_SIZE_PX or fh < MIN_FACE_SIZE_PX:
            print(f"⚠️ FACE: too small ({fw:.0f}×{fh:.0f}px < {MIN_FACE_SIZE_PX}px)")
            return _unknown()
        size_penalty = 0.0
        if fw < SOFT_SIZE_THRESHOLD or fh < SOFT_SIZE_THRESHOLD:
            ratio = min(fw, fh) / SOFT_SIZE_THRESHOLD
            size_penalty = (1.0 - ratio) * 0.08
            print(f"📐 FACE: small face penalty={size_penalty:.3f} ({fw:.0f}×{fh:.0f}px)")
        effective_threshold = SIMILARITY_THRESHOLD + size_penalty

        emb = face.embedding.copy().astype(np.float32)
        emb /= np.linalg.norm(emb)

        best_folder  = None
        best_dist    = float("inf")
        second_dist  = float("inf")

        with _db_lock:
            db_snapshot = {k: list(v) for k, v in _db.items()}

        for folder_name, embeddings in db_snapshot.items():
            for db_emb in embeddings:
                dist = 1.0 - float(np.dot(emb, db_emb))
                if dist < best_dist:
                    second_dist = best_dist
                    best_dist   = dist
                    best_folder = folder_name
                elif dist < second_dist:
                    second_dist = dist

        print(f"🔬 FACE DIST: best={best_dist:.3f} 2nd={second_dist:.3f} → {best_folder}")

        if best_dist > effective_threshold or best_folder is None:
            return _unknown()

        # Layer 4: inter-class margin — skip if match is very confident
        if best_dist < HIGH_CONF_THRESHOLD:
            print(f"✅ FACE: high-confidence match dist={best_dist:.3f}, skip margin check")
        else:
            if second_dist < float("inf"):
                margin = second_dist - best_dist
                if margin < MIN_INTERCLASS_MARGIN:
                    print(f"⚠️ FACE: ambiguous (margin={margin:.3f} < {MIN_INTERCLASS_MARGIN})")
                    return _unknown()

        confidence = round(1.0 - best_dist, 2)

        if access_rules:
            rule = next(
                (r for r in access_rules if r.get("face_folder") == best_folder), None
            )
            if rule:
                display_name = rule.get("name", best_folder)
                if not rule.get("is_active", True):
                    return {"name": display_name, "status": "unauthorized", "confidence": confidence}
                now_str = datetime.now().strftime("%H:%M")
                start = _time_to_str(rule.get("access_start", "00:00"))
                end   = _time_to_str(rule.get("access_end",   "23:59"))
                status = "authorized" if start <= now_str <= end else "unauthorized"
                return {"name": display_name, "status": status, "confidence": confidence}

        return {"name": best_folder, "status": "authorized", "confidence": confidence}

    except Exception as e:
        print(f"❌ FACE RECOGNITION: {e}")
        return _unknown()


def clear_face_cache():
    """Invalidate the in-memory embedding cache so it rebuilds on next call."""
    global _db_ready
    with _db_lock:
        _db.clear()
    _db_ready = False
    print("🗑️ FACE DB: embedding cache cleared — will rebuild on next recognition")


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
    """Delete one specific photo and invalidate cache. Returns True on success."""
    safe = os.path.basename(filename)
    photo_path = os.path.join(FACES_DB, face_folder, safe)
    if os.path.isfile(photo_path):
        os.remove(photo_path)
        clear_face_cache()
        return True
    return False


def delete_face_folder(face_folder):
    """Remove a person's entire face folder and clear the cache."""
    folder_path = os.path.join(FACES_DB, face_folder)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    clear_face_cache()

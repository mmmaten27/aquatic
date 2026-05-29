"""
Face recognition diagnostic.
Run: python test_face_recog.py
Tests every photo in faces_db against itself and cross-tests to show actual distances.
"""
import os, sys, glob
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))

FACES_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faces_db")

SIMILARITY_THRESHOLD  = 0.45
MIN_FACE_QUALITY      = 0.72
MIN_FACE_SIZE_PX      = 60
MIN_INTERCLASS_MARGIN = 0.08

print("Loading InsightFace buffalo_l...")
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))
print("Model ready.\n")


def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.merge((c.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ── Build DB ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Building embeddings from faces_db/")
print("=" * 60)
db = {}  # {folder: [embeddings]}

for folder in os.listdir(FACES_DB):
    path = os.path.join(FACES_DB, folder)
    if not os.path.isdir(path) or folder.startswith('.'):
        continue
    embeddings = []
    for img_path in sorted(glob.glob(os.path.join(path, "*.jpg")) +
                           glob.glob(os.path.join(path, "*.png"))):
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ❌ Cannot read: {img_path}")
            continue
        faces = app.get(img)
        if not faces:
            faces = app.get(clahe(img))  # try with CLAHE
        if not faces:
            print(f"  ⚠️  NO FACE detected in {os.path.basename(img_path)} (folder: {folder})")
            continue
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        fw = face.bbox[2] - face.bbox[0]
        fh = face.bbox[3] - face.bbox[1]
        score = face.det_score
        emb = face.embedding.copy().astype(np.float32)
        emb /= np.linalg.norm(emb)
        embeddings.append(emb)
        q_ok = "✅" if score >= MIN_FACE_QUALITY else f"❌ LOW QUALITY ({score:.2f}<{MIN_FACE_QUALITY})"
        s_ok = "✅" if fw >= MIN_FACE_SIZE_PX and fh >= MIN_FACE_SIZE_PX else f"❌ TOO SMALL ({fw:.0f}x{fh:.0f}px)"
        print(f"  [{folder}] {os.path.basename(img_path)}: det_score={score:.3f} {q_ok}  size={fw:.0f}x{fh:.0f} {s_ok}")
    if embeddings:
        mean_emb = np.mean(embeddings, axis=0).astype(np.float32)
        mean_emb /= np.linalg.norm(mean_emb)
        db[folder] = embeddings + [mean_emb]
        print(f"  → {folder}: {len(embeddings)} embeddings + 1 mean  TOTAL={len(db[folder])}\n")
    else:
        print(f"  → {folder}: NO EMBEDDINGS — folder will be ignored!\n")

# ── Distance matrix ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Cosine distance matrix (lower = more similar)")
print("  Threshold to MATCH: < {:.2f}".format(SIMILARITY_THRESHOLD))
print("=" * 60)

folders = list(db.keys())
for folder in folders:
    probe_emb = db[folder][-1]  # use mean embedding as probe
    print(f"\n  Probe: {folder} (mean embedding)")
    best_dist   = float('inf')
    second_dist = float('inf')
    best_folder = None
    for cmp_folder, embs in db.items():
        min_d = min(1.0 - float(np.dot(probe_emb, e)) for e in embs)
        match = "← MATCH" if min_d < SIMILARITY_THRESHOLD else "      "
        print(f"    vs {cmp_folder:20s}: dist={min_d:.4f}  {match}")
        if min_d < best_dist:
            second_dist = best_dist
            best_dist   = min_d
            best_folder = cmp_folder
        elif min_d < second_dist:
            second_dist = min_d

    margin = second_dist - best_dist if second_dist < float('inf') else float('inf')
    result_ok   = best_dist < SIMILARITY_THRESHOLD
    margin_ok   = margin >= MIN_INTERCLASS_MARGIN if second_dist < float('inf') else True
    if result_ok and margin_ok:
        print(f"  ✅ Would recognise as: {best_folder}  (dist={best_dist:.4f}, margin={margin:.4f})")
    elif not result_ok:
        print(f"  ❌ dist={best_dist:.4f} > threshold {SIMILARITY_THRESHOLD} → UNKNOWN")
    else:
        print(f"  ❌ margin={margin:.4f} < {MIN_INTERCLASS_MARGIN} → UNKNOWN (ambiguous)")

print("\n" + "=" * 60)
print("  Summary of gate values:")
print(f"    SIMILARITY_THRESHOLD  = {SIMILARITY_THRESHOLD}  (distance must be LOWER)")
print(f"    MIN_FACE_QUALITY      = {MIN_FACE_QUALITY}  (InsightFace det_score)")
print(f"    MIN_FACE_SIZE_PX      = {MIN_FACE_SIZE_PX}  (face px)")
print(f"    MIN_INTERCLASS_MARGIN = {MIN_INTERCLASS_MARGIN}  (gap between 1st and 2nd best)")
print("=" * 60)

"""
Bước 1 — Đo góc thực tế từ dữ liệu của bạn
============================================
Chạy model gốc trên toàn bộ my_captures/ để biết
trung bình yaw/pitch cho từng nhãn → dùng làm regression target.

Chạy:
    python measure_angles.py

Output:
    angle_targets.json   ← dùng cho fine_tune.py
"""

import os, json, cv2, time
import numpy as np
import onnxruntime as ort
from collections import defaultdict

DATASET_DIR = "my_captures"
MODEL_PATH  = "weights/mobileone_s0_gaze.onnx"
OUTPUT_JSON = "angle_targets.json"
LABELS      = ["left", "right", "up", "down", "center"]
INPUT_SIZE  = (448, 448)

# ─── Load model ──────────────────────────────────────────────────────────────
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess      = ort.InferenceSession(MODEL_PATH, opts, providers=["CPUExecutionProvider"])
inp_name  = sess.get_inputs()[0].name

def predict_raw(img):
    x = cv2.resize(img, INPUT_SIZE)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = ((x - mean) / std).transpose(2, 0, 1)
    x = np.expand_dims(x, 0).astype(np.float32)
    out = sess.run(None, {inp_name: x})

    def decode(logits):
        e = np.exp(logits - np.max(logits))
        p = e / e.sum()
        return float(np.sum(p * np.arange(len(p))) * 4 - 180)

    return decode(out[0][0]), decode(out[1][0])   # yaw, pitch

# ─── Haar + CLAHE detect ─────────────────────────────────────────────────────
cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def crop_face(img):
    gray  = clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    faces = cascade.detectMultiScale(gray, 1.2, 4, minSize=(40, 40))
    if len(faces) == 0:
        faces = cascade.detectMultiScale(gray, 1.1, 2, minSize=(30, 30))
    if len(faces) == 0:
        return img   # fallback toàn ảnh
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    m = int(0.2 * w)
    return img[max(0,y-m):y+h+m, max(0,x-m):x+w+m]

# ─── Đo ──────────────────────────────────────────────────────────────────────
results = defaultdict(lambda: {"yaws": [], "pitches": [], "n": 0, "skipped": 0})

for label in LABELS:
    folder = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(folder):
        print(f"[SKIP] {folder} không tồn tại")
        continue

    files = [f for f in sorted(os.listdir(folder))
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"\n[{label}]  {len(files)} ảnh")

    for fname in files:
        img = cv2.imread(os.path.join(folder, fname))
        if img is None:
            results[label]["skipped"] += 1
            continue

        face = crop_face(img)
        if face.size == 0:
            results[label]["skipped"] += 1
            continue

        yaw, pitch = predict_raw(face)
        results[label]["yaws"].append(yaw)
        results[label]["pitches"].append(pitch)
        results[label]["n"] += 1

# ─── Tổng hợp + in kết quả ───────────────────────────────────────────────────
print("\n" + "=" * 50)
print(f"{'Label':<8}  {'N':>5}  {'Yaw mean':>10}  {'Yaw std':>8}  {'Pitch mean':>11}  {'Pitch std':>9}")
print("-" * 50)

targets = {}
for label in LABELS:
    r = results[label]
    if r["n"] == 0:
        print(f"{label:<8}  {'—':>5}")
        continue

    yaw_mean   = float(np.mean(r["yaws"]))
    yaw_std    = float(np.std(r["yaws"]))
    pitch_mean = float(np.mean(r["pitches"]))
    pitch_std  = float(np.std(r["pitches"]))

    print(f"{label:<8}  {r['n']:>5}  {yaw_mean:>+10.2f}  {yaw_std:>8.2f}"
          f"  {pitch_mean:>+11.2f}  {pitch_std:>9.2f}"
          + (f"  (skipped {r['skipped']})" if r["skipped"] else ""))

    targets[label] = {
        "yaw":         round(yaw_mean, 2),
        "pitch":       round(pitch_mean, 2),
        "yaw_std":     round(yaw_std, 2),
        "pitch_std":   round(pitch_std, 2),
        "n":           r["n"],
    }

print("=" * 50)

# ─── Lưu JSON ────────────────────────────────────────────────────────────────
with open(OUTPUT_JSON, "w") as f:
    json.dump(targets, f, indent=2)

print(f"\nSaved → {OUTPUT_JSON}")
print("\nKiểm tra:")
print("  • left/right  nên có |yaw_mean| lớn, pitch_mean gần 0")
print("  • up/down     nên có |pitch_mean| lớn, yaw_mean gần 0")
print("  • center      nên có cả 2 gần 0")
print("  • std nhỏ = dữ liệu nhất quán, std lớn = cần thêm ảnh hoặc loại ảnh nhiễu")

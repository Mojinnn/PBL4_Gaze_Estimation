import os
import cv2
import time
import numpy as np
import onnxruntime as ort

DATASET_DIR = "data"
LABELS = ["left", "right", "center", "up", "down"]

# ═══════════════════════════════════════════════════════════════════
#  CẤU HÌNH RIÊNG CHO TỪNG MODEL
#  Phân tích từ benchmark data:
#
#  MobileOne s0:
#    GT:up   → pitch thường +1 đến +15  (dương)   → pitch_up_sign = +1
#    Range nhỏ (~±8°) → threshold nhỏ
#
#  MobileNet v2:
#    GT:up   → pitch thường -8 đến -20  (âm lớn)  → pitch_up_sign = -1
#    GT:down → pitch thường -5 đến -23  (âm lớn)  → cùng chiều up → model bị nhầm
#    Range lớn hơn (~±15°) → threshold lớn hơn
# ═══════════════════════════════════════════════════════════════════

MODEL_CONFIGS = {
    "weights/mobileone_s0_gaze.onnx": {
        "input_size": (448, 448),
        "yaw_threshold": 20.0,
        "pitch_threshold": 6.0,
        "pitch_up_sign": +1,   # pitch > 0 = up
    },
    "weights/mobilenetv2_gaze.onnx": {
        "input_size": (448, 448),
        "yaw_threshold": 20.0,
        "pitch_threshold": 8.0,
        "pitch_up_sign": -1,   # pitch < 0 = up
    },
}

# ← Chọn model ở đây
# MODEL_PATH = "weights/mobileone_s0_gaze.onnx"
MODEL_PATH = "weights/mobilenetv2_gaze.onnx"


class GazeBenchmark:
    def __init__(self, model_path):
        cfg = MODEL_CONFIGS[model_path]
        self.input_size      = cfg["input_size"]
        self.yaw_threshold   = cfg["yaw_threshold"]
        self.pitch_threshold = cfg["pitch_threshold"]
        self.pitch_up_sign   = cfg["pitch_up_sign"]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        self.model = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )

    def preprocess(self, img):
        img  = cv2.resize(img, self.input_size)
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img  = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img  = (img - mean) / std
        img  = np.transpose(img, (2, 0, 1))
        img  = np.expand_dims(img, axis=0)
        return img

    def predict(self, img):
        input_tensor = self.preprocess(img)
        input_name   = self.model.get_inputs()[0].name
        start        = time.time()
        outputs      = self.model.run(None, {input_name: input_tensor})
        inference_time = time.time() - start

        yaw_pred   = outputs[0][0]
        pitch_pred = outputs[1][0]

        yaw_exp    = np.exp(yaw_pred   - np.max(yaw_pred))
        yaw_prob   = yaw_exp / np.sum(yaw_exp)
        pitch_exp  = np.exp(pitch_pred - np.max(pitch_pred))
        pitch_prob = pitch_exp / np.sum(pitch_exp)

        bins  = np.arange(len(yaw_prob))
        yaw   = np.sum(yaw_prob  * bins) * 4 - 180
        bins  = np.arange(len(pitch_prob))
        pitch = np.sum(pitch_prob * bins) * 4 - 180

        print(f"yaw:{yaw:.2f} pitch:{pitch:.2f}")
        return pitch, yaw, inference_time

    def get_direction(self, pitch, yaw):
        """
        Priority-based classification:
          1. Pitch axis trước (up/down) — dùng pitch_up_sign để chuẩn hóa chiều
          2. Yaw axis sau (left/right)
          3. Fallback → center

        pitch_up_sign = +1: pitch dương = nhìn lên  (MobileOne)
        pitch_up_sign = -1: pitch âm    = nhìn lên  (MobileNetV2)
        """
        signed_pitch = pitch * self.pitch_up_sign

        if signed_pitch > self.pitch_threshold:
            return "up"
        if signed_pitch < -self.pitch_threshold:
            return "down"

        if yaw > self.yaw_threshold:
            return "right"
        if yaw < -self.yaw_threshold:
            return "left"

        return "center"


def benchmark():
    model = GazeBenchmark(MODEL_PATH)
    print(f"Model : {MODEL_PATH}")
    print(f"Config: yaw_thr={model.yaw_threshold}°  "
          f"pitch_thr={model.pitch_threshold}°  "
          f"pitch_up_sign={model.pitch_up_sign:+d}\n")

    total      = 0
    correct    = 0
    total_time = 0
    per_class  = {label: {"total": 0, "correct": 0} for label in LABELS}

    for label in LABELS:
        folder = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(folder):
            print(f"[WARN] Folder not found: {folder}")
            continue
        for file in sorted(os.listdir(folder)):
            path = os.path.join(folder, file)
            img  = cv2.imread(path)
            if img is None:
                continue

            pitch, yaw, t = model.predict(img)
            pred           = model.get_direction(pitch, yaw)

            total      += 1
            total_time += t
            per_class[label]["total"] += 1

            if pred == label:
                correct += 1
                per_class[label]["correct"] += 1

            print(f"{total} | GT:{label} | Pred:{pred}")

    if total == 0:
        print("No images found.")
        return

    avg_time = total_time / total
    accuracy = correct / total

    print("\n======= BENCHMARK RESULT =======")
    print(f"Model              : {MODEL_PATH}")
    print(f"Total images       : {total}")
    print(f"Average infer time : {avg_time*1000:.2f} ms")
    print(f"FPS                : {1/avg_time:.2f}")
    print(f"Overall Accuracy   : {accuracy*100:.2f}%")
    print("\n── Per-class Accuracy ──")
    for label in LABELS:
        c   = per_class[label]["correct"]
        n   = per_class[label]["total"]
        acc = (c / n * 100) if n > 0 else 0.0
        bar = "█" * int(acc / 5)
        print(f"  {label:<8}: {c:>3}/{n:<3}  ({acc:5.1f}%)  {bar}")
    print("================================")


if __name__ == "__main__":
    benchmark()

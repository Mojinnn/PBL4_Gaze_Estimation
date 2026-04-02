import os
import cv2
import time
import numpy as np
import onnxruntime as ort

INPUT_DIR  = "data_video"
OUTPUT_DIR = "results_video"

MODEL_CONFIGS = {
    "weights/mobileone_s0_gaze.onnx": {
        "input_size": (448, 448),
        "yaw_threshold": 3.0,
        "pitch_threshold": 10.0,
        "pitch_up_sign": +1,
        "yaw_sign": -1
    },
    "weights/mobilenetv2_gaze.onnx": {
        "input_size": (448, 448),
        "yaw_threshold": 20.0,
        "pitch_threshold": 8.0,
        "pitch_up_sign": -1,
        "yaw_sign": 1
    },
}

MODEL_PATH = "weights/mobileone_s0_gaze.onnx"


# ─── Model ───────────────────────────────────────────────────────────────────

class GazeModel:
    def __init__(self, model_path):
        cfg = MODEL_CONFIGS[model_path]
        self.input_size      = cfg["input_size"]
        self.yaw_threshold   = cfg["yaw_threshold"]
        self.pitch_threshold = cfg["pitch_threshold"]
        self.pitch_up_sign   = cfg["pitch_up_sign"]
        self.yaw_sign        = cfg.get("yaw_sign", 1)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            model_path, opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, img):
        x = cv2.resize(img, self.input_size)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std                          # giữ float32
        x = np.expand_dims(x.transpose(2, 0, 1), 0).astype(np.float32)

        t0      = time.time()
        out     = self.session.run(None, {self.input_name: x})
        elapsed = time.time() - t0

        def softmax_expectation(logits):
            e = np.exp(logits - np.max(logits))
            p = e / e.sum()
            return np.sum(p * np.arange(len(p))) * 4 - 180

        yaw   = softmax_expectation(out[0][0])
        pitch = softmax_expectation(out[1][0])
        return pitch, yaw, elapsed

    def get_direction(self, pitch, yaw):
        sp = pitch * self.pitch_up_sign
        sy = yaw   * self.yaw_sign
        if   sy >  self.yaw_threshold:   return "right"
        elif sy < -self.yaw_threshold:   return "left"
        elif sp >  self.pitch_threshold: return "up"
        elif sp < -self.pitch_threshold: return "down"
        return "center"


# ─── Face detector ───────────────────────────────────────────────────────────

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_face(frame):
    """Trả về (face_crop, (x1,y1,x2,y2)). Fallback toàn frame nếu không thấy mặt."""
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        h, w = frame.shape[:2]
        return frame, (0, 0, w, h)

    x, y, w, h = faces[0]
    margin = int(0.2 * w)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(frame.shape[1], x + w + margin)
    y2 = min(frame.shape[0], y + h + margin)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


# ─── Draw ────────────────────────────────────────────────────────────────────

COLOR_MAP = {
    "left":   (0, 180, 255),
    "right":  (0, 180, 255),
    "up":     (0, 220, 100),
    "down":   (0, 220, 100),
    "center": (200, 200, 200),
}

def draw_gaze(frame, face_box, pitch, yaw, direction):
    x1, y1, x2, y2 = face_box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    face_size = min(x2 - x1, y2 - y1)
    color = COLOR_MAP[direction]

    # Bounding box mặt
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    # Mũi tên hướng nhìn
    yaw_rad, pitch_rad = np.radians(yaw), np.radians(pitch)
    dx, dy = np.tan(yaw_rad), -np.tan(pitch_rad)
    norm = np.hypot(dx, dy)
    if norm > 0:
        dx /= norm
        dy /= norm
    length = int(face_size * 0.55)
    ep = (int(cx + dx * length), int(cy + dy * length))
    cv2.arrowedLine(frame, (cx, cy), ep, color, 2, tipLength=0.28)

    # Label phía trên box
    font  = cv2.FONT_HERSHEY_SIMPLEX
    label = direction.upper()
    (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
    tx, ty = cx - tw // 2, y1 - 8
    cv2.rectangle(frame, (tx - 4, ty - th - 2), (tx + tw + 4, ty + 4), color, -1)
    cv2.putText(frame, label, (tx, ty), font, 0.6, (0, 0, 0), 2)


# ─── Main ────────────────────────────────────────────────────────────────────

def run(video_path):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = GazeModel(MODEL_PATH)
    cap   = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Không mở được: {video_path}")
        return

    raw_count    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = raw_count if 0 < raw_count < 10_000_000 else None  # .h264 trả rác
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30
    W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_name = os.path.splitext(os.path.basename(video_path))[0] + "_gaze.mp4"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (W, H))

    frames_str = str(total_frames) if total_frames else "?"
    print(f"Input  : {video_path}  ({W}x{H}, {src_fps:.1f}fps, {frames_str} frames)")
    print(f"Output : {out_path}")
    print(f"Model  : {MODEL_PATH}\n")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1

        face_crop, face_box = detect_face(frame)
        pitch, yaw, t       = model.predict(face_crop)
        direction            = model.get_direction(pitch, yaw)

        draw_gaze(frame, face_box, pitch, yaw, direction)

        # HUD nhỏ góc trên trái
        hud = f"{idx}/{frames_str}  {1/t:.0f}fps  {direction}"
        cv2.putText(frame, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)

        writer.write(frame)

        if idx % 50 == 0 or idx == total_frames:
            print(f"  [{idx:>5}/{frames_str}]  {direction:<7}  yaw:{yaw:6.1f}  pitch:{pitch:6.1f}  {t*1000:.1f}ms")

    cap.release()
    writer.release()
    print(f"\nDone → {out_path}")


if __name__ == "__main__":
    # Tự động lấy file video đầu tiên trong data_video/
    exts = (".mp4", ".avi", ".mov", ".mkv", ".h264", ".h265")
    videos = [
        f for f in sorted(os.listdir(INPUT_DIR))
        if os.path.splitext(f)[1].lower() in exts
    ]

    if not videos:
        print(f"[ERROR] Không tìm thấy video nào trong {INPUT_DIR}/")
    else:
        run(os.path.join(INPUT_DIR, videos[0]))

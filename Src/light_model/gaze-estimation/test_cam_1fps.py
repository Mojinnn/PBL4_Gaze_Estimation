"""
Gaze calibration — 1 FPS, chỉ vẽ mũi tên + hiển thị yaw/pitch thô
===================================================================
Mục tiêu: xác định mũi tên có đúng hướng không, đọc giá trị yaw/pitch
thực tế để tune threshold sau.

Ctrl+C để dừng.
"""

import os
import cv2
import time
import subprocess
import threading
import numpy as np
import onnxruntime as ort

OUTPUT_DIR = "results_video"
MODEL_PATH = "weights/mobileone_s0_gaze.onnx"

CAPTURE_W   = 640
CAPTURE_H   = 480
CAPTURE_FPS = 1          # Chụp chậm để quan sát kỹ từng frame
DETECT_EVERY = 1
SAVE_EVERY   = 1

MODEL_CONFIGS = {
    "weights/mobileone_s0_gaze.onnx": {
        "input_size":    (448, 448),
        "pitch_up_sign": +1,
        "yaw_sign":      1,
    },
    "weights/mobilenetv2_gaze.onnx": {
        "input_size":    (448, 448),
        "pitch_up_sign": -1,
        "yaw_sign":      1,
    },
}


# ─── Camera ──────────────────────────────────────────────────────────────────
class CameraBuffer:
    def __init__(self, width, height, fps):
        self.width  = width
        self.height = height
        self._frame_bytes = width * height * 3 // 2

        cmd = [
            "rpicam-vid",
            "--width",     str(width),
            "--height",    str(height),
            "--framerate", str(fps),
            "--codec",     "yuv420",
            "--timeout",   "0",
            "--nopreview",
            "-o", "-",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self._frame_bytes * 4,
        )
        self._frame  = None
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        threading.Thread(target=self._reader, daemon=True).start()
        time.sleep(1.5)

    def _reader(self):
        while not self._stop.is_set():
            raw = self._proc.stdout.read(self._frame_bytes)
            if len(raw) < self._frame_bytes:
                self._stop.set()
                break
            yuv = np.frombuffer(raw, dtype=np.uint8).reshape(
                (self.height * 3 // 2, self.width)
            )
            with self._lock:
                self._frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def is_alive(self):
        return not self._stop.is_set()

    def release(self):
        self._stop.set()
        self._proc.terminate()
        self._proc.wait(timeout=3)


# ─── Model ───────────────────────────────────────────────────────────────────
class GazeModel:
    def __init__(self, model_path):
        cfg = MODEL_CONFIGS[model_path]
        self.input_size    = cfg["input_size"]
        self.pitch_up_sign = cfg["pitch_up_sign"]
        self.yaw_sign      = cfg["yaw_sign"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        self.session    = ort.InferenceSession(
            model_path, opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        print("=== MODEL OUTPUTS ===")
        for i, o in enumerate(self.session.get_outputs()):
            print(f"  [{i}] {o.name}  shape={o.shape}")
        print("=====================\n")

    def predict(self, img):
        x = cv2.resize(img, self.input_size)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = ((x - mean) / std).transpose(2, 0, 1)
        x = np.expand_dims(x, 0).astype(np.float32)

        t0  = time.time()
        out = self.session.run(None, {self.input_name: x})
        t   = time.time() - t0

        def decode(logits):
            e = np.exp(logits - np.max(logits))
            p = e / e.sum()
            # return float(np.sum(p * np.arange(len(p))) * 4 - 180)
            return np.sum(p * np.arange(len(p))) * 4 - 180

        yaw = decode(out[0][0])
        pitch = decode(out[1][0])
        
        # raw0 = decode(out[0])
        # raw1 = decode(out[1])

        # Heuristic: output nào có |value| lớn hơn thường là yaw
        # if abs(raw0) >= abs(raw1):
        #    yaw, pitch = raw0, raw1
        # else:
        #   yaw, pitch = raw1, raw0

        # Áp sign config
        # yaw   = yaw   * self.yaw_sign
        # pitch = pitch * self.pitch_up_sign

        return pitch, yaw, t


# ─── Face detector + CSRT tracker ────────────────────────────────────────────
class FaceDetector:
    CASCADE = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    _CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __init__(self, detect_every=10):
        self.detect_every  = detect_every
        self._tracker      = None
        self._box          = None
        self._frame_count  = 0
        self._tracking     = False
        self._miss_streak  = 0
        self.last_mode     = "—"

    def _detect(self, frame):
        gray  = self._CLAHE.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        faces = self.CASCADE.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50)
        )
        if len(faces) == 0:
            faces = self.CASCADE.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=2, minSize=(40, 40)
            )
        if len(faces) == 0:
            self._miss_streak += 1
            return None
        self._miss_streak = 0
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        m  = int(0.2 * w)
        x1 = max(0, x - m);                  y1 = max(0, y - m)
        x2 = min(frame.shape[1], x + w + m); y2 = min(frame.shape[0], y + h + m)
        return (x1, y1, x2, y2)

    def get(self, frame):
        self._frame_count += 1
        interval    = max(3, self.detect_every - self._miss_streak * 2)
        need_detect = self._frame_count % interval == 1 or not self._tracking

        if need_detect:
            self.last_mode = "DET"
            box = self._detect(frame)
            if box:
                self._box = box
                x1, y1, x2, y2 = box
                self._tracker = cv2.TrackerCSRT_create()
                self._tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                self._tracking = True
            else:
                self._tracking = False
        else:
            self.last_mode = "TRK"
            ok, rect = self._tracker.update(frame)
            if ok:
                rx, ry, rw, rh = [int(v) for v in rect]
                self._box = (rx, ry, rx + rw, ry + rh)
            else:
                self._tracking = False

        if self._box is None:
            h, w = frame.shape[:2]
            return frame, (0, 0, w, h)

        x1, y1, x2, y2 = self._box
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


# ─── Draw: chỉ mũi tên + số liệu thô ────────────────────────────────────────
def draw(frame, face_box, pitch, yaw, infer_t, mode, miss, idx):
    x1, y1, x2, y2 = face_box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    sz = min(x2 - x1, y2 - y1)

    # Bounding box mặt — xanh lá
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 80), 1)

    # Mũi tên hướng nhìn — trắng
    dx = -np.tan(np.radians(yaw))
    dy = -np.tan(np.radians(pitch))
    n  = np.hypot(dx, dy)
    if n > 0:
        dx /= n; dy /= n
    ep = (int(cx + dx * sz * 0.6), int(cy + dy * sz * 0.6))
    cv2.arrowedLine(frame, (cx, cy), ep, (255, 255, 255), 2, tipLength=0.3)

    # Panel thông số góc trái — dễ đọc
    font  = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        (f"frame   : {idx}",                        (200, 200, 200)),
        (f"yaw     : {yaw:+.1f} deg",               (100, 200, 255)),
        (f"pitch   : {pitch:+.1f} deg",             (100, 255, 180)),
        # (f"raw[0]  : {raw0:+.1f}",                  (160, 160, 160)),
        # (f"raw[1]  : {raw1:+.1f}",                  (160, 160, 160)),
        (f"infer   : {infer_t*1000:.0f} ms",        (160, 160, 160)),
        (f"mode    : {mode}",                        (200, 200, 100)),
        (f"miss    : {miss}",  (80, 80, 255) if miss > 2 else (160, 160, 160)),
    ]
    for i, (text, color) in enumerate(lines):
        y = 24 + i * 22
        cv2.putText(frame, text, (8, y), font, 0.55, color, 1, cv2.LINE_AA)

    # Timestamp góc phải trên
    ts = time.strftime("%H:%M:%S")
    cv2.putText(frame, ts, (CAPTURE_W - 80, 20), font, 0.5, (100, 100, 100), 1)


# ─── Main ─────────────────────────────────────────────────────────────────────
def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session_dir = os.path.join(OUTPUT_DIR, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(session_dir, exist_ok=True)

    model    = GazeModel(MODEL_PATH)
    detector = FaceDetector(detect_every=DETECT_EVERY)
    cam      = CameraBuffer(CAPTURE_W, CAPTURE_H, CAPTURE_FPS)

    print(f"Camera  : {CAPTURE_W}x{CAPTURE_H} @ {CAPTURE_FPS} FPS")
    print(f"Output  : {session_dir}/")
    print(f"Model   : {MODEL_PATH}")
    print("Ctrl+C để dừng\n")

    idx   = 0
    saved = 0

    try:
        while cam.is_alive():
            ok, frame = cam.read()
            if not ok:
                time.sleep(0.02)
                continue

            idx += 1

            face_crop, face_box = detector.get(frame)

            if face_crop.size == 0:
                pitch = yaw = infer_t = 0.0
            else:
                pitch, yaw, infer_t = model.predict(face_crop)

            vis = frame.copy()
            draw(vis, face_box, pitch, yaw, infer_t, detector.last_mode, detector._miss_streak, idx)

            fname = os.path.join(session_dir, f"{idx:06d}.jpg")
            cv2.imwrite(fname, vis)
            saved += 1

            # Log ra terminal mỗi frame (1 FPS nên không nhiều)
            print(f"[{idx:04d}]  yaw={yaw:+6.1f}  pitch={pitch:+6.1f}"
                  f"  {infer_t*1000:.0f}ms  {detector.last_mode}")

    except KeyboardInterrupt:
        pass

    cam.release()
    print(f"\nDone. {saved} frames → {session_dir}/")


if __name__ == "__main__":
    run()

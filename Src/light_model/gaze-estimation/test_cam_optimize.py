"""
Gaze estimation real-time từ Pi Camera (không dùng picamera2)
=============================================================
Dùng rpicam-vid stream raw BGR qua stdout pipe → OpenCV đọc trực tiếp.
Không cần ghi file tạm, không cần picamera2.

Chạy:
    python run_camera_gaze.py

Nhấn  Ctrl+C  để dừng.
"""

import os
import cv2
import time
import subprocess
import threading
import numpy as np
import onnxruntime as ort
from collections import deque

# ─── Paths ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = "results_video"
MODEL_PATH = "weights/mobileone_s0_gaze.onnx"

# ─── Params ──────────────────────────────────────────────────────────────────
CAPTURE_W    = 640
CAPTURE_H    = 480
CAPTURE_FPS  = 30
DETECT_EVERY = 10
VOTE_WINDOW  = 8
VOTE_THRESH  = 0.60
SAVE_EVERY   = 1        # Lưu mỗi N frame (1 = lưu tất cả)

MODEL_CONFIGS = {
    "weights/mobileone_s0_gaze.onnx": {
        "input_size":      (448, 448),
        "yaw_threshold":   3.0,
        "pitch_threshold": 10.0,
        "pitch_up_sign":   +1,
        "yaw_sign":        1,
    },
    "weights/mobilenetv2_gaze.onnx": {
        "input_size":      (448, 448),
        "yaw_threshold":   20.0,
        "pitch_threshold": 8.0,
        "pitch_up_sign":   -1,
        "yaw_sign":        1,
    },
}

COLOR_MAP = {
    "left":   (0, 180, 255),
    "right":  (0, 180, 255),
    "up":     (0, 220, 100),
    "down":   (0, 220, 100),
    "center": (200, 200, 200),
}


# ─── Camera buffer dùng rpicam-vid pipe ──────────────────────────────────────
class CameraBuffer:
    """
    Spawn rpicam-vid với codec yuv420, đọc raw bytes từ stdout.
    Thread riêng đọc liên tục và giữ frame MỚI NHẤT.
    """

    def __init__(self, width=640, height=480, fps=30):
        self.width  = width
        self.height = height
        # Số byte 1 frame YUV420: W*H*3//2
        self._frame_bytes = width * height * 3 // 2

        cmd = [
            "rpicam-vid",
            "--width",   str(width),
            "--height",  str(height),
            "--framerate", str(fps),
            "--codec",   "yuv420",   # raw YUV, không encode H264
            "--timeout", "0",        # chạy mãi
            "--nopreview",
            "-o", "-",               # output ra stdout
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
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        time.sleep(1.0)   # Chờ camera khởi động

    def _reader(self):
        while not self._stop.is_set():
            raw = self._proc.stdout.read(self._frame_bytes)
            if len(raw) < self._frame_bytes:
                self._stop.set()
                break
            yuv = np.frombuffer(raw, dtype=np.uint8).reshape(
                (self.height * 3 // 2, self.width)
            )
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            with self._lock:
                self._frame = bgr

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
        self._thread.join(timeout=2)


# ─── Gaze model ──────────────────────────────────────────────────────────────
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
        self.session    = ort.InferenceSession(
            model_path, opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, img):
        x    = cv2.resize(img, self.input_size)
        x    = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x    = ((x - mean) / std).transpose(2, 0, 1)
        x    = np.expand_dims(x, 0).astype(np.float32)

        t0  = time.time()
        out = self.session.run(None, {self.input_name: x})
        t   = time.time() - t0

        def soft_exp(logits):
            e = np.exp(logits - np.max(logits))
            p = e / e.sum()
            return np.sum(p * np.arange(len(p))) * 4 - 180

        return soft_exp(out[1][0]), soft_exp(out[0][0]), t

    def direction(self, pitch, yaw):
        if   yaw   * self.yaw_sign      >  self.yaw_threshold:   return "right"
        elif yaw   * self.yaw_sign      < -self.yaw_threshold:   return "left"
        elif pitch * self.pitch_up_sign >  self.pitch_threshold:  return "up"
        elif pitch * self.pitch_up_sign < -self.pitch_threshold:  return "down"
        return "center"


# ─── Face detector + CSRT tracker ────────────────────────────────────────────
class FaceDetector:
    CASCADE = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    def __init__(self, detect_every=10):
        self.detect_every = detect_every
        self._tracker     = None
        self._box         = None
        self._frame_count = 0
        self._tracking    = False

    def _detect(self, frame):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.CASCADE.detectMultiScale(gray, 1.3, 5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]
        m  = int(0.2 * w)
        x1 = max(0, x - m)
        y1 = max(0, y - m)
        x2 = min(frame.shape[1], x + w + m)
        y2 = min(frame.shape[0], y + h + m)
        return (x1, y1, x2, y2)

    def _init_tracker(self, frame, box):
        x1, y1, x2, y2 = box
        self._tracker   = cv2.TrackerCSRT_create()
        self._tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
        self._tracking  = True

    def get(self, frame):
        self._frame_count += 1
        need_detect = (
            self._frame_count % self.detect_every == 1
            or not self._tracking
        )
        if need_detect:
            box = self._detect(frame)
            if box:
                self._box = box
                self._init_tracker(frame, box)
            else:
                self._tracking = False
        else:
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
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


# ─── Voting window ────────────────────────────────────────────────────────────
class VotingWindow:
    LABELS = ["left", "right", "up", "down", "center"]

    def __init__(self, window=8, thresh=0.60):
        self.window  = window
        self.thresh  = thresh
        self._buffer = deque(maxlen=window)

    def push(self, direction):
        self._buffer.append(direction)

    def decision(self):
        if len(self._buffer) < self.window:
            return None
        counts = {d: self._buffer.count(d) for d in self.LABELS}
        best   = max(counts, key=counts.get)
        return best if counts[best] / self.window >= self.thresh else None


# ─── Draw ─────────────────────────────────────────────────────────────────────
def draw_gaze(frame, face_box, pitch, yaw, raw_dir, decision):
    x1, y1, x2, y2 = face_box
    cx  = (x1 + x2) // 2
    cy  = (y1 + y2) // 2
    sz  = min(x2 - x1, y2 - y1)
    col = COLOR_MAP.get(raw_dir, (200, 200, 200))

    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 1)

    dx = np.tan(np.radians(yaw))
    dy = -np.tan(np.radians(pitch))
    n  = np.hypot(dx, dy)
    if n > 0:
        dx /= n; dy /= n
    ep = (int(cx + dx * sz * 0.55), int(cy + dy * sz * 0.55))
    cv2.arrowedLine(frame, (cx, cy), ep, col, 2, tipLength=0.28)

    font = cv2.FONT_HERSHEY_SIMPLEX
    lbl  = raw_dir.upper()
    (tw, th), _ = cv2.getTextSize(lbl, font, 0.55, 2)
    tx, ty = cx - tw // 2, y1 - 8
    cv2.rectangle(frame, (tx - 3, ty - th - 2), (tx + tw + 3, ty + 4), col, -1)
    cv2.putText(frame, lbl, (tx, ty), font, 0.55, (0, 0, 0), 2)

    if decision:
        msg = f">> {decision.upper()} <<"
        (dw, dh), _ = cv2.getTextSize(msg, font, 1.0, 3)
        bx = (frame.shape[1] - dw) // 2
        cv2.rectangle(frame, (bx - 8, 4), (bx + dw + 8, dh + 16), (0, 0, 0), -1)
        cv2.putText(frame, msg, (bx, dh + 8), font, 1.0, (0, 255, 128), 3)


# ─── Main ─────────────────────────────────────────────────────────────────────
def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session_dir = os.path.join(OUTPUT_DIR, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(session_dir, exist_ok=True)

    model    = GazeModel(MODEL_PATH)
    detector = FaceDetector(detect_every=DETECT_EVERY)
    voter    = VotingWindow(window=VOTE_WINDOW, thresh=VOTE_THRESH)
    cam      = CameraBuffer(width=CAPTURE_W, height=CAPTURE_H, fps=CAPTURE_FPS)

    print(f"Camera : {CAPTURE_W}x{CAPTURE_H} @ {CAPTURE_FPS}fps  (rpicam-vid pipe)")
    print(f"Output : {session_dir}/")
    print(f"Config : detect_every={DETECT_EVERY}  vote_window={VOTE_WINDOW}"
          f"  vote_thresh={VOTE_THRESH}")
    print("Nhấn  Ctrl+C  để dừng\n")

    idx           = 0
    saved         = 0
    fps_smooth    = 0.0
    t_prev        = time.time()
    last_decision = None

    try:
        while cam.is_alive():
            ok, frame = cam.read()
            if not ok:
                time.sleep(0.005)
                continue

            idx += 1

            face_crop, face_box = detector.get(frame)

            if face_crop.size == 0:
                pitch, yaw, infer_t = 0.0, 0.0, 0.0
                raw_dir = "center"
            else:
                pitch, yaw, infer_t = model.predict(face_crop)
                raw_dir             = model.direction(pitch, yaw)

            voter.push(raw_dir)
            decision = voter.decision()
            if decision:
                last_decision = decision

            vis = frame.copy()
            draw_gaze(vis, face_box, pitch, yaw, raw_dir, last_decision)

            now        = time.time()
            fps_smooth = 0.85 * fps_smooth + 0.15 / max(now - t_prev, 1e-6)
            t_prev     = now
            mode       = "DET" if idx % DETECT_EVERY == 1 else "TRK"
            cv2.putText(vis,
                        f"{mode}  {fps_smooth:.1f}fps  {infer_t*1000:.0f}ms",
                        (CAPTURE_W - 175, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            if idx % SAVE_EVERY == 0:
                cv2.imwrite(os.path.join(session_dir, f"{idx:06d}.jpg"), vis)
                saved += 1

            if idx % 60 == 0:
                print(f"  [{idx:>6}]  raw:{raw_dir:<7}  cmd:{str(last_decision):<7}"
                      f"  {fps_smooth:.1f}fps  saved:{saved}")

    except KeyboardInterrupt:
        pass

    cam.release()
    print(f"\nDone. Saved {saved} frames → {session_dir}/")


if __name__ == "__main__":
    run()

"""
gaze_realtime.py
────────────────
Real-time gaze estimation cho Raspberry Pi.
Dùng model L2CS-Net vừa train (mobileone_s0_gaze.onnx).

Pipeline:
  rpicam-vid (YUV420 pipe)
      → CameraBuffer (thread riêng)
      → FaceDetector (Haar + CSRT tracker)
      → GazeModel    (ONNX, L2CS decode)
      → VotingWindow (làm mượt quyết định)
      → Lưu frame annotated ra session folder

Cách dùng:
  python3 gaze_realtime.py

Nhấn Ctrl+C để dừng.
"""

import os
import cv2
import math
import time
import subprocess
import threading
import numpy as np
import onnxruntime as ort
from collections import deque

# ══════════════════════════════════════════════════════════════════════════════
#  CẤU HÌNH — chỉnh tại đây
# ══════════════════════════════════════════════════════════════════════════════
OUTPUT_DIR   = "results_video"
MODEL_PATH   = "weights/mobileone_s0_gaze_re.onnx"

# Camera
CAPTURE_W    = 640
CAPTURE_H    = 480
CAPTURE_FPS  = 30

# Detect & tracking
DETECT_EVERY = 10     # detect lại mỗi N frame
FACE_PAD     = 0.15   # padding quanh bbox mặt (tỷ lệ)

# Voting (làm mượt nhãn)
VOTE_WINDOW  = 8      # số frame giữ trong buffer
VOTE_THRESH  = 0.60   # tỷ lệ đa số tối thiểu để ra quyết định

# Lưu ảnh
SAVE_EVERY   = 1      # lưu 1 trong N frame (1 = lưu tất cả)

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL CONFIG — thresholds + quy ước sign từ notebook training
#
#  Lấy pitch_up_sign và yaw_right_sign từ Cell 11 notebook sau khi train:
#
#  pitch_up_sign = +1  → pitch dương = nhìn lên   (MobileOne convention)
#  pitch_up_sign = -1  → pitch âm   = nhìn lên   (MobileNetV2 convention)
#
#  yaw_right_sign = -1 → yaw dương  = nhìn trái  (L2CS convention)
#  yaw_right_sign = +1 → yaw dương  = nhìn phải
# ══════════════════════════════════════════════════════════════════════════════
MODEL_CONFIGS = {
    "weights/mobileone_s0_gaze_re.onnx": {
        "input_size"     : (448, 448),   # kích thước input model đã train
        "num_bins"       : 90,           # số bins L2CS (phải khớp với CFG['num_bins'] lúc train)
        "angle_range"    : 360,          # tổng range góc (phải khớp CFG['angle_range'])
        "yaw_threshold"  : 20.0,         # ° — tune bằng +/- nếu cần
        "pitch_threshold": 6.0,          # ° — tune bằng [/] nếu cần
        "pitch_up_sign"  : +1,           # ← lấy từ Cell 11 notebook
        "yaw_right_sign" : -1,           # ← lấy từ Cell 11 notebook
    },
}

# Màu BGR theo hướng
COLOR_MAP = {
    "left"  : (50,  130, 255),   # cam
    "right" : (255, 180,  50),   # xanh dương
    "up"    : (50,  220,  80),   # xanh lá
    "down"  : (60,   60, 230),   # đỏ
    "center": (200, 200, 200),   # xám
}


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA BUFFER — đọc rpicam-vid qua pipe (giống code gốc của bạn)
# ══════════════════════════════════════════════════════════════════════════════
class CameraBuffer:
    """
    Spawn rpicam-vid với codec yuv420, đọc raw bytes từ stdout.
    Thread riêng đọc liên tục và giữ frame MỚI NHẤT.
    """

    def __init__(self, width=640, height=480, fps=30):
        self.width        = width
        self.height       = height
        self._frame_bytes = width * height * 3 // 2   # YUV420

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
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        time.sleep(1.0)   # chờ camera khởi động

    def _reader(self):
        while not self._stop.is_set():
            raw = self._proc.stdout.read(self._frame_bytes)
            if len(raw) < self._frame_bytes:
                self._stop.set()
                break
            yuv = np.frombuffer(raw, dtype=np.uint8).reshape(
                (self.height * 3 // 2, self.width))
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


# ══════════════════════════════════════════════════════════════════════════════
#  GAZE MODEL — L2CS decode đúng chuẩn với model vừa train
# ══════════════════════════════════════════════════════════════════════════════
class GazeModel:
    """
    Wrapper ONNX cho L2CS-Net MobileOne-S0.

    Decode:
      output[0] = yaw_logits  (1, num_bins)
      output[1] = pitch_logits (1, num_bins)
      → softmax → expectation → góc độ
    """

    def __init__(self, model_path):
        if model_path not in MODEL_CONFIGS:
            raise ValueError(
                f"Model '{model_path}' không có trong MODEL_CONFIGS.\n"
                f"Các model hỗ trợ: {list(MODEL_CONFIGS.keys())}")

        cfg = MODEL_CONFIGS[model_path]
        self.input_size      = cfg["input_size"]
        self.num_bins        = cfg["num_bins"]
        self.angle_range     = cfg["angle_range"]
        self.yaw_threshold   = cfg["yaw_threshold"]
        self.pitch_threshold = cfg["pitch_threshold"]
        self.pitch_up_sign   = cfg["pitch_up_sign"]
        self.yaw_right_sign  = cfg["yaw_right_sign"]

        # Tính sẵn bin_centers để tránh tính lại mỗi frame
        bin_size         = self.angle_range / self.num_bins
        self._bin_centers = (np.arange(self.num_bins) * bin_size
                             - self.angle_range / 2
                             + bin_size / 2).astype(np.float32)

        # ONNX session — tối ưu cho RPi
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 2   # RPi 4: 2 thread tối ưu nhất
        self.session    = ort.InferenceSession(
            model_path, opts,
            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        # In thông tin model để debug
        print("\n=== MODEL INFO ===")
        inp = self.session.get_inputs()[0]
        print(f"  Input : {inp.name}  shape={inp.shape}")
        for i, o in enumerate(self.session.get_outputs()):
            print(f"  Output[{i}]: {o.name}  shape={o.shape}")
        print(f"  num_bins={self.num_bins}  angle_range={self.angle_range}")
        print(f"  pitch_up_sign={self.pitch_up_sign:+d}  "
              f"yaw_right_sign={self.yaw_right_sign:+d}")
        print("==================\n")

        # Warmup
        dummy = np.zeros((1, 3, *self.input_size), dtype=np.float32)
        self.session.run(None, {self.input_name: dummy})
        print("Warmup done.\n")

    # ── Preprocess ────────────────────────────────────────────────────────
    def _preprocess(self, bgr_img):
        """BGR → normalized tensor (1, 3, H, W) khớp với lúc train."""
        img = cv2.resize(bgr_img, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) \
              / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return img.transpose(2, 0, 1)[None]   # (1, 3, H, W)

    # ── Softmax expectation ───────────────────────────────────────────────
    def _decode(self, logits):
        """Logits → góc (°) bằng softmax expectation (đúng theo L2CS-Net)."""
        logits = logits[0]                        # bỏ batch dim
        e      = np.exp(logits - logits.max())
        prob   = e / e.sum()
        return float(np.dot(prob, self._bin_centers))

    # ── Predict ───────────────────────────────────────────────────────────
    def predict(self, face_crop):
        """
        Nhận ảnh mặt crop (BGR), trả về (yaw°, pitch°, infer_time).
        output[0] = yaw_logits, output[1] = pitch_logits
        (đúng thứ tự export từ notebook Cell 11)
        """
        x  = self._preprocess(face_crop)
        t0 = time.time()
        outputs = self.session.run(None, {self.input_name: x})
        infer_t = time.time() - t0

        yaw   = self._decode(outputs[0])   # output[0] = yaw
        pitch = self._decode(outputs[1])   # output[1] = pitch
        return yaw, pitch, infer_t

    # ── Classify ──────────────────────────────────────────────────────────
    def classify(self, yaw, pitch):
        """
        Góc → nhãn hướng nhìn.
        Dùng pitch_up_sign và yaw_right_sign để chuẩn hóa chiều.
        """
        sp = pitch * self.pitch_up_sign    # sp > 0 = nhìn lên
        sy = yaw   * self.yaw_right_sign   # sy > 0 = nhìn phải

        if sp >  self.pitch_threshold: return "up"
        if sp < -self.pitch_threshold: return "down"
        if sy >  self.yaw_threshold:   return "right"
        if sy < -self.yaw_threshold:   return "left"
        return "center"


# ══════════════════════════════════════════════════════════════════════════════
#  FACE DETECTOR + CSRT TRACKER (giữ nguyên từ code gốc)
# ══════════════════════════════════════════════════════════════════════════════
class FaceDetector:
    _CASCADE = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    _CLAHE   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __init__(self, detect_every=10, face_pad=0.15):
        self.detect_every = detect_every
        self.face_pad     = face_pad
        self._tracker     = None
        self._box         = None   # (x1,y1,x2,y2) trong frame gốc
        self._frame_count = 0
        self._tracking    = False
        self._miss_streak = 0

    def _preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self._CLAHE.apply(gray)

    def _detect(self, frame):
        gray = self._preprocess(frame)

        # Lần 1: tham số chặt
        faces = self._CASCADE.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=4, minSize=(50, 50))

        # Lần 2: nới lỏng nếu thiếu sáng / góc nghiêng
        if len(faces) == 0:
            faces = self._CASCADE.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=2, minSize=(40, 40))

        if len(faces) == 0:
            self._miss_streak += 1
            return None

        self._miss_streak = 0
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])   # mặt lớn nhất

        # Thêm padding
        m  = int(w * self.face_pad)
        x1 = max(0, x - m)
        y1 = max(0, y - m)
        x2 = min(frame.shape[1], x + w + m)
        y2 = min(frame.shape[0], y + h + m)
        return (x1, y1, x2, y2)

    def _init_tracker(self, frame, box):
        x1, y1, x2, y2 = box
        self._tracker  = cv2.TrackerCSRT_create()
        self._tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
        self._tracking = True

    def get(self, frame):
        """
        Trả về (face_crop, (x1,y1,x2,y2)).
        Dùng tracker giữa các lần detect để tăng FPS.
        """
        self._frame_count += 1

        # Detect sớm hơn nếu đang miss liên tục
        effective_interval = max(3, self.detect_every - self._miss_streak * 2)
        need_detect = (
            self._frame_count % effective_interval == 1
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
                self._tracking = False   # mất track → detect ngay lần sau

        if self._box is None:
            h, w = frame.shape[:2]
            return frame, (0, 0, w, h)

        x1, y1, x2, y2 = self._box
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        return crop, (x1, y1, x2, y2)


# ══════════════════════════════════════════════════════════════════════════════
#  VOTING WINDOW — làm mượt quyết định cuối
# ══════════════════════════════════════════════════════════════════════════════
class VotingWindow:
    LABELS = ["left", "right", "up", "down", "center"]

    def __init__(self, window=8, thresh=0.60):
        self.window  = window
        self.thresh  = thresh
        self._buffer = deque(maxlen=window)

    def push(self, direction):
        self._buffer.append(direction)

    def decision(self):
        """Trả về nhãn nếu vượt ngưỡng, None nếu chưa đủ đa số."""
        if len(self._buffer) < self.window:
            return None
        counts = {d: self._buffer.count(d) for d in self.LABELS}
        best   = max(counts, key=counts.get)
        return best if counts[best] / self.window >= self.thresh else None


# ══════════════════════════════════════════════════════════════════════════════
#  DRAW
# ══════════════════════════════════════════════════════════════════════════════
def draw_gaze_arrow(frame, face_box, yaw_deg, pitch_deg, label):
    """Vẽ mũi tên tại trung tâm mặt theo hướng yaw/pitch thực."""
    x1, y1, x2, y2 = face_box
    cx  = (x1 + x2) // 2
    cy  = (y1 + y2) // 2
    sz  = min(x2 - x1, y2 - y1)
    col = COLOR_MAP.get(label, (200, 200, 200))
    thick = max(2, sz // 60)

    # Vector 2D từ yaw/pitch
    dx =  math.sin(math.radians(yaw_deg)) * math.cos(math.radians(pitch_deg))
    dy = -math.sin(math.radians(pitch_deg))
    n  = math.hypot(dx, dy)
    if n > 1e-6:
        dx /= n; dy /= n

    arrow_len = int(sz * 0.50)
    ex = int(cx + dx * arrow_len)
    ey = int(cy + dy * arrow_len)

    # Glow
    cv2.arrowedLine(frame, (cx, cy), (ex, ey),
                    (255, 255, 255), thick + 2, tipLength=0.28)
    # Mũi tên chính
    cv2.arrowedLine(frame, (cx, cy), (ex, ey),
                    col, thick, tipLength=0.28)
    # Điểm gốc
    cv2.circle(frame, (cx, cy), thick + 2, (255, 255, 255), -1)
    cv2.circle(frame, (cx, cy), thick,     col,             -1)


def draw_face_box(frame, face_box, label, yaw, pitch):
    """Bbox + label badge + góc yaw/pitch."""
    x1, y1, x2, y2 = face_box
    col   = COLOR_MAP.get(label, (200, 200, 200))
    thick = max(1, (x2 - x1) // 120)

    cv2.rectangle(frame, (x1, y1), (x2, y2), col, thick)

    # Label badge phía trên bbox
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, min(0.70, (x2 - x1) / 220))
    text  = label.upper()
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    bx = x1
    by = y1 - th - 8 if y1 - th - 8 > 0 else y2 + 4
    cv2.rectangle(frame, (bx - 2, by - 2), (bx + tw + 4, by + th + 4), col, -1)
    cv2.putText(frame, text, (bx + 1, by + th + 1),
                font, scale, (0, 0, 0), 1, cv2.LINE_AA)

    # Góc nhỏ phía dưới bbox
    angle_text = f"y:{yaw:+.0f} p:{pitch:+.0f}"
    ascale = max(0.35, scale * 0.65)
    (atw, ath), _ = cv2.getTextSize(angle_text, font, ascale, 1)
    ax = x2 - atw - 4
    ay = y2 + ath + 4
    if ay > frame.shape[0] - 2:
        ay = y1 - 4
    cv2.putText(frame, angle_text, (ax + 1, ay + 1),
                font, ascale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, angle_text, (ax, ay),
                font, ascale, col, 1, cv2.LINE_AA)


def draw_decision_banner(frame, decision):
    """Banner to ở giữa trên khi có quyết định voting."""
    font = cv2.FONT_HERSHEY_DUPLEX
    msg  = f">> {decision.upper()} <<"
    col  = COLOR_MAP.get(decision, (0, 255, 128))
    (dw, dh), _ = cv2.getTextSize(msg, font, 1.1, 2)
    bx = (frame.shape[1] - dw) // 2
    cv2.rectangle(frame, (bx - 10, 2), (bx + dw + 10, dh + 18),
                  (0, 0, 0), -1)
    cv2.putText(frame, msg, (bx, dh + 10),
                font, 1.1, col, 2, cv2.LINE_AA)


def draw_hud(frame, fps, infer_ms, mode, miss_streak):
    """HUD góc trên phải: FPS, inference time, mode."""
    lines = [
        f"{mode}  {fps:.1f} fps",
        f"infer: {infer_ms:.0f} ms",
    ]
    if miss_streak > 0:
        lines.append(f"miss: {miss_streak}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    x0   = frame.shape[1] - 160
    for i, line in enumerate(lines):
        col = (80, 80, 255) if "miss" in line else (180, 180, 180)
        cv2.putText(frame, line, (x0, 18 + i * 20),
                    font, 0.48, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (x0, 18 + i * 20),
                    font, 0.48, col, 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session_dir = os.path.join(OUTPUT_DIR, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(session_dir, exist_ok=True)

    # Khởi tạo các thành phần
    model    = GazeModel(MODEL_PATH)
    detector = FaceDetector(detect_every=DETECT_EVERY, face_pad=FACE_PAD)
    voter    = VotingWindow(window=VOTE_WINDOW, thresh=VOTE_THRESH)
    cam      = CameraBuffer(width=CAPTURE_W, height=CAPTURE_H, fps=CAPTURE_FPS)

    print(f"Camera  : {CAPTURE_W}×{CAPTURE_H} @ {CAPTURE_FPS}fps")
    print(f"Output  : {session_dir}/")
    print(f"Config  : detect_every={DETECT_EVERY}  "
          f"vote_window={VOTE_WINDOW}  vote_thresh={VOTE_THRESH}")
    print("Nhấn Ctrl+C để dừng\n")

    idx           = 0
    saved         = 0
    fps_smooth    = 0.0
    t_prev        = time.time()
    last_decision = None
    last_infer_ms = 0.0

    try:
        while cam.is_alive():
            ok, frame = cam.read()
            if not ok:
                time.sleep(0.005)
                continue

            idx += 1

            # ── Detect mặt ────────────────────────────────────────────────
            face_crop, face_box = detector.get(frame)

            # ── Predict ───────────────────────────────────────────────────
            if face_crop.size > 0:
                yaw, pitch, infer_t  = model.predict(face_crop)
                last_infer_ms        = infer_t * 1000
                raw_label            = model.classify(yaw, pitch)
            else:
                yaw, pitch, raw_label = 0.0, 0.0, "center"

            # ── Voting ────────────────────────────────────────────────────
            voter.push(raw_label)
            decision = voter.decision()
            if decision:
                last_decision = decision

            # ── Vẽ annotation ─────────────────────────────────────────────
            vis  = frame.copy()
            mode = "DET" if (not detector._tracking
                             or idx % DETECT_EVERY == 1) else "TRK"

            # FPS
            now        = time.time()
            fps_smooth = 0.85 * fps_smooth + 0.15 / max(now - t_prev, 1e-6)
            t_prev     = now

            draw_face_box(vis, face_box, raw_label, yaw, pitch)
            draw_gaze_arrow(vis, face_box, yaw, pitch, raw_label)
            if last_decision:
                draw_decision_banner(vis, last_decision)
            draw_hud(vis, fps_smooth, last_infer_ms, mode,
                     detector._miss_streak)

            # ── Lưu frame ─────────────────────────────────────────────────
            if idx % SAVE_EVERY == 0:
                path = os.path.join(session_dir, f"{idx:06d}.jpg")
                cv2.imwrite(path, vis)
                saved += 1

            # ── Log terminal mỗi 60 frame ─────────────────────────────────
            if idx % 60 == 0:
                print(f"  [{idx:>6}]  raw:{raw_label:<7}  "
                      f"decision:{str(last_decision):<7}  "
                      f"{fps_smooth:.1f}fps  "
                      f"infer:{last_infer_ms:.0f}ms  "
                      f"saved:{saved}")

    except KeyboardInterrupt:
        pass

    cam.release()
    print(f"\nDone. {idx} frames processed, {saved} saved → {session_dir}/")


if __name__ == "__main__":
    run()

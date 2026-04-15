"""
Gaze calibration + GPIO LED
============================
yaw trong [-6, +6]  → LED giữa (left + right tắt)
yaw < -6            → LED trái
yaw > +6            → LED phải

Ctrl+C để dừng.
"""

import os
import cv2
import time
import subprocess
import threading
import numpy as np
import onnxruntime as ort
import RPi.GPIO as GPIO

OUTPUT_DIR = "results_video"
MODEL_PATH = "weights/mobileone_s0_gaze.onnx"

CAPTURE_W    = 640
CAPTURE_H    = 480
CAPTURE_FPS  = 1
DETECT_EVERY = 1
SAVE_EVERY   = 1

YAW_CENTER_MIN = -6.0   # Ngưỡng trái
YAW_CENTER_MAX = +6.0   # Ngưỡng phải

LED_PINS = {
    "left":  17,   # GPIO 17 - Pin 11
    "right": 27,   # GPIO 27 - Pin 13
    "center":    22,   # GPIO 22 - Pin 15
    "down":  23,   # GPIO 23 - Pin 16
}

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


# ─── GPIO setup ──────────────────────────────────────────────────────────────
def gpio_setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in LED_PINS.values():
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)

def gpio_cleanup():
    for pin in LED_PINS.values():
        GPIO.output(pin, GPIO.LOW)
    GPIO.cleanup()

def set_leds(state):
    """
    state: "left" | "center" | "right"
    center → tắt left + right (up/down không dùng ở bước này)
    """
    GPIO.output(LED_PINS["left"],  GPIO.HIGH if state == "left"  else GPIO.LOW)
    GPIO.output(LED_PINS["right"], GPIO.HIGH if state == "right" else GPIO.LOW)
    GPIO.output(LED_PINS["center"], GPIO.HIGH if state == "center" else GPIO.LOW)

def yaw_to_state(yaw):
    if yaw < YAW_CENTER_MIN:
        return "left"
    elif yaw > YAW_CENTER_MAX:
        return "right"
    return "center"


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
        self._frame = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
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
            return np.sum(p * np.arange(len(p))) * 4 - 180

        yaw   = decode(out[0][0])
        pitch = decode(out[1][0])
        return pitch, yaw, t


# ─── Face detector ───────────────────────────────────────────────────────────
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


# ─── Draw ─────────────────────────────────────────────────────────────────────
LED_COLOR = {
    "left":   (0, 180, 255),   # cam
    "center": (0, 220, 80),    # xanh lá
    "right":  (0, 180, 255),   # cam
}

def draw(frame, face_box, pitch, yaw, infer_t, mode, miss, idx, led_state):
    x1, y1, x2, y2 = face_box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    sz = min(x2 - x1, y2 - y1)

    col = LED_COLOR[led_state]
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)

    # Mũi tên hướng nhìn
    dx = -np.tan(np.radians(yaw))
    dy = -np.tan(np.radians(pitch))
    n  = np.hypot(dx, dy)
    if n > 0:
        dx /= n; dy /= n
    ep = (int(cx + dx * sz * 0.6), int(cy + dy * sz * 0.6))
    cv2.arrowedLine(frame, (cx, cy), ep, (255, 255, 255), 2, tipLength=0.3)

    # Thông số
    font  = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        (f"frame  : {idx}",                   (200, 200, 200)),
        (f"yaw    : {yaw:+.1f} deg",          (100, 200, 255)),
        (f"pitch  : {pitch:+.1f} deg",        (100, 255, 180)),
        (f"infer  : {infer_t*1000:.0f} ms",   (160, 160, 160)),
        (f"mode   : {mode}",                  (200, 200, 100)),
        (f"miss   : {miss}",   (80, 80, 255) if miss > 2 else (160, 160, 160)),
    ]
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (8, 24 + i * 22),
                    font, 0.55, color, 1, cv2.LINE_AA)

    # LED state banner — lớn, dễ thấy
    state_text  = led_state.upper()
    state_color = LED_COLOR[led_state]
    (sw, sh), _ = cv2.getTextSize(state_text, font, 1.4, 3)
    bx = (CAPTURE_W - sw) // 2
    by = CAPTURE_H - 20
    cv2.rectangle(frame,
                  (bx - 10, by - sh - 8),
                  (bx + sw + 10, by + 8),
                  (0, 0, 0), -1)
    cv2.putText(frame, state_text, (bx, by),
                font, 1.4, state_color, 3, cv2.LINE_AA)

    # Thanh yaw ngang (visualize ngưỡng)
    bar_x1, bar_y  = 8, CAPTURE_H - 55
    bar_w, bar_h   = CAPTURE_W - 16, 14
    bar_x2         = bar_x1 + bar_w

    cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bar_h), (50, 50, 50), -1)

    # Vùng center
    yaw_range = 90.0
    def yaw2px(v):
        return int(bar_x1 + bar_w * (v + yaw_range) / (2 * yaw_range))

    cx_l = yaw2px(YAW_CENTER_MIN)
    cx_r = yaw2px(YAW_CENTER_MAX)
    cv2.rectangle(frame, (cx_l, bar_y), (cx_r, bar_y + bar_h), (0, 80, 0), -1)

    # Vị trí yaw hiện tại
    dot_x = np.clip(yaw2px(yaw), bar_x1 + 1, bar_x2 - 1)
    cv2.circle(frame, (dot_x, bar_y + bar_h // 2), 6, state_color, -1)

    # Label ngưỡng
    cv2.putText(frame, f"{YAW_CENTER_MIN:+.0f}", (cx_l - 24, bar_y - 3),
                font, 0.4, (140, 140, 140), 1)
    cv2.putText(frame, f"{YAW_CENTER_MAX:+.0f}", (cx_r + 2, bar_y - 3),
                font, 0.4, (140, 140, 140), 1)

    ts = time.strftime("%H:%M:%S")
    cv2.putText(frame, ts, (CAPTURE_W - 75, 20), font, 0.45, (80, 80, 80), 1)


# ─── Main ─────────────────────────────────────────────────────────────────────
def run():
    gpio_setup()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session_dir = os.path.join(OUTPUT_DIR, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(session_dir, exist_ok=True)

    model    = GazeModel(MODEL_PATH)
    detector = FaceDetector(detect_every=DETECT_EVERY)
    cam      = CameraBuffer(CAPTURE_W, CAPTURE_H, CAPTURE_FPS)

    print(f"Camera  : {CAPTURE_W}x{CAPTURE_H} @ {CAPTURE_FPS} FPS")
    print(f"Output  : {session_dir}/")
    print(f"Ngưỡng  : yaw < {YAW_CENTER_MIN} → LEFT  |  "
          f"[{YAW_CENTER_MIN}, {YAW_CENTER_MAX}] → CENTER  |  "
          f"yaw > {YAW_CENTER_MAX} → RIGHT")
    print(f"GPIO    : LEFT=pin{LED_PINS['left']}  RIGHT=pin{LED_PINS['right']}")
    print("Ctrl+C để dừng\n")

    idx       = 0
    saved     = 0
    last_state = None

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

            # ── LED control ─────────────────────────────────────────────────
            led_state = yaw_to_state(yaw)
            if led_state != last_state:
                set_leds(led_state)
                last_state = led_state

            # ── Vẽ + lưu ────────────────────────────────────────────────────
            vis = frame.copy()
            draw(vis, face_box, pitch, yaw, infer_t,
                 detector.last_mode, detector._miss_streak, idx, led_state)

            cv2.imwrite(os.path.join(session_dir, f"{idx:06d}.jpg"), vis)
            saved += 1

            print(f"[{idx:04d}]  yaw={yaw:+6.1f}  pitch={pitch:+6.1f}"
                  f"  → {led_state:<7}  {infer_t*1000:.0f}ms  {detector.last_mode}")

    except KeyboardInterrupt:
        pass
    finally:
        gpio_cleanup()
        cam.release()
        print(f"\nDone. {saved} frames → {session_dir}/")


if __name__ == "__main__":
    run()

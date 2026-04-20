"""
Gaze estimation trên Raspberry Pi — model fine-tuned
=====================================================
Đầu vào : rpicam-vid pipe (không dùng picamera2)
Đầu ra  : LED GPIO + frame ảnh lưu vào results_video/

Thay MODEL_PATH sau khi copy gaze_ft.onnx sang Pi.
Ctrl+C để dừng.
"""

import os, cv2, time, subprocess, threading
import numpy as np
import onnxruntime as ort
import RPi.GPIO as GPIO
from collections import deque

# ─── Paths & model ───────────────────────────────────────────────────────────
MODEL_PATH = "weights/gaze_mobileone.onnx"      # ← đổi sang model fine-tuned
OUTPUT_DIR = "results_video"

MODEL_CONFIGS = {
    # Model gốc (448×448)
    "weights/mobileone_s0_gaze.onnx": {
        "input_size": (448, 448), "pitch_up_sign": +1, "yaw_sign": 1,
    },
    # Model fine-tuned ResNet50 (224×224)
    "weights/gaze_mobileone.onnx": {
        "input_size": (224, 224), "pitch_up_sign": +1, "yaw_sign": 1,
    },
}

# ─── Params ──────────────────────────────────────────────────────────────────
CAPTURE_W    = 640
CAPTURE_H    = 480
CAPTURE_FPS  = 1

DETECT_EVERY = 10
VOTE_WINDOW  = 6        # số frame tích lũy trước khi ra quyết định
VOTE_THRESH  = 0.65     # tỉ lệ đa số tối thiểu

YAW_LEFT_MAX  = -8.0    # yaw < -8  → left
YAW_RIGHT_MIN = +3.0    # yaw > +3  → right
                        # khoảng giữa → center
# (Chỉnh 2 ngưỡng này sau khi chạy gaze_calibrate.py với model mới)

NUM_BINS  = 90
BIN_WIDTH = 4.0

LED_PINS = {
    "left":  17,
    "right": 27,
    "up":    22,
    "down":  23,
}

COLOR_MAP = {
    "left":   (0, 180, 255),
    "right":  (0, 180, 255),
    "up":     (0, 220, 100),
    "down":   (0, 220, 100),
    "center": (180, 180, 180),
}


# ─── GPIO ─────────────────────────────────────────────────────────────────────
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
    GPIO.output(LED_PINS["left"],  GPIO.HIGH if state == "left"   else GPIO.LOW)
    GPIO.output(LED_PINS["right"], GPIO.HIGH if state == "right"  else GPIO.LOW)
    GPIO.output(LED_PINS["up"],    GPIO.HIGH if state == "up"     else GPIO.LOW)
    GPIO.output(LED_PINS["down"],  GPIO.HIGH if state == "down"   else GPIO.LOW)


# ─── Camera buffer (rpicam-vid pipe) ─────────────────────────────────────────
class CameraBuffer:
    def __init__(self, width, height, fps):
        self.width  = width
        self.height = height
        self._fbytes = width * height * 3 // 2
        cmd = ["rpicam-vid",
               "--width", str(width), "--height", str(height),
               "--framerate", str(fps), "--codec", "yuv420",
               "--timeout", "0", "--nopreview", "-o", "-"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL,
                                      bufsize=self._fbytes * 4)
        self._frame = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        threading.Thread(target=self._reader, daemon=True).start()
        time.sleep(1.5)

    def _reader(self):
        while not self._stop.is_set():
            raw = self._proc.stdout.read(self._fbytes)
            if len(raw) < self._fbytes:
                self._stop.set(); break
            yuv = np.frombuffer(raw, dtype=np.uint8).reshape((self.height*3//2, self.width))
            with self._lock:
                self._frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

    def read(self):
        with self._lock:
            return (True, self._frame.copy()) if self._frame is not None else (False, None)

    def is_alive(self): return not self._stop.is_set()

    def release(self):
        self._stop.set()
        self._proc.terminate(); self._proc.wait(timeout=3)


# ─── Gaze model ───────────────────────────────────────────────────────────────
class GazeModel:
    def __init__(self, model_path):
        cfg = MODEL_CONFIGS[model_path]
        self.input_size    = cfg["input_size"]
        self.pitch_up_sign = cfg["pitch_up_sign"]
        self.yaw_sign      = cfg["yaw_sign"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        self.sess      = ort.InferenceSession(model_path, opts,
                                              providers=["CPUExecutionProvider"])
        self.inp_name  = self.sess.get_inputs()[0].name

        print("=== Model outputs ===")
        for i, o in enumerate(self.sess.get_outputs()):
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
        out = self.sess.run(None, {self.inp_name: x})
        t   = time.time() - t0

        def decode(logits):
            e = np.exp(logits - np.max(logits))
            p = e / e.sum()
            return float(np.sum(p * np.arange(len(p))) * BIN_WIDTH - 180)

        yaw   = decode(out[0][0]) * self.yaw_sign
        pitch = decode(out[1][0]) * self.pitch_up_sign
        return pitch, yaw, t

    def raw_direction(self, pitch, yaw):
        if   yaw < YAW_LEFT_MAX:  return "left"
        elif yaw > YAW_RIGHT_MIN: return "right"
        return "center"


# ─── Face detector + CSRT tracker ────────────────────────────────────────────
class FaceDetector:
    CASCADE = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    _CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def __init__(self, detect_every=10):
        self.detect_every = detect_every
        self._tracker     = None
        self._box         = None
        self._count       = 0
        self._tracking    = False
        self._miss        = 0
        self.last_mode    = "—"

    def _detect(self, frame):
        gray  = self._CLAHE.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        faces = self.CASCADE.detectMultiScale(gray, 1.2, 4, minSize=(50,50))
        if len(faces) == 0:
            faces = self.CASCADE.detectMultiScale(gray, 1.1, 2, minSize=(40,40))
        if len(faces) == 0:
            self._miss += 1; return None
        self._miss = 0
        x,y,w,h = max(faces, key=lambda f: f[2]*f[3])
        m = int(0.2*w)
        return (max(0,x-m), max(0,y-m),
                min(frame.shape[1],x+w+m), min(frame.shape[0],y+h+m))

    def get(self, frame):
        self._count += 1
        interval    = max(3, self.detect_every - self._miss * 2)
        need_det    = self._count % interval == 1 or not self._tracking

        if need_det:
            self.last_mode = "DET"
            box = self._detect(frame)
            if box:
                self._box = box
                x1,y1,x2,y2 = box
                self._tracker = cv2.TrackerCSRT_create()
                self._tracker.init(frame, (x1,y1,x2-x1,y2-y1))
                self._tracking = True
            else:
                self._tracking = False
        else:
            self.last_mode = "TRK"
            ok, rect = self._tracker.update(frame)
            if ok:
                rx,ry,rw,rh = [int(v) for v in rect]
                self._box = (rx,ry,rx+rw,ry+rh)
            else:
                self._tracking = False

        if self._box is None:
            h,w = frame.shape[:2]
            return frame, (0,0,w,h)

        x1,y1,x2,y2 = self._box
        x1=max(0,x1); y1=max(0,y1)
        x2=min(frame.shape[1],x2); y2=min(frame.shape[0],y2)
        return frame[y1:y2,x1:x2], (x1,y1,x2,y2)


# ─── Voting window ────────────────────────────────────────────────────────────
class VotingWindow:
    LABELS = ["left","right","up","down","center"]
    def __init__(self, window=6, thresh=0.65):
        self.window  = window
        self.thresh  = thresh
        self._buf    = deque(maxlen=window)
    def push(self, d): self._buf.append(d)
    def decision(self):
        if len(self._buf) < self.window: return None
        counts = {d: self._buf.count(d) for d in self.LABELS}
        best   = max(counts, key=counts.get)
        return best if counts[best]/self.window >= self.thresh else None


# ─── Draw ─────────────────────────────────────────────────────────────────────
def draw(frame, face_box, pitch, yaw, raw_dir, decision, infer_t, mode, miss, idx):
    x1,y1,x2,y2 = face_box
    cx,cy = (x1+x2)//2, (y1+y2)//2
    sz    = min(x2-x1, y2-y1)
    col   = COLOR_MAP.get(raw_dir, (180,180,180))

    cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)

    dx = np.tan(np.radians(yaw))
    dy = -np.tan(np.radians(pitch))
    n  = np.hypot(dx, dy)
    if n > 0: dx/=n; dy/=n
    ep = (int(cx+dx*sz*0.6), int(cy+dy*sz*0.6))
    cv2.arrowedLine(frame, (cx,cy), ep, (255,255,255), 2, tipLength=0.3)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        (f"frame  : {idx}",              (200,200,200)),
        (f"yaw    : {yaw:+.1f}",         (100,200,255)),
        (f"pitch  : {pitch:+.1f}",       (100,255,180)),
        (f"raw    : {raw_dir}",          col),
        (f"infer  : {infer_t*1000:.0f}ms",(160,160,160)),
        (f"mode   : {mode}",             (200,200,100)),
        (f"miss   : {miss}", (80,80,255) if miss>2 else (160,160,160)),
    ]
    for i,(txt,c) in enumerate(lines):
        cv2.putText(frame, txt, (8, 24+i*22), font, 0.52, c, 1, cv2.LINE_AA)

    # Decision banner
    if decision:
        msg = f">> {decision.upper()} <<"
        (dw,dh),_ = cv2.getTextSize(msg, font, 1.2, 3)
        bx = (CAPTURE_W-dw)//2
        cv2.rectangle(frame, (bx-8,4), (bx+dw+8,dh+16), (0,0,0), -1)
        cv2.putText(frame, msg, (bx,dh+8), font, 1.2, COLOR_MAP.get(decision,(200,200,200)), 3)

    # Yaw bar
    by   = CAPTURE_H - 40
    bw   = CAPTURE_W - 16
    cv2.rectangle(frame, (8,by), (8+bw,by+12), (50,50,50), -1)
    yr   = 90.0
    def y2px(v): return int(8 + bw*(v+yr)/(2*yr))
    cv2.rectangle(frame, (y2px(YAW_LEFT_MAX),by), (y2px(YAW_RIGHT_MIN),by+12), (0,80,0), -1)
    dot  = np.clip(y2px(yaw), 9, 8+bw-1)
    cv2.circle(frame, (dot, by+6), 6, col, -1)
    cv2.putText(frame, f"{YAW_LEFT_MAX:+.0f}", (y2px(YAW_LEFT_MAX)-22, by-3), font, 0.38, (140,140,140), 1)
    cv2.putText(frame, f"{YAW_RIGHT_MIN:+.0f}", (y2px(YAW_RIGHT_MIN)+2, by-3), font, 0.38, (140,140,140), 1)

    cv2.putText(frame, time.strftime("%H:%M:%S"), (CAPTURE_W-75,20), font, 0.45, (80,80,80), 1)


# ─── Main ─────────────────────────────────────────────────────────────────────
def run():
    gpio_setup()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session_dir = os.path.join(OUTPUT_DIR, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(session_dir, exist_ok=True)

    model    = GazeModel(MODEL_PATH)
    detector = FaceDetector(detect_every=DETECT_EVERY)
    voter    = VotingWindow(window=VOTE_WINDOW, thresh=VOTE_THRESH)
    cam      = CameraBuffer(CAPTURE_W, CAPTURE_H, CAPTURE_FPS)

    print(f"Model  : {MODEL_PATH}  input={MODEL_CONFIGS[MODEL_PATH]['input_size']}")
    print(f"Camera : {CAPTURE_W}x{CAPTURE_H} @ {CAPTURE_FPS}fps")
    print(f"Output : {session_dir}/")
    print(f"Thresh : yaw < {YAW_LEFT_MAX} → LEFT  |  [{YAW_LEFT_MAX},{YAW_RIGHT_MIN}] → CENTER  |  yaw > {YAW_RIGHT_MIN} → RIGHT")
    print("Ctrl+C để dừng\n")

    idx           = 0
    saved         = 0
    last_decision = None
    last_led      = None

    try:
        while cam.is_alive():
            ok, frame = cam.read()
            if not ok:
                time.sleep(0.02); continue

            idx += 1
            face_crop, face_box = detector.get(frame)

            if face_crop.size == 0:
                pitch = yaw = infer_t = 0.0
                raw_dir = "center"
            else:
                pitch, yaw, infer_t = model.predict(face_crop)
                raw_dir             = model.raw_direction(pitch, yaw)

            voter.push(raw_dir)
            decision = voter.decision()
            if decision:
                last_decision = decision
                if decision != last_led:
                    set_leds(decision)
                    last_led = decision

            vis = frame.copy()
            draw(vis, face_box, pitch, yaw, raw_dir, last_decision,
                 infer_t, detector.last_mode, detector._miss, idx)

            cv2.imwrite(os.path.join(session_dir, f"{idx:06d}.jpg"), vis)
            saved += 1

            print(f"[{idx:04d}]  yaw={yaw:+6.1f}  pitch={pitch:+6.1f}"
                  f"  raw={raw_dir:<7}  cmd={str(last_decision):<7}"
                  f"  {infer_t*1000:.0f}ms  {detector.last_mode}")

    except KeyboardInterrupt:
        pass
    finally:
        gpio_cleanup()
        cam.release()
        print(f"\nDone. {saved} frames → {session_dir}/")


if __name__ == "__main__":
    run()

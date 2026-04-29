"""
Gaze calibration — Web Streaming bằng Flask
===================================================================
Mục tiêu: Hiển thị trực tiếp video + yaw/pitch lên giao diện web.
Truy cập: http://<IP_CỦA_PI>:5000/
Ctrl+C ở terminal để dừng server.
"""

import os
import cv2
import time
import subprocess
import threading
import numpy as np
import onnxruntime as ort
from flask import Flask, Response, render_template_string

import serial
import collections

# Cấu hình Serial (Thay đổi cổng '/dev/ttyUSB0' hoặc '/dev/serial0' tuỳ phần cứng)
try:
    uart = serial.Serial('/dev/ttyAMA0', baudrate=9600, timeout=0.1)
    print("UART Connected!")
except Exception as e:
    print(f"Lỗi mở UART: {e}")
    uart = None

# 2. Khởi tạo hàng đợi để làm mượt (Lưu 5 giá trị gần nhất)
HISTORY_LEN = 5
yaw_history = collections.deque(maxlen=HISTORY_LEN)
pitch_history = collections.deque(maxlen=HISTORY_LEN)

# ─── Cấu hình ─────────────────────────────────────────────────────────────────
MODEL_PATH = "weights/mobileone_s0_gaze.onnx"

CAPTURE_W   = 640
CAPTURE_H   = 480
CAPTURE_FPS = 10         # Tăng FPS lên một chút (ví dụ 10) để stream web mượt hơn
DETECT_EVERY = 5         # Đổi thành 5 để dùng Tracker, giảm tải CPU cho Pi

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

        yaw = decode(out[0][0])
        pitch = decode(out[1][0])

        return pitch, yaw, t


# ─── Face detector + KCF tracker ────────────────────────────────────────────
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
                # Đổi sang KCF để nhẹ hơn cho Raspberry Pi và tránh lỗi thiếu thư viện CSRT
                self._tracker = cv2.TrackerKCF_create() 
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

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 80), 1)

    dx = -np.tan(np.radians(yaw))
    dy = -np.tan(np.radians(pitch))
    n  = np.hypot(dx, dy)
    if n > 0:
        dx /= n; dy /= n
    ep = (int(cx + dx * sz * 0.6), int(cy + dy * sz * 0.6))
    cv2.arrowedLine(frame, (cx, cy), ep, (255, 255, 255), 2, tipLength=0.3)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        (f"frame   : {idx}",                        (200, 200, 200)),
        (f"yaw     : {yaw:+.1f} deg",               (100, 200, 255)),
        (f"pitch   : {pitch:+.1f} deg",             (100, 255, 180)),
        (f"infer   : {infer_t*1000:.0f} ms",        (160, 160, 160)),
        (f"mode    : {mode}",                        (200, 200, 100)),
        (f"miss    : {miss}",  (80, 80, 255) if miss > 2 else (160, 160, 160)),
    ]
    for i, (text, color) in enumerate(lines):
        y = 24 + i * 22
        cv2.putText(frame, text, (8, y), font, 0.55, color, 1, cv2.LINE_AA)

    ts = time.strftime("%H:%M:%S")
    cv2.putText(frame, ts, (CAPTURE_W - 80, 20), font, 0.5, (100, 100, 100), 1)



def send_control_signal(yaw, pitch, is_face_detected):
    if not uart: return

    # Nếu không thấy mặt -> DỪNG KHẨN CẤP
    if not is_face_detected:
        cmd = "Stop#"
        uart.write(cmd.encode('utf-8'))
        return

    # Khởi tạo mặc định là Dừng để an toàn
    turn_cmd = "Stop#"

    # 1. Liếc phải (Yaw > 20)
    if yaw > 20:
        turn_cmd = "Right#"
        
    # 2. Liếc trái (Yaw < -20)
    elif yaw < -20:
        turn_cmd = "Left#"
    # 3. Nhìn thẳng (Yaw từ -20 đến 20 VÀ Pitch từ -20 đến 20)
    elif -20 <= yaw <= 20 and -20 <= pitch <= 20:
        turn_cmd = "Forward#"

    # Nối thêm ký tự '\n' (xuống dòng) để MCU biết kết thúc 1 lệnh
    # msg = f"{turn_cmd}#"
    
    # Gửi qua UART
    uart.write(turn_cmd.encode('utf-8'))
    
    # In ra terminal để debug xem code chạy đúng không
    print(f"Sent UART: {turn_cmd.strip()}")
    
# ─── Flask App & Streaming Logic ──────────────────────────────────────────────
app = Flask(__name__)

# Khởi tạo toàn cục (Global) để không load lại model mỗi khi reload trang web
print("Loading model...")
model = GazeModel(MODEL_PATH)
detector = FaceDetector(detect_every=DETECT_EVERY)
print("Starting camera...")
cam = CameraBuffer(CAPTURE_W, CAPTURE_H, CAPTURE_FPS)

def generate_frames():
    idx = 0
    while cam.is_alive():
        ok, frame = cam.read()
        if not ok:
            time.sleep(0.01)
            continue

        idx += 1
        face_crop, face_box = detector.get(frame)

        if face_crop.size == 0:
            pitch = yaw = infer_t = 0.0
            send_control_signal(0, 0, is_face_detected=False) # Mất dấu mặt -> Dừng
        else:
            pitch, yaw, infer_t = model.predict(face_crop)
            send_control_signal(yaw, pitch, is_face_detected=True)
        vis = frame.copy()
        draw(vis, face_box, pitch, yaw, infer_t, detector.last_mode, detector._miss_streak, idx)

        # In log ra terminal để debug
        # print(f"[{idx:04d}]  yaw={yaw:+6.1f}  pitch={pitch:+6.1f}  {infer_t*1000:.0f}ms  {detector.last_mode}")

        # Mã hóa frame thành định dạng JPEG
        ret, buffer = cv2.imencode('.jpg', vis)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        # Yield frame để Flask stream dạng multipart
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Smart Wheelchair - Gaze Control</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { background-color: #222; color: #fff; font-family: sans-serif; text-align: center; margin-top: 20px; }
            img { max-width: 100%; height: auto; border: 3px solid #555; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.5); }
        </style>
    </head>
    <body>
        <h2>Smart Wheelchair - Gaze Stream</h2>
        <p>Real-time Yaw/Pitch Calibration</p>
        <img src="/video_feed" alt="Gaze Video Stream">
    </body>
    </html>
    """
    return render_template_string(html)




if __name__ == "__main__":
    print(f"Camera  : {CAPTURE_W}x{CAPTURE_H} @ {CAPTURE_FPS} FPS")
    print("=====================================================")
    print("Mở trình duyệt trên máy tính/điện thoại và truy cập:")
    print("http://<IP_CỦA_RASPBERRY_PI>:5000")
    print("=====================================================")
    
    try:
        # Chạy server Flask (host 0.0.0.0 cho phép thiết bị khác trong mạng LAN truy cập)
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        print("Tắt camera và dọn dẹp bộ nhớ...")
        cam.release()
# -*- coding: utf-8 -*-
"""
Gaze Estimation - Wheelchair Control
Python 3.6 | TensorRT 8.2 GPU | YuNet Face Detect | Flask Stream | UART
Run: python3.6 gaze_trt.py
"""

import cv2
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
cuda.init()  # init thủ công, không dùng autoinit
import collections
import threading
import serial
from flask import Flask, Response
import time
from collections import deque
# ================================================================
# UART
# ================================================================
try:
    uart = serial.Serial('/dev/ttyTHS1', baudrate=9600, timeout=0.1)
    print("[UART] Connected OK")
except Exception as e:
    print(f"[UART] Not available: {e}")
    uart = None

# ================================================================
# CONFIG
# ================================================================
ENGINE_PATH = "weights/gaze.engine"
YUNET_PATH  = "face_detection_yunet_2023mar.onnx"
CAPTURE_W   = 640
CAPTURE_H   = 480
CAPTURE_FPS = 15
HISTORY_LEN = 5

# ================================================================
# FILTER
# ================================================================
fps_times = collections.deque(maxlen=30)

# ================================================================
# CAMERA
# ================================================================
class CameraBuffer:
    def __init__(self, width, height, fps):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS,          fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        for _ in range(5):
            self.cap.read()
        print("[CAM] Ready")

    def read(self):
        return self.cap.read()

    def is_alive(self):
        return self.cap.isOpened()

    def release(self):
        self.cap.release()

# ================================================================
# TENSORRT MODEL
# ================================================================
class TRTModel:
    def __init__(self, engine_path):
        # Tạo CUDA context riêng — không dùng autoinit
        self.cuda_ctx = cuda.Device(0).make_context()

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Output shape thực tế: (1, 90) bins
        self.input_shape = (1, 3, 448, 448)
        self.out_shape   = (1, 90)
        self.num_bins    = 90

        self.h_input = cuda.pagelocked_empty(self.input_shape, dtype=np.float32)
        self.h_out0  = cuda.pagelocked_empty(self.out_shape,   dtype=np.float32)
        self.h_out1  = cuda.pagelocked_empty(self.out_shape,   dtype=np.float32)

        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_out0  = cuda.mem_alloc(self.h_out0.nbytes)
        self.d_out1  = cuda.mem_alloc(self.h_out1.nbytes)

        self.stream   = cuda.Stream()
        self.bindings = [int(self.d_input), int(self.d_out0), int(self.d_out1)]

        # Pop context — worker thread sẽ push lại khi cần
        self.cuda_ctx.pop()

        print(f"[TRT] Engine loaded: {engine_path} | bins={self.num_bins}")

    def preprocess(self, img):
        x = cv2.resize(img, (448, 448))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = ((x - mean) / std).transpose(2, 0, 1)
        np.copyto(self.h_input, x.reshape(self.input_shape))

    def decode(self, logits):
        """90 bins → góc độ: mỗi bin = 4 độ, range [-180, +180)"""
        e = np.exp(logits - np.max(logits))
        p = e / e.sum()
        return float(np.sum(p * np.arange(self.num_bins)) * 4 - 180)

    def predict(self, face_img):
        """Phải gọi từ thread đã push cuda_ctx"""
        self.preprocess(face_img)

        t0 = time.time()
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_out0, self.d_out0, self.stream)
        cuda.memcpy_dtoh_async(self.h_out1, self.d_out1, self.stream)
        self.stream.synchronize()
        infer_t = time.time() - t0

        yaw   = self.decode(self.h_out0[0])
        pitch = self.decode(self.h_out1[0])
        return yaw, pitch, infer_t

# ================================================================
# FACE DETECTOR (SSD res10 — CPU, compatible mọi OpenCV version)
# ================================================================
class FaceDetector:
    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe(
            "deploy.prototxt",
            "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        )
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("[FaceDetector] SSD res10 loaded OK (CPU)")

    def get(self, frame):
        if frame is None or frame.size == 0:
            return None, None

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        best_conf = 0
        best_box  = None
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf > 0.5 and conf > best_conf:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    best_conf = conf
                    best_box  = (x1, y1, x2, y2)

        if best_box is None:
            return None, None
        x1, y1, x2, y2 = best_box
        return frame[y1:y2, x1:x2].copy(), best_box

# ================================================================
# BACKGROUND WORKER
# ================================================================
class GazeWorker:
    def __init__(self, detector, model):
        self.detector = detector
        self.model    = model
        self.frame    = None
        self.result   = {"box": None, "yaw": 0.0, "pitch": 0.0, "infer_t": 0.0}
        self.lock     = threading.Lock()
        self.running  = True
        self.thread   = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print("[Worker] Background thread started")

    def update_frame(self, frame):
        with self.lock:
            self.frame = frame.copy()

    def get_result(self):
        with self.lock:
            return dict(self.result)

    def _run(self):
        # Push CUDA context vào thread này
        self.model.cuda_ctx.push()

        yaw_hist   = collections.deque(maxlen=HISTORY_LEN)
        pitch_hist = collections.deque(maxlen=HISTORY_LEN)

        try:
            while self.running:
                with self.lock:
                    frame = self.frame

                if frame is None:
                    time.sleep(0.005)
                    continue

                face, box = self.detector.get(frame)

                infer_t = 0.0
                if face is not None:
                    yaw_raw, pitch_raw, infer_t = self.model.predict(face)
                    yaw_hist.append(yaw_raw)
                    pitch_hist.append(pitch_raw)

                if yaw_hist:
                    yaw   = float(np.mean(yaw_hist))
                    pitch = float(np.mean(pitch_hist))
                else:
                    yaw = pitch = 0.0

                with self.lock:
                    self.result = {
                        "box": box, "yaw": yaw,
                        "pitch": pitch, "infer_t": infer_t
                    }
        finally:
            self.model.cuda_ctx.pop()

    def stop(self):
        self.running = False

# ================================================================
# UART CONTROL
# ================================================================
# _last_cmd = None

class CommandController:
    def __init__(self,
                 yaw_left=-20, yaw_right=20,
                 pitch_stop=-10,
                 confirm_frames=5,      # số frame liên tiếp cùng lệnh mới gửi
                 cooldown_sec=0.4):     # thời gian tối thiểu giữa 2 lệnh
        self.yaw_left    = yaw_left
        self.yaw_right   = yaw_right
        self.pitch_stop  = pitch_stop
        self.confirm_frames = confirm_frames
        self.cooldown_sec   = cooldown_sec

        self._cmd_history  = deque(maxlen=confirm_frames)
        self._last_cmd     = None
        self._last_sent_t  = 0.0

    def _classify(self, yaw, pitch, detected):
        if not detected:
            return b"Stop#"
        if pitch < self.pitch_stop:       # nhìn xuống quá → dừng
            return b"Stop#"
        if yaw > self.yaw_right:          # yaw dương → nhìn phải
            return b"Right#"
        if yaw < self.yaw_left:           # yaw âm → nhìn trái
            return b"Left#"
        return b"Forward#"

    def update(self, yaw, pitch, detected):
        """
        Gọi mỗi frame. Trả về lệnh nếu được gửi, None nếu không.
        """
        cmd = self._classify(yaw, pitch, detected)
        self._cmd_history.append(cmd)

        # Chỉ xử lý khi đủ confirm_frames
        if len(self._cmd_history) < self.confirm_frames:
            return None

        # Tất cả frame trong window phải đồng thuận
        if len(set(self._cmd_history)) != 1:
            return None

        # Không gửi lại lệnh giống lệnh cũ
        if cmd == self._last_cmd:
            return None

        # Cooldown — tránh gửi quá nhanh
        now = time.time()
        if now - self._last_sent_t < self.cooldown_sec:
            return None

        # Gửi lệnh
        self._last_cmd    = cmd
        self._last_sent_t = now
        return cmd


controller = CommandController(
    yaw_left=-20,
    yaw_right=20,
    pitch_stop=-20,
    confirm_frames=5,     # ~5 frame liên tiếp ≈ 0.33s ở 15fps
    cooldown_sec=0.4
)
_current_cmd = "---"
def send_control_signal(yaw, pitch, detected):
    global _current_cmd
    cmd = controller.update(yaw, pitch, detected)
    if cmd is not None:
        _current_cmd = cmd.decode().replace("#", "")
        print(f"[CMD] {_current_cmd}")
        if uart:
            uart.write(cmd)

# ================================================================
# DRAW
# ================================================================
def draw(frame, box, pitch, yaw, infer_t, fps):
    if box:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # ===== GAZE BALL (góc kế) =====
    r      = 50   # bán kính vòng tròn
    cx, cy = 580, 430  # vị trí trên frame (góc dưới phải)
    max_angle = 30.0   # góc tối đa map về mép vòng tròn

    # Vẽ vòng tròn nền
    cv2.circle(frame, (cx, cy), r, (80, 80, 80), -1)   # nền xám
    cv2.circle(frame, (cx, cy), r, (200, 200, 200), 2) # viền
    # Crosshair
    cv2.line(frame, (cx - r, cy), (cx + r, cy), (60, 60, 60), 1)
    cv2.line(frame, (cx, cy - r), (cx, cy + r), (60, 60, 60), 1)
    # Vòng dead zone
    cv2.circle(frame, (cx, cy), int(r * 5 / max_angle), (100, 100, 100), 1)

    # Map góc → tọa độ bi
    bx = int(cx + np.clip(yaw   / max_angle, -1, 1) * r)
    by = int(cy - np.clip(pitch / max_angle, -1, 1) * r)  # pitch+ = lên

    # Màu bi theo vùng
    dist = np.hypot(yaw, pitch)
    if dist < 5:
        ball_color = (0, 255, 0)      # xanh = dead zone / Forward
    elif abs(yaw) > abs(pitch):
        ball_color = (0, 255, 255)    # vàng = trái/phải
    else:
        ball_color = (0, 100, 255)    # cam = lên/xuống

    cv2.circle(frame, (bx, by), 10, ball_color, -1)
    cv2.circle(frame, (bx, by), 10, (255, 255, 255), 1)

    # Text
    cv2.putText(frame, f"Yaw:   {yaw:+.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 100), 2)
    cv2.putText(frame, f"Pitch: {pitch:+.1f}", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 100), 2)
    cv2.putText(frame, f"Infer: {infer_t*1000:.0f}ms", (10, 86),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS:   {fps:.1f}", (10, 114),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # CMD to ở giữa dưới
    cmd_color = {
        "Forward": (0, 255, 0),
        "Left":    (0, 255, 255),
        "Right":   (0, 255, 255),
        "Stop":    (0, 0, 255),
    }.get(_current_cmd, (255, 255, 255))
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, _), _ = cv2.getTextSize(_current_cmd, font, 1.2, 3)
    cv2.putText(frame, _current_cmd,
                ((frame.shape[1] - tw) // 2, frame.shape[0] - 15),
                font, 1.2, cmd_color, 3)
# ================================================================
# FLASK
# ================================================================
app      = Flask(__name__)
model    = TRTModel(ENGINE_PATH)
detector = FaceDetector()
cam      = CameraBuffer(CAPTURE_W, CAPTURE_H, CAPTURE_FPS)
worker   = GazeWorker(detector, model)

def generate_frames():
    while cam.is_alive():
        ret, frame = cam.read()
        if not ret or frame is None or frame.size == 0:
            continue

        worker.update_frame(frame)
        r = worker.get_result()
        # Nhin phai yaw -, trai +
        # Nhin len la pitch +, xuong pitch -
        send_control_signal(r["yaw"], r["pitch"], r["box"] is not None)

        fps_times.append(time.time())
        fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0]) if len(fps_times) >= 2 else 0.0

        draw(frame, r["box"], r["pitch"], r["yaw"], r["infer_t"], fps)

        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return "<h2>Gaze Tracking (TensorRT GPU)</h2><img src='/video_feed'>"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    print("🚀 Open: http://<JETSON_IP>:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
# -*- coding: utf-8 -*-
"""
Gaze Estimation - Wheelchair Control
Python 3.6 | TensorRT 8.2 GPU | SSD Face Detect | Fast UART | No Stream

Bản không stream để giảm tải CPU.
Giữ nguyên kiểu crop mặt của code cũ để không làm lệch yaw/pitch,
tối ưu SSD bằng cache box và tách UART ra thread riêng 50 Hz.

Run:
    python3.6 gaze_trt_olddetect_uartfast_nostream.py
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

# ================================================================
# CONFIG
# ================================================================
ENGINE_PATH = "weights/gaze.engine"

CAPTURE_W   = 640
CAPTURE_H   = 480
CAPTURE_FPS = 15
HISTORY_LEN = 5

# SSD optimize nhưng vẫn giữ crop y như code cũ
DETECT_EVERY = 3       # 1 = chính xác nhất nhưng chậm; 3 = cân bằng; 5 = nhanh hơn
FACE_MAX_AGE = 0.50    # giây giữ box cũ nếu SSD miss ngắn
SSD_CONF_TH  = 0.50

# UART fast
UART_PORT = "/dev/ttyTHS1"
UART_BAUD = 115200
UART_HZ   = 50
UART_KEEPALIVE_SEC = 1.5   # chỉ nhắc lại lệnh mỗi 1.5s nếu không đổi trạng thái
USE_SHORT_CMD = False  # False: Forward#/Left#/Right#/Stop# | True: F/L/R/S

# ================================================================
# GLOBAL STATE
# ================================================================
_current_cmd = "Stop"

# ================================================================
# UART
# ================================================================
try:
    uart = serial.Serial(UART_PORT, baudrate=UART_BAUD, timeout=0)
    print("[UART] Connected OK | {} @ {}".format(UART_PORT, UART_BAUD))
except Exception as e:
    print("[UART] Not available: {}".format(e))
    uart = None

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


class CameraThread:
    """Đọc camera liên tục, độc lập với Flask để điều khiển vẫn chạy khi không mở web."""
    def __init__(self, cam):
        self.cam = cam
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.running = True
        self.last_read_t = 0.0
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        print("[CAM_THREAD] Started")

    def _run(self):
        while self.running and self.cam.is_alive():
            ret, frame = self.cam.read()
            if not ret or frame is None or frame.size == 0:
                time.sleep(0.005)
                continue
            with self.lock:
                self.frame = frame
                self.frame_id += 1
                self.last_read_t = time.time()

    def get_frame(self, copy=True):
        with self.lock:
            if self.frame is None:
                return None, -1
            if copy:
                return self.frame.copy(), self.frame_id
            return self.frame, self.frame_id

    def stop(self):
        self.running = False

# ================================================================
# TENSORRT MODEL
# ================================================================
class TRTModel:
    def __init__(self, engine_path):
        self.cuda_ctx = cuda.Device(0).make_context()

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

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

        self.cuda_ctx.pop()
        print("[TRT] Engine loaded: {} | bins={}".format(engine_path, self.num_bins))

    def preprocess(self, img):
        # GIỮ Y NHƯ CODE CŨ: crop mặt đưa vào resize 448, RGB, normalize ImageNet
        x = cv2.resize(img, (448, 448))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = ((x - mean) / std).transpose(2, 0, 1)
        np.copyto(self.h_input, x.reshape(self.input_shape))

    def decode(self, logits):
        e = np.exp(logits - np.max(logits))
        p = e / e.sum()
        return float(np.sum(p * np.arange(self.num_bins)) * 4 - 180)

    def predict(self, face_img):
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
# FACE DETECTOR - OLD CROP + SSD CACHE
# ================================================================
class FaceDetector:
    """
    Giữ nguyên cách detect/crop của code cũ:
        return frame[y1:y2, x1:x2].copy(), best_box

    Tối ưu:
        - SSD chỉ chạy mỗi DETECT_EVERY frame.
        - Frame giữa các lần detect dùng lại last_box.
        - Không expand box, không ROI crop, không đổi input khuôn mặt -> tránh lệch yaw/pitch.
    """
    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe(
            "deploy.prototxt",
            "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        )
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.frame_count = 0
        self.last_box = None
        self.last_seen_t = 0.0
        self.miss_count = 0
        self.last_detect_t = 0.0
        print("[FaceDetector] SSD res10 loaded OK (CPU, old-crop cached mode)")

    def _detect_ssd(self, frame):
        h, w = frame.shape[:2]

        # GIỮ Y NHƯ CODE CŨ
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        best_conf = 0.0
        best_box  = None
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf > SSD_CONF_TH and conf > best_conf:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    best_conf = conf
                    best_box  = (x1, y1, x2, y2)

        return best_box, best_conf

    def get(self, frame):
        if frame is None or frame.size == 0:
            return None, None

        now = time.time()
        self.frame_count += 1

        cache_alive = (self.last_box is not None) and ((now - self.last_seen_t) <= FACE_MAX_AGE)
        need_detect = (self.last_box is None) or (self.frame_count % DETECT_EVERY == 0)

        if need_detect:
            t0 = time.time()
            box, conf = self._detect_ssd(frame)
            self.last_detect_t = time.time() - t0

            if box is not None:
                self.last_box = box
                self.last_seen_t = now
                self.miss_count = 0
            else:
                self.miss_count += 1
                # SSD thỉnh thoảng miss 1 frame, không xóa ngay nếu cache còn mới
                if (not cache_alive) or self.miss_count >= 2:
                    self.last_box = None

        if self.last_box is None:
            return None, None

        # GIỮ CROP Y NHƯ CODE CŨ, KHÔNG MARGIN
        x1, y1, x2, y2 = self.last_box
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None, None

        return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

# ================================================================
# BACKGROUND GAZE WORKER
# ================================================================
class GazeWorker:
    def __init__(self, camera_thread, detector, model):
        self.camera_thread = camera_thread
        self.detector = detector
        self.model    = model
        self.result   = {
            "box": None, "yaw": 0.0, "pitch": 0.0,
            "infer_t": 0.0, "det_t": 0.0, "frame_id": -1,
            "updated_t": 0.0
        }
        self.lock = threading.Lock()
        self.running = True
        self.last_processed_id = -1
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        print("[Worker] Background gaze thread started")

    def get_result(self):
        with self.lock:
            return dict(self.result)

    def _run(self):
        self.model.cuda_ctx.push()

        yaw_hist   = collections.deque(maxlen=HISTORY_LEN)
        pitch_hist = collections.deque(maxlen=HISTORY_LEN)

        try:
            while self.running:
                frame, fid = self.camera_thread.get_frame(copy=True)

                if frame is None:
                    time.sleep(0.005)
                    continue

                if fid == self.last_processed_id:
                    time.sleep(0.002)
                    continue
                self.last_processed_id = fid

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
                        "box": box,
                        "yaw": yaw,
                        "pitch": pitch,
                        "infer_t": infer_t,
                        "det_t": self.detector.last_detect_t,
                        "frame_id": fid,
                        "updated_t": time.time()
                    }
        finally:
            self.model.cuda_ctx.pop()

    def stop(self):
        self.running = False

# ================================================================
# UART CONTROL
# ================================================================
class CommandController:
    IDLE     = "IDLE"
    RUNNING  = "RUNNING"
    STOPPING = "STOPPING"

    def __init__(self):
        # Giữ hướng phân loại như code cũ
        self.YAW_THRESHOLD = 20
        self.DEAD_ZONE     = 15

        # Hysteresis pitch để không nhảy Stop khi pitch quanh ngưỡng
        self.PITCH_STOP_ENTER = -18
        self.PITCH_STOP_EXIT  = -10
        self._pitch_stop_latched = False

        # Mất mặt ngắn không Stop ngay để tránh SSD miss/cache nháy
        self.FACE_LOST_TIMEOUT = 0.20
        self._last_face_t = 0.0

        # Thời gian phản hồi
        self.T_CONFIRM_MOVE = 0.10
        self.T_CONFIRM_TURN = 0.14

        # Khi đổi Forward/Left/Right phải gửi Stop trước,
        # chờ buffer này rồi mới cho gửi lệnh mới.
        self.T_STOP_BUFFER  = 0.08

        # Không gửi liên tục nữa.
        # Chỉ gửi khi đổi lệnh hoặc nhắc lại keepalive sau 1.5s.
        self.T_KEEPALIVE    = UART_KEEPALIVE_SEC

        self._state        = self.IDLE
        self._intent       = None
        self._intent_since = time.time()

        self._current_cmd  = self._cmd_bytes("STOP")
        self._last_sent_t  = 0.0
        self._stop_since   = time.time()

    def _cmd_bytes(self, intent):
        if USE_SHORT_CMD:
            return {
                "FORWARD": b"F",
                "LEFT":    b"L",
                "RIGHT":   b"R",
                "STOP":    b"S",
            }.get(intent, b"S")
        return {
            "FORWARD": b"Forward#",
            "LEFT":    b"Left#",
            "RIGHT":   b"Right#",
            "STOP":    b"Stop#",
        }.get(intent, b"Stop#")

    def _cmd_to_name(self, cmd):
        if cmd in (b"F", b"Forward#"):
            return "FORWARD"
        if cmd in (b"L", b"Left#"):
            return "LEFT"
        if cmd in (b"R", b"Right#"):
            return "RIGHT"
        return "STOP"

    def _face_ok(self, detected, now):
        if detected:
            self._last_face_t = now
            return True
        return (now - self._last_face_t) <= self.FACE_LOST_TIMEOUT

    def _pitch_stop(self, pitch):
        if not self._pitch_stop_latched:
            if pitch < self.PITCH_STOP_ENTER:
                self._pitch_stop_latched = True
        else:
            if pitch > self.PITCH_STOP_EXIT:
                self._pitch_stop_latched = False
        return self._pitch_stop_latched

    def _classify_intent(self, yaw, pitch, detected, now):
        if not self._face_ok(detected, now):
            return "STOP"

        if self._pitch_stop(pitch):
            return "STOP"

        # GIỮ CHIỀU YAW NHƯ CODE CŨ
        if yaw > self.YAW_THRESHOLD:
            return "RIGHT"
        if yaw < -self.YAW_THRESHOLD:
            return "LEFT"
        if abs(yaw) <= self.DEAD_ZONE:
            return "FORWARD"

        # Vùng mơ hồ giữa DEAD_ZONE và YAW_THRESHOLD:
        # giữ intent cũ để tránh Stop xen kẽ
        if self._intent in ("FORWARD", "LEFT", "RIGHT"):
            return self._intent
        return "STOP"

    def update(self, yaw, pitch, detected):
        now = time.time()
        intent = self._classify_intent(yaw, pitch, detected, now)

        # STOP an toàn: mất mặt lâu hoặc cúi xuống rõ
        if intent == "STOP":
            self._state = self.STOPPING
            self._stop_since = now
            self._intent = None
            return self._send(self._cmd_bytes("STOP"), now)

        # Theo dõi intent có giữ ổn định không
        if intent != self._intent:
            self._intent = intent
            self._intent_since = now

        held_time = now - self._intent_since
        desired_cmd = self._cmd_bytes(intent)
        current_name = self._cmd_to_name(self._current_cmd)

        # Nếu đang chạy một hướng mà người dùng đổi sang hướng khác,
        # PHẢI gửi Stop trước, không nhảy thẳng Forward -> Left/Right.
        if self._state == self.RUNNING:
            if current_name != "STOP" and current_name != intent:
                self._state = self.STOPPING
                self._stop_since = now
                return self._send(self._cmd_bytes("STOP"), now)

        # Nếu đang IDLE/STOPPING thì chờ dừng đủ buffer + confirm intent
        if self._state in (self.IDLE, self.STOPPING):
            stop_ok = (now - self._stop_since) >= self.T_STOP_BUFFER

            if not stop_ok:
                return self._send(self._cmd_bytes("STOP"), now)

            t_confirm = self.T_CONFIRM_TURN if intent in ("LEFT", "RIGHT") else self.T_CONFIRM_MOVE

            if held_time < t_confirm:
                return self._send(self._cmd_bytes("STOP"), now)

            self._state = self.RUNNING

        if self._state == self.RUNNING:
            return self._send(desired_cmd, now)

        return None

    def _send(self, cmd, now):
        """
        Chỉ trả về cmd khi:
        - Lệnh thay đổi, ví dụ Forward -> Stop, Stop -> Right
        - Hoặc keepalive sau UART_KEEPALIVE_SEC giây.
        """
        changed = cmd != self._current_cmd
        keepalive = (now - self._last_sent_t) >= self.T_KEEPALIVE

        if changed or keepalive:
            self._current_cmd = cmd
            self._last_sent_t = now
            return cmd

        return None

    @property
    def state(self):
        return self._state

    @property
    def current_cmd(self):
        if self._current_cmd == b"F":
            return "Forward"
        if self._current_cmd == b"L":
            return "Left"
        if self._current_cmd == b"R":
            return "Right"
        if self._current_cmd == b"S":
            return "Stop"
        return self._current_cmd.decode(errors="ignore").replace("#", "")


class UARTControlThread:
    def __init__(self, worker, controller, hz=UART_HZ):
        self.worker = worker
        self.controller = controller
        self.period = 1.0 / float(hz)
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        print("[UART_CTRL] Started | check {} Hz | keepalive {}s".format(hz, UART_KEEPALIVE_SEC))

    def _run(self):
        global _current_cmd
        while self.running:
            t0 = time.time()

            r = self.worker.get_result()
            detected = r["box"] is not None

            # Dùng yaw/pitch RAW như code cũ, không bù center để không làm lệch left/right
            cmd = self.controller.update(r["yaw"], r["pitch"], detected)

            # Chỉ vào đây khi thực sự gửi UART:
            # đổi trạng thái hoặc keepalive sau UART_KEEPALIVE_SEC giây.
            if cmd is not None:
                _current_cmd = self.controller.current_cmd
                print("[UART SEND][{}] yaw={:+.1f} pitch={:+.1f} -> {}".format(
                    self.controller.state, r["yaw"], r["pitch"], _current_cmd
                ))
                if uart:
                    try:
                        uart.write(cmd)
                    except Exception as e:
                        print("[UART] write error: {}".format(e))

            dt = time.time() - t0
            sleep_t = self.period - dt
            if sleep_t > 0:
                time.sleep(sleep_t)

    def stop(self):
        self.running = False


# ================================================================
# STATUS LOGGER - NO STREAM
# ================================================================
class StatusLoggerThread:
    def __init__(self, worker, interval=1.0):
        self.worker = worker
        self.interval = float(interval)
        self.running = True
        self.last_frame_id = -1
        self.last_t = time.time()
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        print("[STATUS] Started | interval {}s".format(self.interval))

    def _run(self):
        while self.running:
            time.sleep(self.interval)
            r = self.worker.get_result()
            now = time.time()
            fid = r.get("frame_id", -1)
            dt = now - self.last_t
            fps = 0.0
            if dt > 0 and self.last_frame_id >= 0:
                fps = float(fid - self.last_frame_id) / dt
            self.last_frame_id = fid
            self.last_t = now
            age = now - r.get("updated_t", 0.0) if r.get("updated_t", 0.0) else -1.0
            print("[STATUS] face={} yaw={:+.1f} pitch={:+.1f} cmd={} infer={:.0f}ms ssd={:.0f}ms fps~{:.1f} age={:.2f}s".format(
                r.get("box") is not None,
                r.get("yaw", 0.0),
                r.get("pitch", 0.0),
                _current_cmd,
                r.get("infer_t", 0.0) * 1000.0,
                r.get("det_t", 0.0) * 1000.0,
                fps,
                age
            ))

    def stop(self):
        self.running = False

# ================================================================
# INIT / MAIN - NO FLASK STREAM
# ================================================================
def main():
    model = TRTModel(ENGINE_PATH)
    detector = FaceDetector()
    cam = CameraBuffer(CAPTURE_W, CAPTURE_H, CAPTURE_FPS)
    cam_thr = CameraThread(cam)
    worker = GazeWorker(cam_thr, detector, model)
    controller = CommandController()
    uart_thr = UARTControlThread(worker, controller, hz=UART_HZ)
    status_thr = StatusLoggerThread(worker, interval=1.0)

    print("[MAIN] No-stream mode started")
    print("[MAIN] UART: {} @ {}, check {} Hz, keepalive {}s".format(
        UART_PORT, UART_BAUD, UART_HZ, UART_KEEPALIVE_SEC
    ))
    print("[MAIN] Press Ctrl+C to stop")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[MAIN] Stopping...")
    finally:
        status_thr.stop()
        uart_thr.stop()
        worker.stop()
        cam_thr.stop()
        cam.release()
        if uart:
            try:
                uart.write(controller._cmd_bytes("STOP"))
                uart.close()
            except Exception:
                pass
        print("[MAIN] Stopped")

if __name__ == "__main__":
    main()

"""
Face detection debug stream — Flask MJPEG
==========================================
Chạy:  python debug_stream.py
Mở:    http://<IP_RASPBERRY_PI>:5000

Không cần picamera2, dùng rpicam-vid pipe giống run_camera_gaze.py.
Cài Flask nếu chưa có:  pip install flask
"""

import cv2
import time
import threading
import subprocess
import numpy as np
from flask import Flask, Response, render_template_string

# ─── Params ──────────────────────────────────────────────────────────────────
CAPTURE_W = 640
CAPTURE_H = 480
CAPTURE_FPS = 30
DETECT_EVERY = 10       # Haar detect mỗi N frame
STREAM_QUALITY = 80     # JPEG quality cho stream (1-100)

# ─── Camera buffer (rpicam-vid pipe) ─────────────────────────────────────────
class CameraBuffer:
    def __init__(self, width=640, height=480, fps=30):
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
        t = threading.Thread(target=self._reader, daemon=True)
        t.start()
        time.sleep(1.0)

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

    def release(self):
        self._stop.set()
        self._proc.terminate()
        self._proc.wait(timeout=3)


# ─── Face detector với CLAHE ──────────────────────────────────────────────────
class FaceDetector:
    CASCADE = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    _CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __init__(self, detect_every=10):
        self.detect_every = detect_every
        self._tracker     = None
        self._box         = None
        self._frame_count = 0
        self._tracking    = False
        self._miss_streak = 0
        self._last_mode   = "—"   # "DET" hoặc "TRK" để hiển thị trên stream

    def _preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self._CLAHE.apply(gray)

    def _detect(self, frame):
        gray  = self._preprocess(frame)
        faces = self.CASCADE.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=4, minSize=(50, 50)
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
        x1 = max(0, x - m);                 y1 = max(0, y - m)
        x2 = min(frame.shape[1], x + w + m); y2 = min(frame.shape[0], y + h + m)
        return (x1, y1, x2, y2)

    def _init_tracker(self, frame, box):
        x1, y1, x2, y2 = box
        self._tracker  = cv2.TrackerCSRT_create()
        self._tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
        self._tracking = True

    def get(self, frame):
        self._frame_count += 1
        effective_interval = max(3, self.detect_every - self._miss_streak * 2)
        need_detect = (
            self._frame_count % effective_interval == 1
            or not self._tracking
        )

        if need_detect:
            self._last_mode = "DET"
            box = self._detect(frame)
            if box:
                self._box = box
                self._init_tracker(frame, box)
            else:
                self._tracking = False
        else:
            self._last_mode = "TRK"
            ok, rect = self._tracker.update(frame)
            if ok:
                rx, ry, rw, rh = [int(v) for v in rect]
                self._box = (rx, ry, rx + rw, ry + rh)
            else:
                self._tracking = False

        return self._box   # None nếu không thấy mặt


# ─── Global state ─────────────────────────────────────────────────────────────
cam      = CameraBuffer(CAPTURE_W, CAPTURE_H, CAPTURE_FPS)
detector = FaceDetector(DETECT_EVERY)

_stream_frame = None
_stream_lock  = threading.Lock()
_stats = {
    "fps":    0.0,
    "mode":   "—",
    "miss":   0,
    "found":  False,
    "bright": 0,
}

def _process_loop():
    """Thread xử lý: detect + vẽ → ghi vào _stream_frame."""
    global _stream_frame, _stats
    fps_smooth = 0.0
    t_prev     = time.time()

    while True:
        ok, frame = cam.read()
        if not ok:
            time.sleep(0.01)
            continue

        # ── Tính độ sáng trung bình (debug thiếu sáng) ──────────────────────
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bright = int(np.mean(gray))

        # ── Detect / Track ───────────────────────────────────────────────────
        box   = detector.get(frame)
        found = box is not None

        vis = frame.copy()

        if found:
            x1, y1, x2, y2 = box
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(vis.shape[1], x2); y2 = min(vis.shape[0], y2)

            # Box màu xanh = tracking, vàng = vừa detect
            color = (0, 255, 100) if detector._last_mode == "TRK" else (0, 200, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Tâm + crosshair nhỏ
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.drawMarker(vis, (cx, cy), color,
                           cv2.MARKER_CROSS, 16, 1)

            # Nhãn mode + kích thước box
            label = f"{detector._last_mode}  {x2-x1}x{y2-y1}px"
            cv2.putText(vis, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        else:
            # Không tìm thấy mặt → hiển thị cảnh báo
            msg = f"No face — miss:{detector._miss_streak}"
            cv2.putText(vis, msg, (10, CAPTURE_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)

        # ── CLAHE preview (góc phải dưới, thu nhỏ) ──────────────────────────
        clahe_gray  = detector._CLAHE.apply(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        )
        clahe_bgr   = cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2BGR)
        thumb_w, thumb_h = 160, 120
        thumb = cv2.resize(clahe_bgr, (thumb_w, thumb_h))
        # Viền + label
        cv2.rectangle(thumb, (0, 0), (thumb_w - 1, thumb_h - 1), (80, 80, 80), 1)
        cv2.putText(thumb, "CLAHE", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        vis[CAPTURE_H - thumb_h:CAPTURE_H, CAPTURE_W - thumb_w:CAPTURE_W] = thumb

        # ── FPS ─────────────────────────────────────────────────────────────
        now        = time.time()
        fps_smooth = 0.85 * fps_smooth + 0.15 / max(now - t_prev, 1e-6)
        t_prev     = now

        # ── HUD góc trên trái ───────────────────────────────────────────────
        lines = [
            f"FPS   : {fps_smooth:.1f}",
            f"Mode  : {detector._last_mode}",
            f"Miss  : {detector._miss_streak}",
            f"Bright: {bright}/255",
            f"Face  : {'YES' if found else 'NO'}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(vis, line, (8, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (230, 230, 230), 1, cv2.LINE_AA)

        # Cảnh báo thiếu sáng
        if bright < 60:
            cv2.putText(vis, "LOW LIGHT", (CAPTURE_W // 2 - 60, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)

        # ── Ghi vào buffer stream ────────────────────────────────────────────
        _, jpeg = cv2.imencode(".jpg", vis,
                               [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY])
        with _stream_lock:
            _stream_frame = jpeg.tobytes()
            _stats.update({
                "fps":   round(fps_smooth, 1),
                "mode":  detector._last_mode,
                "miss":  detector._miss_streak,
                "found": found,
                "bright": bright,
            })


# Khởi động thread xử lý
threading.Thread(target=_process_loop, daemon=True).start()


# ─── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__)

PAGE = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gaze Debug Stream</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #0d0d0d;
    color: #e0e0e0;
    font-family: 'Courier New', monospace;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 20px;
    gap: 16px;
  }
  h1 {
    font-size: 13px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #555;
    padding-top: 4px;
  }
  .stream-wrap {
    position: relative;
    border: 1px solid #222;
    border-radius: 6px;
    overflow: hidden;
    background: #000;
  }
  .stream-wrap img {
    display: block;
    max-width: 100%;
    width: 640px;
  }
  .badge {
    position: absolute;
    top: 10px;
    right: 10px;
    background: rgba(0,0,0,0.65);
    border: 1px solid #333;
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 11px;
    letter-spacing: 0.05em;
    color: #aaa;
  }
  .stats {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    justify-content: center;
  }
  .stat {
    background: #141414;
    border: 1px solid #222;
    border-radius: 6px;
    padding: 10px 18px;
    min-width: 110px;
    text-align: center;
  }
  .stat-label {
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #444;
    margin-bottom: 6px;
  }
  .stat-value {
    font-size: 22px;
    font-weight: bold;
    color: #c8ff60;
  }
  .stat-value.warn { color: #ff6060; }
  .stat-value.ok   { color: #60ffaa; }
  .stat-value.dim  { color: #888; }
  .note {
    font-size: 11px;
    color: #333;
    text-align: center;
    padding-bottom: 8px;
  }
</style>
</head>
<body>
<h1>Pi Camera — Face Detection Debug</h1>

<div class="stream-wrap">
  <img src="/video_feed" alt="stream">
  <div class="badge">MJPEG LIVE</div>
</div>

<div class="stats" id="stats">
  <div class="stat">
    <div class="stat-label">FPS</div>
    <div class="stat-value dim" id="s-fps">—</div>
  </div>
  <div class="stat">
    <div class="stat-label">Mode</div>
    <div class="stat-value dim" id="s-mode">—</div>
  </div>
  <div class="stat">
    <div class="stat-label">Miss streak</div>
    <div class="stat-value dim" id="s-miss">—</div>
  </div>
  <div class="stat">
    <div class="stat-label">Brightness</div>
    <div class="stat-value dim" id="s-bright">—</div>
  </div>
  <div class="stat">
    <div class="stat-label">Face</div>
    <div class="stat-value dim" id="s-face">—</div>
  </div>
</div>

<p class="note">Góc phải dưới: preview CLAHE (ảnh sau xử lý tăng sáng)</p>

<script>
async function poll() {
  try {
    const r = await fetch('/stats');
    const d = await r.json();

    const fps = document.getElementById('s-fps');
    fps.textContent = d.fps;
    fps.className   = 'stat-value ' + (d.fps < 5 ? 'warn' : 'ok');

    const mode = document.getElementById('s-mode');
    mode.textContent = d.mode;
    mode.className   = 'stat-value ' + (d.mode === 'DET' ? 'warn' : 'ok');

    const miss = document.getElementById('s-miss');
    miss.textContent = d.miss;
    miss.className   = 'stat-value ' + (d.miss > 3 ? 'warn' : d.miss > 0 ? 'dim' : 'ok');

    const bright = document.getElementById('s-bright');
    bright.textContent = d.bright;
    bright.className   = 'stat-value ' + (d.bright < 60 ? 'warn' : d.bright < 100 ? 'dim' : 'ok');

    const face = document.getElementById('s-face');
    face.textContent = d.found ? 'YES' : 'NO';
    face.className   = 'stat-value ' + (d.found ? 'ok' : 'warn');
  } catch(e) {}
}
setInterval(poll, 500);
poll();
</script>
</body>
</html>"""


def _gen_frames():
    while True:
        with _stream_lock:
            frame = _stream_frame
        if frame is None:
            time.sleep(0.02)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame +
            b"\r\n"
        )
        time.sleep(1.0 / CAPTURE_FPS)


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/video_feed")
def video_feed():
    return Response(
        _gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/stats")
def stats():
    from flask import jsonify
    with _stream_lock:
        s = dict(_stats)
    return jsonify(s)


if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\nDebug stream running:")
    print(f"  Local  : http://localhost:5000")
    print(f"  Network: http://{local_ip}:5000\n")
    app.run(host="0.0.0.0", port=5000, threaded=True)

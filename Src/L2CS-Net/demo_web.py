import cv2
import sys
import os
from flask import Flask, Response

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from l2cs.pipeline import Pipeline

print("Loading model...")

pipeline = Pipeline(
    weights="models/L2CSNet_gaze360.pkl",
    arch="ResNet50",
    device="cpu",
    include_detector=False
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("No camera")

app = Flask(__name__)

def gen():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = pipeline.step(frame)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("http://<IP_PI>:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)


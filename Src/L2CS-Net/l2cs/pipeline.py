import cv2
import torch
import numpy as np

from l2cs.model import L2CS
from l2cs.face_detector import RetinaFace
from l2cs.utils import draw_gaze


print("Loading model...")

pipeline = Pipeline(
    weights="models/L2CSNet_gaze360.pkl",
    arch="ResNet50",
    device="cpu",
    include_detector=True
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("No camera")

print("Camera started")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = pipeline.predict(frame)

    if len(results) == 0:
        print("No face detected")
        continue

    for res in results:
        yaw = res["yaw"]
        pitch = res["pitch"]

        print(f"Yaw: {yaw:.2f} deg | Pitch: {pitch:.2f} deg")

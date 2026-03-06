import cv2
import torch
import numpy as np
from l2cs.model import L2CS

device = torch.device("cpu")

# =====================
# Load model (KHÔNG truyền arch, weights)
# =====================
model = L2CS()
checkpoint = torch.load(
    "models/L2CSNet_gaze360.pkl",
    map_location=device
)

# Một số repo dùng key khác nhau
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

print("Model loaded")

# =====================
# Camera
# =====================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("No camera")

print("Camera started")

# =====================
# Main loop
# =====================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w, _ = frame.shape

    # ROI giữa ảnh
    cx, cy = w // 2, h // 2
    size = min(w, h) // 2

    face = frame[
        cy - size//2 : cy + size//2,
        cx - size//2 : cx + size//2
    ]

    if face.size == 0:
        continue

    face = cv2.resize(face, (224, 224))
    face = face[:, :, ::-1] / 255.0   # BGR -> RGB

    face = torch.from_numpy(face).permute(2, 0, 1)
    face = face.unsqueeze(0).float().to(device)

    with torch.no_grad():
        yaw, pitch = model(face)

    print(f"Yaw: {yaw.item():.2f} deg | Pitch: {pitch.item():.2f} deg")

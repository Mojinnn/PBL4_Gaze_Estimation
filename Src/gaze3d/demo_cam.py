# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# Modified for webcam support

import argparse
import warnings
from functools import partial

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from boxmot import OCSORT
from matplotlib import colormaps
from PIL import Image

from src.models.gat_model import GaT, HeadDict, MLPHead, Swin3D

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# ================================ ARGS ================================ #
parser = argparse.ArgumentParser(description="Predict gaze using webcam")
parser.add_argument(
    "--camera-id", type=int, default=0, help="Camera ID (default: 0 for primary webcam)"
)
parser.add_argument(
    "--output-dir", type=str, default="output", help="Name of the folder where to save the output."
)
parser.add_argument(
    "--ckpt-path",
    type=str,
    default="./checkpoints/gat_stwsge_gaze360_gf.ckpt",
    help="Path to the pre-trained model checkpoint.",
)
parser.add_argument(
    "--device", type=str, default="cpu", help="Device to use for inference (cpu or cuda)."
)
parser.add_argument(
    "--save-output", action="store_true", help="Save the output video"
)
args = parser.parse_args()

# =============================== GLOBALS =============================== #
CMAP = colormaps.get_cmap("brg")
COLORS = [
    (199, 21, 133),
    (0, 128, 0),
    (30, 144, 255),
    (220, 20, 60),
    (218, 165, 32),
    (47, 79, 79),
    (139, 69, 19),
    (128, 0, 128),
    (0, 128, 128),
]
DET_THR = 0.4

# Image normalization constants
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

# ========================= UTILITY FUNCTIONS =========================== #

def load_head_detection_model(device):
    ckpt_path = "./weights/crowdhuman_yolov5m.pt"
    model = torch.hub.load("ultralytics/yolov5", "custom", path=ckpt_path, verbose=False)
    model.conf = 0.25
    model.iou = 0.45
    model.classes = [1]
    model.amp = False
    model = model.to(device)
    model.eval()
    return model

def detect_heads(image, model):
    detections = model(image, size=640).pred[0].cpu().numpy()[:, :-1]
    return detections

def load_gaze_model(ckpt_path, device):
    model = GaT(
        encoder=Swin3D(pretrained=False),
        head_dict=HeadDict(
            names=["gaze"],
            modules=[
                partial(
                    MLPHead,
                    hidden_dim=256,
                    num_layers=1,
                    out_features=3,
                )
            ],
        ),
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model

def preprocess_image(image, bbox):
    """Preprocess image crop for gaze estimation"""
    xmin, ymin, xmax, ymax = map(int, bbox)
    
    # Square bbox with padding
    w = xmax - xmin
    h = ymax - ymin
    size = max(w, h)
    
    # Add padding
    pad = int(size * 0.1)
    size = size + 2 * pad
    
    # Center crop
    cx = (xmin + xmax) // 2
    cy = (ymin + ymax) // 2
    
    xmin_new = max(0, cx - size // 2)
    ymin_new = max(0, cy - size // 2)
    xmax_new = min(image.shape[1], cx + size // 2)
    ymax_new = min(image.shape[0], cy + size // 2)
    
    # Crop and resize
    crop = image[ymin_new:ymax_new, xmin_new:xmax_new]
    crop = cv2.resize(crop, (224, 224))
    
    # Normalize
    crop = crop.astype(np.float32) / 255.0
    crop = (crop - IMG_MEAN) / IMG_STD
    
    # Convert to tensor [C, H, W]
    crop = torch.from_numpy(crop).permute(2, 0, 1).float()
    
    return crop

def draw_arrow2D(image, gaze, position, head_size, color=(255, 0, 0), thickness=10):
    length = head_size
    gaze_dir = gaze / np.linalg.norm(gaze)
    dx = -length * gaze_dir[0]
    dy = -length * gaze_dir[1]
    
    cv2.arrowedLine(
        image,
        tuple(np.round(position).astype(np.int32)),
        tuple(np.round([position[0] + dx, position[1] + dy]).astype(int)),
        color,
        thickness,
        cv2.LINE_AA,
        tipLength=0.2,
    )
    return image

def draw_gaze(image, head_bbox, head_pid, gaze, cmap, colors):
    img_h, img_w = image.shape[0], image.shape[1]
    scale = max(img_h, img_w) / 1920
    fs = 0.8 * scale
    thickness = int(scale * 10)
    thickness_gaze = int(scale * 10)
    
    xmin, ymin, xmax, ymax = map(int, head_bbox)
    
    # Head center and radius
    head_center = np.array([(xmin + xmax) // 2, (ymin + ymax) // 2])
    head_radius = max(xmax - xmin, ymax - ymin) // 2
    head_radius = int(head_radius * 1.2)
    
    color = colors[head_pid % len(colors)]
    cv2.circle(image, head_center, head_radius + 1, color, thickness)
    
    # Draw header
    header_text = f"P{int(head_pid)}"
    (w_text, h_text), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    header_ul = (
        int(head_center[0] - w_text / 2),
        int(head_center[1] - head_radius - 1 - thickness / 2),
    )
    header_br = (
        int(head_center[0] + w_text / 2),
        int(head_center[1] - head_radius - 1 + h_text + 5),
    )
    cv2.rectangle(image, header_ul, header_br, color, -1)
    cv2.putText(
        image,
        header_text,
        (header_ul[0], int(head_center[1] - head_radius - 1 + h_text)),
        cv2.FONT_HERSHEY_SIMPLEX,
        fs,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    
    # Draw gaze vector with color based on angle
    gaze_norm = gaze / np.linalg.norm(gaze)
    gaze_tensor = torch.tensor(gaze_norm)[None]
    target = torch.tensor([[0.0, 0.0, -1.0]])
    sim = F.cosine_similarity(gaze_tensor, target, dim=1, eps=1e-10)
    sim = F.hardtanh_(sim, min_val=-1.0, max_val=1.0)
    angle_gaze = torch.acos(sim)[0] * 180 / np.pi
    angle_gaze /= 180
    gaze_color = np.array(cmap(angle_gaze)[:3]) * 255
    
    image = draw_arrow2D(
        image=image,
        gaze=gaze_norm,
        position=head_center,
        head_size=head_radius,
        color=gaze_color,
        thickness=thickness_gaze,
    )
    
    return image

# ========================= Main Function =========================== #

def run_webcam_demo():
    print("=" * 50)
    print("Gaze Estimation Webcam Demo")
    print("=" * 50)
    print(f"Camera ID: {args.camera_id}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Device: {args.device}")
    print(f"Save output: {args.save_output}")
    print("=" * 50)
    
    # Load models
    print("Loading models...")
    gaze_model = load_gaze_model(args.ckpt_path, args.device)
    head_detector = load_head_detection_model(args.device)
    tracker = OCSORT()
    print("Models loaded successfully!")
    
    # Open webcam
    print(f"Opening webcam {args.camera_id}...")
    cap = cv2.VideoCapture(args.camera_id)
    
    if not cap.isOpened():
        print(f"Error: Cannot open webcam {args.camera_id}")
        return
    
    # Get webcam properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Webcam: {width}x{height} @ {fps}fps")
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current frame")
    print("=" * 50)
    
    # Video writer
    out = None
    if args.save_output:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = f"{args.output_dir}/webcam_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving output to: {output_path}")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break
            
            frame_count += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect heads
            raw_detections = detect_heads(frame_rgb, head_detector)
            detections = []
            for raw_detection in raw_detections:
                bbox, conf = raw_detection[:4], raw_detection[4]
                if conf > DET_THR:
                    cls_ = np.array([0.0])
                    detection = np.concatenate([bbox, conf[None], cls_])
                    detections.append(detection)
            
            if len(detections) > 0:
                detections = np.stack(detections)
                
                # Track
                tracks = tracker.update(detections, frame_rgb)
                
                # Process each tracked head
                for track in tracks:
                    bbox = track[:4]
                    pid = int(track[4])
                    
                    # Preprocess for gaze estimation
                    crop = preprocess_image(frame_rgb, bbox)
                    crop_batch = crop.unsqueeze(0).unsqueeze(0).to(args.device)  # [B, T, C, H, W]
                    
                    # Predict gaze
                    with torch.no_grad():
                        pred = gaze_model(crop_batch)
                        gaze = torch.nn.functional.normalize(pred["gaze"], p=2, dim=2, eps=1e-8)
                        gaze = gaze[0, 0].cpu().numpy()  # [3]
                    
                    # Draw on frame
                    frame = draw_gaze(frame, bbox, pid, gaze, CMAP, COLORS)
            
            # Add frame info
            cv2.putText(frame, f"Frame: {frame_count} | Heads: {len(detections) if len(detections) > 0 else 0}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display
            cv2.imshow('Gaze Estimation - Press Q to quit', frame)
            
            # Save if enabled
            if out is not None:
                out.write(frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                import os
                os.makedirs(args.output_dir, exist_ok=True)
                save_path = f"{args.output_dir}/frame_{frame_count}.jpg"
                cv2.imwrite(save_path, frame)
                print(f"Saved frame to: {save_path}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        print(f"\nProcessed {frame_count} frames")
        print("Done!")

if __name__ == "__main__":
    run_webcam_demo()
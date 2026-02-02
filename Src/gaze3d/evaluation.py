import os
import sys
import time
import torch
import psutil
from functools import partial

# ---------------- CPU benchmark config ----------------
torch.set_num_threads(1)
device = "cpu"

# ---------------- Import model ----------------

sys.path.insert(
    0,
    os.path.abspath("D:/AI/PBL4/Src/gaze3d/src")
)

from models.gat_model import GaT, HeadDict, MLPHead, Swin3D

# ---------------- Build model ----------------
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
).to(device)

# ---------------- Load checkpoint ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ckpt_path = os.path.join(
    BASE_DIR,
    "checkpoints",
    "gat_stwsge_gaze360_gf.ckpt"
)

checkpoint = torch.load(ckpt_path, map_location="cpu")

# Lightning checkpoint
model.load_state_dict(checkpoint["state_dict"], strict=True)
model.eval()

# ---------------- Params ----------------
params = sum(p.numel() for p in model.parameters())
print(f"Params: {params/1e6:.2f} M")

# ---------------- FPS ----------------
# Input shape GaT: [B, T, C, H, W]
dummy = torch.randn(1, 1, 3, 224, 224)

with torch.no_grad():
    # warm-up
    for _ in range(5):
        out = model(dummy)

    N = 50
    t0 = time.time()
    for _ in range(N):
        out = model(dummy)
    t1 = time.time()

fps = N / (t1 - t0)
print(f"FPS: {fps:.2f}")

# ---------------- RAM ----------------
ram = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
print(f"RAM: {ram:.1f} MB")

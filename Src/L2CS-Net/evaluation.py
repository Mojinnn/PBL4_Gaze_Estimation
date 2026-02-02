import os, sys, time, torch, psutil

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

L2CS_DIR = os.path.join(BASE_DIR, "l2cs")
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "L2CSNet_gaze360.pkl"
)

sys.path.insert(0, L2CS_DIR)

# ---------------- Imports ----------------
from model import L2CS
from torchvision.models.resnet import Bottleneck

device = "cpu"

# ---------------- Build model ----------------
model = L2CS(
    block=Bottleneck,
    layers=[3,4,6,3],
    num_bins=90
).to(device)

# ---------------- Load weights ----------------
print("Loading:", MODEL_PATH)
print("Exists:", os.path.exists(MODEL_PATH))

state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# Params
params = sum(p.numel() for p in model.parameters())
print(f"Params: {params/1e6:.2f} M")

# FPS
dummy = torch.randn(1,3,224,224)
for _ in range(10):
    model(dummy)

N = 100
t0 = time.time()
for _ in range(N):
    model(dummy)
t1 = time.time()

print(f"FPS: {1/((t1-t0)/N):.2f}")

# RAM
ram = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
print(f"RAM: {ram:.1f} MB")
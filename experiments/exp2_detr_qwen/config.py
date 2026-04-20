from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent.parent

# Paths
ANNO_FILE = str(REPO_ROOT / "road_waymo_trainval_v1.0.json")
FRAMES_DIR = str(REPO_ROOT / "road_waymo_frames")
CKPT_DIR = str(EXP_DIR / "checkpoints")
LOG_DIR = str(EXP_DIR / "logs")
EXP1B_CKPT = str(REPO_ROOT / "experiments" / "exp1b_road_r" / "checkpoints" / "best.pt")

# Model
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
VIT_DIM = 3584
D_MODEL = 256
NUM_QUERIES = 100
NUM_DECODER_LAYERS = 6
NHEAD = 8
DIM_FFN = 1024
DROPOUT = 0.1

# LoRA
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_N_LAYERS = 8
LORA_TARGET_MODULES = ["qkv", "proj"]

# Data
CLIP_LEN = 8
CLIP_STRIDE = 16
MIN_PIXELS = 448 * 448
MAX_PIXELS = 448 * 448

# Labels
N_AGENTS = 10
N_ACTIONS = 22
N_LOCS = 16
N_DUPLEXES = 49
N_TRIPLETS = 86

HEAD_SIZES = {
    "agent": N_AGENTS,
    "action": N_ACTIONS,
    "loc": N_LOCS,
    "duplex": N_DUPLEXES,
    "triplet": N_TRIPLETS,
}

# Hungarian cost
COST_CLASS = 2.0
COST_BBOX = 5.0
COST_GIOU = 2.0

# Loss
LAMBDA_CLS = 2.0
LAMBDA_BBOX = 5.0
LAMBDA_GIOU = 2.0
LAMBDA_TNORM = 1.0
EOS_COEF = 0.1
FOCAL_GAMMA = 2.0
NOOBJ_GAMMA = 2.0
TUBE_LINK_IOU = 0.3

# Training
BATCH_SIZE = 1
MAX_EPOCHS = 30
GRAD_ACCUM = 4
LR_LORA = 5e-5
LR_DECODER = 1e-4
LR_HEADS = 1e-4
WARMUP_STEPS = 500
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.01

# Inference
CONFIDENCE_THRESHOLD = 0.3
CLASS_THRESHOLD = 0.05


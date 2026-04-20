"""
Experiment 1b — ROAD-R: Qwen2.5-VL + LoRA + FCOS dense detection + focal loss + Gödel t-norm.

Architecture changes from Exp1:
  - FCOS-style dense detection: every ViT token predicts agentness + box + labels
  - No GT boxes at inference (paper-analogous to 3D-RetinaNet anchor-per-location)
  - Agentness focal loss (handles ~99% background tokens)
  - Box regression (SmoothL1 on FCOS ltrb targets)
  - Focal loss (γ=2, per-class α from train frequencies) on foreground tokens
  - Gödel t-norm with predicted agentness (not hardcoded 1.0)
  - LoRA on first 8 ViT blocks (r=8, alpha=16)
  - Best checkpoint by action head macro-mAP on foreground tokens
  - Warm-starts from Exp1 best.pt
"""

# ── Paths ──────────────────────────────────────────────────────────────────────
ANNO_FILE  = "/data/datasets/road_waymo/road_waymo_trainval_v1.1.json"
FRAMES_DIR = "/data/datasets/road_waymo/rgb-images"
CKPT_DIR   = "/data/repos/ROAD_Reason/experiments/exp1b_road_r/checkpoints"
LOG_DIR    = "/data/repos/ROAD_Reason/experiments/exp1b_road_r/logs"

# Warm-start: load Exp1 best checkpoint before adding LoRA
EXP1_CKPT  = "/data/repos/ROAD_Reason/experiments/exp1_road_r/checkpoints/best.pt"

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen2.5-VL-7B-Instruct"
VIT_DIM     = 3584   # LLM hidden size (merger output dim); 3584 for 7B
FREEZE_VIT  = False  # LoRA handles partial unfreezing

# ── LoRA ───────────────────────────────────────────────────────────────────────
LORA_R               = 8
LORA_ALPHA           = 16
LORA_DROPOUT         = 0.05
LORA_TARGET_MODULES  = ["qkv", "proj"]   # QKV projection + output projection in each ViT block
LORA_N_LAYERS        = 8                 # apply to first 8 of 32 blocks only

# ── Data ───────────────────────────────────────────────────────────────────────
CLIP_LEN    = 8
MIN_PIXELS  = 448 * 448
MAX_PIXELS  = 448 * 448
TRAIN_SPLIT = "train"
VAL_SPLIT   = "val"
CLIP_STRIDE = 16

# ── Label dims ─────────────────────────────────────────────────────────────────
N_AGENTS    = 10
N_ACTIONS   = 22
N_LOCS      = 16
N_DUPLEXES  = 49
N_TRIPLETS  = 86

# ── Tube-linking module ────────────────────────────────────────────────────────
TUBE_N_HEADS = 8

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE   = 1
LR_LORA      = 5e-5    # LoRA adapters (starting from zero)
LR_HEADS     = 1e-4    # tube_link + classification heads (pre-trained from Exp1)
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500     # longer warmup — LoRA starts from zero
MAX_EPOCHS   = 15
GRAD_CLIP    = 1.0
DTYPE        = "bfloat16"

# ── Loss ───────────────────────────────────────────────────────────────────────
FOCAL_GAMMA      = 2.0
TNORM_TYPE       = "godel"      # paper Table 7 best; also verified in local replication
LAMBDA_TNORM     = 1.0          # increased from 0.1 — real violations expected with LoRA
LAMBDA_BOX       = 1.0          # weight for box regression loss
AGENTNESS_GAMMA  = 2.0          # focal γ for agentness head (same as RetinaNet)

# ── Detection / inference ───────────────────────────────────────────────────────
AGENTNESS_THRESHOLD = 0.5       # threshold for foreground filtering at inference
NMS_IOU_THRESHOLD   = 0.5       # NMS IoU threshold

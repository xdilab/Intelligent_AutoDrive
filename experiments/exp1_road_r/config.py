"""
Experiment 1 — ROAD-R: Qwen2.5-VL ViT encoder + classification heads + t-norm loss.

Training is classification-only in this experiment: GT boxes are used for ROI-pooling
(no learned detection head). The ViT is frozen; only the tube-linking module and
classification heads are trained.
"""

# ── Paths ──────────────────────────────────────────────────────────────────────
ANNO_FILE  = "/data/datasets/road_waymo/road_waymo_trainval_v1.1.json"
FRAMES_DIR = "/data/datasets/road_waymo/rgb-images"
CKPT_DIR   = "/data/repos/ROAD_Reason/experiments/exp1_road_r/checkpoints"
LOG_DIR    = "/data/repos/ROAD_Reason/experiments/exp1_road_r/logs"

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen2.5-VL-7B-Instruct"
# VIT_DIM is the *merger output* dimension = LLM hidden size (not the ViT's internal
# context_dim=1280). The PatchMerger MLP projects 4×context_dim → dim=LLM_hidden_size.
# Qwen2.5-VL-7B: LLM hidden size = 3584. Qwen2.5-VL-3B: 2048.
VIT_DIM     = 3584
FREEZE_VIT  = True   # freeze ViT weights; set False when LoRA is added (Phase 1b)

# ── Data ───────────────────────────────────────────────────────────────────────
CLIP_LEN    = 8      # consecutive annotated frames per training sample
# Frame size is controlled by the Qwen image processor (dynamic resolution).
# For Qwen2.5-VL, min_pixels / max_pixels bound the total token count.
# 448×448 → 16×16 = 256 merged tokens per frame (after 2× merger).
MIN_PIXELS  = 448 * 448          # ~200K — enforce at least this resolution
MAX_PIXELS  = 448 * 448          # same ceiling for consistent spatial grids
TRAIN_SPLIT = "train"
VAL_SPLIT   = "val"
CLIP_STRIDE = 16  # step between clip start points (frames); 16 = ~2x clip length, minimal overlap

# ── Label dims ─────────────────────────────────────────────────────────────────
N_AGENTS    = 10
N_ACTIONS   = 22
N_LOCS      = 16
N_DUPLEXES  = 49
N_TRIPLETS  = 86

# ── Tube-linking module ────────────────────────────────────────────────────────
TUBE_N_HEADS = 8   # multi-head attention heads in TubeLinkingModule

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE   = 1    # one clip at a time (variable box counts; increase with padding later)
LR           = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 200
MAX_EPOCHS   = 10
GRAD_CLIP    = 1.0
DTYPE        = "bfloat16"   # bfloat16 for ViT; heads run in float32

# ── Loss ───────────────────────────────────────────────────────────────────────
TNORM_TYPE    = "lukasiewicz"   # per email; paper Table 7 says godel is best — compare both
LAMBDA_TNORM  = 0.1             # weight for constraint violation loss

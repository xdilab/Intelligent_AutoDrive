"""Exp4 — 3D-RetinaNet (frozen) + MoonViT (frozen) + Q-Former late fusion.

Architecture rationale:
  exp2 series (c–g) fused CLIP ViT-L/14@336 as a 4th encoder level alongside
  R50 FPN P3/P4/P5. Two structural problems (see findings/exp2-series-narrative
  §"Root Cause #4"):
    1. Spatial mismatch — R50 sees 800x1333, CLIP sees 336x336 squashed.
    2. Frozen encoder can't adapt — DINO pretrain never saw CLIP-shaped tokens.

  Exp4 fixes both by moving fusion *after* detection:
    - Frozen 3D-RetinaNet produces tube proposals at native resolution.
    - Frozen MoonViT processes the same native-res frames (2D RoPE handles any HxW).
    - Q-Former cross-attention fuses tube features (Q) with MoonViT patch features (KV).
    - Flat 184-dim head + focal-on-all (exp2f's negative-supervision fix).

  Smoke-test encoder is SigLIP-SO400M (the base MoonViT was init'd from — same
  arch / patch=14 / hidden=1152). Real training swaps in moonshotai/Kimi-VL-A3B
  vision_tower; gated by ENCODER_NAME below.
"""

from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent.parent
CKPT_DIR = str(EXP_DIR / "checkpoints")
LOG_DIR = str(EXP_DIR / "logs")

# ---- Data ----
ANNO_FILE = "/data/datasets/ROAD_plusplus/road_waymo_trainval_v1.1.json"
FRAMES_DIR = "/data/datasets/ROAD_plusplus/rgb-images"
CLIP_LEN = 8
CLIP_STRIDE = 16

# ---- Frozen detector (3D-RetinaNet baseline, epoch 25 = 17.76% agent f-mAP) ----
RETINANET_CKPT = (
    "/data/repos/PedestrianIntent++/ROAD_plus_plus_Baseline/output/"
    "road_waymo/cache/resnet50I3D600-Pkinetics-b4s8x1x1-road_waymo-alltn-h3x3x3/"
    "model_000025.pth"
)
RETINANET_SPATIAL_DIM = 256            # head_size in baseline
RETINANET_TOPK_TUBES = 100              # top-K post-NMS proposals kept per clip

# ---- Frozen semantic encoder ----
# MoonViT (Kimi-VL vision_tower) at native clip resolution.
#
# Used like NVIDIA's LocateAnything: input at native (H, W), 2D RoPE handles
# the non-square grid, pixel-shuffle 2x2 built into MoonViT's forward bundles
# 4 sub-patches per output token. Output token dim = 4 × hidden = 4608.
#
# Pre-merger grid at 600x840 = (42, 60) = 2520 patches/frame.
# Post-merger ("merged") grid at 600x840 = (21, 30) = 630 tokens/frame.
# Tokens carry inner shape (4, 1152) which we flatten to 4608 for our consumer.
#
# Future Qwen-as-decoder consumer (LocateAnything-style) attaches downstream of
# this same grid output — wrapper interface does not change.
# Architecture: standalone MoonViT-SO-400M (clean modeling file, no LLM cruft).
# Weights: NVIDIA's LocateAnything-3B vision_model — same architecture but
# continual-pretrained for visual grounding (max_abs_diff up to 0.06 vs base).
# This is exactly the encoder NVIDIA's LocateAnything uses, with the
# grounding-specialization the paper trained for.
ENCODER_ARCH_REPO    = "moonshotai/MoonViT-SO-400M"
ENCODER_WEIGHTS_REPO = "nvidia/LocateAnything-3B"
ENCODER_WEIGHTS_FILE = "model-00001-of-00002.safetensors"
ENCODER_WEIGHTS_PREFIX = "vision_model."             # LA-3B prefix on MoonViT weights
ENCODER_NAME = ENCODER_WEIGHTS_REPO                   # display label
ENCODER_HIDDEN = 1152                              # MoonViT internal hidden dim
ENCODER_PATCH = 14
ENCODER_MERGE = 2                                  # pixel-shuffle 2x2
ENCODER_TOKEN_DIM = ENCODER_HIDDEN * (ENCODER_MERGE ** 2)  # 4608 per merged token

# ---- Q-Former fusion ----
# Single query is the information-theoretic match for our 6-token KV (1 spatial + 4
# RoI'd semantic + 1 global). BLIP-2 uses 32 because it compresses 257 ViT tokens.
# 4 layers is the depth budget — interpolation between BLIP-2's 12 (high-compression)
# and Frozen-DETR's 6 (large KV); ours has small KV so we don't need much depth.
# Dropout 0.1 because the trainable surface (~3.5M params) is small relative to ROAD-Waymo
# tube count (~25k); 0.0 invites overfit.
D_MODEL = 256
N_QUERY_TOKENS = 1
NUM_FUSION_LAYERS = 4
NHEAD = 8
DIM_FFN = 1024
DROPOUT = 0.1
USE_GLOBAL_TOKEN = True                 # mean-pooled MoonViT patches as 6th KV token
USE_TEMPORAL_SELFATTN = True            # single TransformerEncoderLayer over T after Q-Former

# ---- Training (per plan §G) ----
# All values traceable to exp2f_flat_head/config.py and exp2g_msdetr/config.py
# (replicate-before-innovate) with two principled deviations: dropout 0.1 and bf16.
BATCH_SIZE = 1
GRAD_ACCUM = 4                          # effective batch = 4 clips/step
MAX_EPOCHS = 30
LR = 2e-4                               # Q-Former + temporal + head — all fresh-init
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 500
GRAD_CLIP = 0.1
LR_DROP_EPOCH = 20                      # step decay
LR_DROP_FACTOR = 0.1
MIXED_PRECISION = "bf16"                # A6000 native; SigLIP+MoonViT trained in bf16

# ---- Loss (per plan §F) ----
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
TNORM_START_EPOCH = 10                  # curriculum: focal-on-all alone for ep 1-10
LAMBDA_TNORM = 0.5                      # half exp2f's 1.0; sweep {0.25, 0.5, 1.0} if plateaus
TNORM_TYPE = "godel"                    # best per ROAD-R Table 7

# ---- Output head (flat 184-dim, lesson from exp2f) ----
N_AGENTS = 10
N_ACTIONS = 22
N_LOCS = 16
N_DUPLEXES = 49
N_TRIPLETS = 86
NUM_CLASSES = 1 + N_AGENTS + N_ACTIONS + N_LOCS + N_DUPLEXES + N_TRIPLETS  # 184
NUM_CLASSES_LIST = [1, N_AGENTS, N_ACTIONS, N_LOCS, N_DUPLEXES, N_TRIPLETS]

CLS_OFFSETS = {
    "agentness": 0,
    "agent":     1,
    "action":    1 + N_AGENTS,                                    # 11
    "loc":       1 + N_AGENTS + N_ACTIONS,                        # 33
    "duplex":    1 + N_AGENTS + N_ACTIONS + N_LOCS,               # 49
    "triplet":   1 + N_AGENTS + N_ACTIONS + N_LOCS + N_DUPLEXES,  # 98
}

# ---- Resolution (baseline trained at MIN_SIZE=600, MAX=840) ----
# Smoke test runs at baseline-matched resolution to keep frozen detector's
# behaviour in its training distribution. MoonViT handles whatever it's given
# via 2D RoPE; SigLIP smoke-test wrapper resizes per-frame to its own 384px.
VAL_SHORT_SIDE = 600
VAL_MAX_SIZE = 840

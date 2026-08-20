"""Exp9 — joint round-robin training with per-corpus losses (see DESIGN.md).

The three open Moradi choices are config flags so his answers are one-line
changes, not rewrites:
  COVLA_ENABLED   — (1) CoVLA in round one?          default True (exp8 precedent)
  ROAD_INPUT_MODE — (2) 'drawn' (box-overlay renders, cached-prompt lineage,
                    reuses exp8's cache/road_frames) vs 'clean' (raw frame,
                    RoI-only).                        default 'drawn'
  LANG_LR_SCALE   — (3) LR multiplier on the language legs' LoRA updates.
                    default 1.0 (single LR)

Run with base miniconda python (/home/brandon/miniconda3/bin/python).
"""

from __future__ import annotations

from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
CKPT_DIR = EXP_DIR / "checkpoints"
LOG_DIR = EXP_DIR / "logs"

EXP8_DIR = EXP_DIR.parent / "exp8_joint_lora"
EXP6_DIR = EXP_DIR.parent / "exp6_detection_steered"
EXP5_DIR = EXP_DIR.parent / "exp5_qwen_reasoning"
EXP4_DIR = EXP_DIR.parent / "exp4_retinamoon"
EXP1_DIR = EXP_DIR.parent / "exp1_road_r"

# ---- Corpora --------------------------------------------------------------
COVLA_ENABLED = True
BDDX_TRAIN_JSON = EXP_DIR.parent / "exp7_bddx_lora" / "cache" / "bddx_sft_train.json"
COVLA_TRAIN_JSON = EXP8_DIR / "cache" / "covla_sft_train.json"

# ROAD leg: detector boxes + GT from exp6's train dump; same 2/12-shard subset
# (9,596 frames) as the exp8/Stage-1 pipeline so results stay comparable.
ROAD_DET_TRAIN_PKL = EXP6_DIR / "cache" / "detections_train.pkl"
ROAD_DET_VAL_PKL = EXP6_DIR / "cache" / "detections_val.pkl"
ROAD_TRAIN_SHARDS = (0, 1)
ROAD_NUM_SHARDS = 12
ROAD_INPUT_MODE = "drawn"                       # 'drawn' | 'clean'
ROAD_DRAWN_DIR = EXP8_DIR / "cache" / "road_frames" / "train"
ROAD_FRAMES_DIR = "/data/datasets/ROAD_plusplus/rgb-images"
ROAD_BOX_H, ROAD_BOX_W = 600, 840               # detections pkl box space
MAX_BOXES_PER_FRAME = 40
IOU_MATCH_THRESH = 0.5

# ---- Label space (matches exp4/exp5/exp6) ---------------------------------
N_AGENTS, N_ACTIONS, N_LOCS, N_DUPLEXES, N_TRIPLETS = 10, 22, 16, 49, 86
NUM_CLASSES = 1 + N_AGENTS + N_ACTIONS + N_LOCS + N_DUPLEXES + N_TRIPLETS  # 184
ANNO_FILE = "/data/datasets/ROAD_plusplus/road_waymo_trainval_v1.1.json"

# ---- Constraints (corrected — NEVER the JSON childs arrays) ---------------
CONSTRAINTS_JSON = EXP_DIR / "constraints_verified.json"
TNORM = "godel"                                  # ROAD-R Table 7 best
TNORM_LAMBDA = 0.0                               # R1/R3: 0.0; R2/R4 sweep {0.1, 1.0, 10.0}

# ---- Model ----------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
# LoRA covers BOTH the ViT attention (shared with the ROAD leg, which forwards
# through the visual encoder only) and the LLM self-attention (language legs).
# The ViT LoRA is the shared trainable surface that lets language co-training
# affect ROAD performance (R1→R3); heads are ROAD-leg-only.
LORA_TARGET_REGEX = (
    r".*visual\.blocks\.\d+\.attn\.(qkv|proj)$"
    r"|.*self_attn\.(q_proj|k_proj|v_proj|o_proj)$"
)
D_VIT = 3584                                     # merged visual token dim: the 7B
                                                 # merger projects into LLM hidden
                                                 # width (verified empirically,
                                                 # transformers 5.3; exp1's 1280
                                                 # docstring predates this)

# ---- Training -------------------------------------------------------------
LR_HEAD = 2e-4                                   # fresh flat head (exp6 precedent)
LR_LORA = 1e-5                                   # exp7/exp8 LoRA LR
LANG_LR_SCALE = 1.0
# Moradi 2026-08-18: "new heads start random and need to move fast, may
# destabilize existing pretrained ... give new heads some time, to train, then
# include others." Warmup = first N cycles run the ROAD leg only with the LoRA
# group frozen (heads-only updates); language legs join after. N grounded in
# R1's curve: focal fell 0.0060 -> 0.0033 by cycle ~600-1000 (~10% of an epoch),
# flat after until the epoch-2 drop.
HEAD_WARMUP_CYCLES = 1000
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
EPOCHS = 3                                       # epoch = one pass over ROAD subset
FOCAL_GAMMA = 2.0                                # alphas from exp4 compute_flat_alphas
MODEL_MAX_LENGTH = 8192
MIN_PIXELS = 784
MAX_PIXELS = 501760                              # exp8: box ids legible, deploy-matched
LOG_EVERY = 20                                   # cycles
SEED = 17

"""Exp6 — Detection-steered fusion: Qwen language embedding x RetinaNet logits.

Stage 1 (trained fusion) of the VLM-reasoning-layer direction (Approach 8).
Provenance — Dr. Moradi, 2026-06-02: "use language embedding to fused with
output of 3d retina net and then re predict with adding few layer to be
trained"; 2026-06-11: "Qwen can be reasoning given detections from retinaNet
... not from ground truth ... it is psedu-label" and "fusion layers ... should
be trained on training [split], test always the same in all experiments".

Pipeline (each stage cached; only the fusion head trains):
  dump_detections.py  -> cache/detections_<split>.pkl   (boxes + scores + 184-d
                                                         logits + GT per frame)
  qwen_infer.py (exp5) -> exp5 cache/qwen/<video>/<fid>.json  (val already
                          cached NMS-fixed; train split must be run with exp6's
                          detections pkl via --detections)
  embed_rationale.py  -> cache/rationale_<split>.pkl    (frozen SigLIP text
                                                         embedding per box)
  train.py            -> checkpoints/                   (fusion head only)
  eval.py             -> per-head f-mAP (baseline evaluator, IoU=0.5)

Floor guarantee: the head is residual with a zero-init last layer, so at init
the output IS the detector's logits — the fused model cannot start below the
detector-over-top-K control (eval.py --detector-only).
"""

from __future__ import annotations

from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
CACHE_DIR = EXP_DIR / "cache"
CKPT_DIR = EXP_DIR / "checkpoints"
LOG_DIR = EXP_DIR / "logs"

# ---- Data (matches exp4/exp5) ----
ANNO_FILE = "/data/datasets/ROAD_plusplus/road_waymo_trainval_v1.1.json"
FRAMES_DIR = "/data/datasets/ROAD_plusplus/rgb-images"
CLIP_LEN = 8
TRAIN_STRIDE = 16                       # exp4 train protocol (CLIP_STRIDE=16)
VAL_STRIDE = 32                         # baseline val protocol: SEQ_LEN * 4
VAL_SHORT_SIDE = 600                    # detector-matched resize (h)
VAL_MAX_SIZE = 840                      # detector-matched resize (w)

# ---- Frozen detector (17.76% agent f-mAP, epoch-25 ckpt) ----
RETINANET_CKPT = (
    "/data/repos/PedestrianIntent++/ROAD_plus_plus_Baseline/output/"
    "road_waymo/cache/resnet50I3D600-Pkinetics-b4s8x1x1-road_waymo-alltn-h3x3x3/"
    "model_000025.pth"
)
RETINANET_SPATIAL_DIM = 256
RETINANET_TOPK_TUBES = 100             # top-K post-NMS proposals kept per clip

# ---- ROAD-Waymo class structure (flat 184-dim, mirrors exp4) ----
# RetinaNetWrapper (exp4/model.py) reads these off the shared `config` module
# when building the baseline args, so they must match exp4 exactly.
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

# ---- Qwen cache (exp5's NMS-fixed cache; boxes are deterministic re-runs of
# the same frozen detector, so exp6's dumped boxes match the ones Qwen saw) ----
EXP5_DIR = EXP_DIR.parent / "exp5_qwen_reasoning"
QWEN_CACHE_DIR = EXP5_DIR / "cache" / "qwen"
MAX_BOXES_PER_FRAME = 40               # exp5 cap: Qwen classified the top 40

# ---- Language embedding ----
# Structured fields, vectorized exactly as exp5 parses them (vlm_io.py):
#   agent one-hot(10) + agent-null(1) + actions multi-hot(22) +
#   locations multi-hot(16) + risk one-hot(3) + risk-null(1) + parsed-flag(1)
STRUCT_DIM = (N_AGENTS + 1) + N_ACTIONS + N_LOCS + (3 + 1) + 1     # 54
RISK_LEVELS = ("low", "medium", "high")

# Rationale sentence -> frozen SigLIP text tower (same SigLIP-SO400M already
# used as exp4's smoke encoder; present in the local HF cache — no new deps).
# SigLIP text was trained with padding="max_length", max_length=64; a one-
# sentence rationale fits well inside that window.
RATIONALE_ENCODER = "google/siglip-so400m-patch14-384"
RATIONALE_DIM = 1152
RATIONALE_MAX_LEN = 64
RATIONALE_BATCH = 256

# ---- Fusion head (the ONLY trainable module) ----
D_MODEL = 256                          # per-track projection width
D_FFN = 512                            # hidden width of the few-layer head
DROPOUT = 0.1                          # exp4 precedent for small heads

# ---- Training ----
# LR/WD/focal follow exp2f/exp4 (replicate-before-innovate). Batch size is
# larger than clip-based experiments because samples here are cached vectors
# (one box on one frame), not video clips — no GPU-memory constraint.
BATCH_SIZE = 256
MAX_EPOCHS = 20
LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 0.1
FOCAL_GAMMA = 2.0                      # alphas come from compute_flat_alphas
IOU_MATCH_THRESH = 0.5                 # GT assignment, standard ROAD-Waymo IoU

# ---- Reuse exp4's detector wrapper + exp5's parsing ----
EXP4_DIR = EXP_DIR.parent / "exp4_retinamoon"
BASELINE_ROOT = "/data/repos/PedestrianIntent++/ROAD_plus_plus_Baseline"

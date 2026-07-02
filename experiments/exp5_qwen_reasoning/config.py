"""Exp5 — Vanilla Qwen2.5-VL reasoning over frozen 3D-RetinaNet detections.

Stage 1 (zero-shot) of the VLM-reasoning-layer direction (Approach 8):
  Frozen 3D-RetinaNet (17.76% agent f-mAP) emits top-K *predicted* boxes per
  clip. Those pseudo-label boxes are drawn on each frame and handed to a frozen,
  out-of-the-box Qwen2.5-VL-7B-Instruct with a structured prompt. Qwen classifies
  each box (agent / action / location / risk / rationale). We map its JSON back
  onto the box coords (RetinaNet box score = detection confidence) and run the
  baseline's frame-level f-mAP evaluator.

  NOTE: this scores Qwen over the top-K detector boxes, NOT all anchors. It is the
  *untrained-VLM floor* Moradi asked for (every field asked even zero-shot so any
  later fine-tuning lift is attributable) — not a reproduction of the 17.76%
  all-anchor baseline.

Pipeline (each stage cached so reruns are cheap):
  dump_detections.py  → cache/detections_<split>.pkl   (boxes + scores + GT)
  qwen_infer.py       → cache/qwen/<video>/<fid>.json   (one file per frame)
  eval_qwen.py        → per-head f-mAP table
"""

from __future__ import annotations

from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
CACHE_DIR = EXP_DIR / "cache"
QWEN_CACHE_DIR = CACHE_DIR / "qwen"
# Detection-steered prompting (per Moradi update email 2026-07-02): the
# detector's per-box class predictions go into the prompt as priors and Qwen
# verifies/refines instead of classifying from scratch. Separate cache so the
# plain zero-shot run stays untouched for comparison.
QWEN_STEERED_CACHE_DIR = CACHE_DIR / "qwen_steered"
LOG_DIR = EXP_DIR / "logs"

# ---- Data (matches exp4) ----
ANNO_FILE = "/data/datasets/ROAD_plusplus/road_waymo_trainval_v1.1.json"
FRAMES_DIR = "/data/datasets/ROAD_plusplus/rgb-images"
CLIP_LEN = 8
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

# ---- Qwen ----
QWEN_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
# Sized from the cache: real per-box objects (with rationale) run ~145 chars
# median / 180 p95 at ~3.1 chars/token, so 40 boxes need ~2.3k tokens. 1536 cut
# the median frame off at ~32/40 boxes (65% of frames truncated). 3072 covers 40
# boxes at p95 with ~30% headroom.
QWEN_MAX_NEW_TOKENS = 3072
# Cap boxes drawn/asked per frame — keeps the prompt legible and generation
# bounded. Boxes are pre-sorted by detector score, so we keep the strongest.
MAX_BOXES_PER_FRAME = 40

# ---- Reuse exp4's detector + dataloader code ----
EXP4_DIR = EXP_DIR.parent / "exp4_retinamoon"
EXP1_DIR = EXP_DIR.parent / "exp1_road_r"
BASELINE_ROOT = "/data/repos/PedestrianIntent++/ROAD_plus_plus_Baseline"

"""Exp7 — LoRA SFT of Qwen2.5-VL-7B-Instruct on BDD-X (Approach 8, Stage 2, BDD-X leg).

Provenance: Dr. Moradi 2026-07-24 — the Stage-1 zero-shot fusion (exp6) did not
beat the detector-only control (agent 14.6 floor); that run is kept as
"Baseline Fusion — Zero-shot VLM" and the approved next step is specialized
VLM training on BDD-X then CoVLA (direction page:
wiki/directions/vlm-reasoning-layer.md, Stage 2). This experiment is the BDD-X
leg only: LoRA-tune Qwen2.5-VL-7B-Instruct on (action, justification) pairs so
the driving-tuned VLM can replace the zero-shot one in exp5's caching pipeline
(qwen_infer.py) and exp6's fusion head can be retrained on the new cache.

Reference implementation (lab rule: reference repo first):
  reference/Qwen2.5-VL/qwen-vl-finetune  (QwenLM/Qwen2.5-VL, official vendor
  finetuning recipe; commit 9658872, 2026-01-30).
LoRA hyperparams below mirror the reference exactly; every deviation is listed
in README.md ("Deviations from reference") with its reason.

Run with the base miniconda python (/home/brandon/miniconda3/bin/python,
torch 2.9.0+cu128, transformers 5.3.0, peft 0.19.0, flash_attn 2.8.3) — NOT
the repo's .conda-envs/road_reason env.
"""

from __future__ import annotations

from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
CACHE_DIR = EXP_DIR / "cache"
CKPT_DIR = EXP_DIR / "checkpoints"
LOG_DIR = EXP_DIR / "logs"

# ---- Reference repo (cloned, gitignored; see README for clone command) ----
REFERENCE_ROOT = EXP_DIR / "reference" / "Qwen2.5-VL" / "qwen-vl-finetune"

# ---- BDD-X raw data (local; videos are NOT local — see README "Blockers") ----
BDDX_DIR = Path("/data/datasets/BDD-X")
BDDX_CSV = BDDX_DIR / "BDD-X-Annotations_v1.csv"
BDDX_SPLIT_FILES = {
    "train": BDDX_DIR / "train.txt",
    "val": BDDX_DIR / "val.txt",
    # test.txt exists but is deliberately unused: the downstream fixed test set
    # for the whole Approach-8 series is ROAD-Waymo val, never BDD-X test.
}

# BDD100K still frames (the only locally-available imagery for BDD-X videos).
# Each BDD100K image is the frame at the 10th second of its 40 s source video
# (Yu et al., "BDD100K", CVPR 2020 — the 100K images are sampled at the 10th
# second). BDD-X split stems are BDD100K video names, so <stem>.jpg is the
# t=10 s frame of that exact video. Frames live in the local YOLOPX copy of
# BDD100K; BDD-X splits do not align with BDD100K splits, so index all three.
BDD100K_IMG_DIRS = [
    Path("/data/datasets/bdd100k-yolopx/images/train"),
    Path("/data/datasets/bdd100k-yolopx/images/val"),
    Path("/data/datasets/bdd100k-yolopx/images/test"),
]
FRAME_TIME_S = 10.0  # timestamp of the available frame within each video

# Temporal-alignment filter: keep only CSV segments whose [start, end] window
# contains FRAME_TIME_S, so the frame the model sees is from *inside* the
# annotated action segment. (exp3_bddx paired every segment of a video with
# the single 10 s frame — up to ~30 s of temporal mismatch; that label noise
# is exactly what this filter removes. Yield: 4,339 train / 545 val samples.)
SFT_TRAIN_JSON = CACHE_DIR / "bddx_sft_train.json"
SFT_VAL_JSON = CACHE_DIR / "bddx_sft_val.json"
SFT_STATS_JSON = CACHE_DIR / "bddx_sft_stats.json"

# ---- Model ----
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"  # same id exp5/qwen_infer.py loads

# ---- LoRA (provenance: reference qwenvl/train/train_qwen.py L163-178 +
#      qwenvl/train/argument.py defaults; dropout is `0.0 or 0.05` -> 0.05) ----
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # LLM attention
LORA_BIAS = "none"
# NOTE: under --lora_enable the reference freezes *everything* else — vision
# tower AND merger stay frozen (set_model() is skipped). We match that.

# ---- Trainer args (provenance: reference scripts/sft_30a3b_lora.sh — the
#      repo's only LoRA recipe; its trainer-level values are model-agnostic) ----
LR = 1e-5
PER_DEVICE_BS = 1
GRAD_ACCUM = 4               # effective batch 4
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0
MODEL_MAX_LENGTH = 8192
MIN_PIXELS = 784             # 1 x 28 x 28   (reference script value)
MAX_PIXELS = 50176           # 64 x 28 x 28  (reference script value)
DATA_FLATTEN = True          # reference LoRA script; needs flash-attn varlen
ATTN_IMPLEMENTATION = "flash_attention_2"  # reference default (flash_attn 2.8.3 installed)

# Deviations (reasons in README):
NUM_EPOCHS = 3               # ref: 0.5 (multi-million-sample corpus; ours is 4,339)
LOGGING_STEPS = 10           # ref: 1 (per-step spam; 10 keeps tail -f readable)
SAVE_STRATEGY = "epoch"      # ref: steps/1000, limit 1 (no room for val-loss selection)
EVAL_STRATEGY = "epoch"      # ref: "no" (lab rule 4: BDD-X val = SFT val loss only)
PER_DEVICE_EVAL_BS = 1

# ---- SFT chat format ------------------------------------------------------
# Keeps the tuned model in exp5's dialect: schema-constrained prompt, STRICT
# JSON object as the target (mirrors exp5/vlm_io.py build_prompt style). The
# target keys follow the direction page's field mapping: BDD-X action ->
# `action`, BDD-X justification -> `justification` (the rationale-style field).
USER_PROMPT = """You are a driving-scene reasoning assistant. The image is a dashcam frame from the ego vehicle, captured mid-drive.
State what the ego vehicle is doing right now (its driving action) and justify it from visible scene evidence.

Rules:
- Output STRICT JSON only. No prose, no markdown fences.

Output schema:
{"action": "<what the ego vehicle is doing>", "justification": "<why, one short clause>"}"""


def build_target(action: str, justification: str) -> str:
    """Assistant-turn target: strict JSON, annotator text verbatim."""
    import json

    return json.dumps(
        {"action": action, "justification": justification}, ensure_ascii=False
    )

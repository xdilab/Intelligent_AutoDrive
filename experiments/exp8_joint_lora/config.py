"""Exp8 — Joint LoRA SFT of Qwen2.5-VL-7B on ROAD-Waymo + BDD-X + CoVLA.

Provenance: Dr. Moradi (relayed 2026-07-27) — Stage-2 VLM training is JOINT,
not sequential: interleave the three corpora so each optimizer step updates
the weights with one sample from each dataset ("back and forth"). This matches
the long-approved corpus order ROAD-R → BDD-X → CoVLA → *joint*
(wiki/directions/vlm-reasoning-layer.md, "Re-validated") and supersedes the
BDD-X-only leg (exp7, kept on disk as a potential ablation).

Recipe inherits exp7 (reference: QwenLM/Qwen2.5-VL qwen-vl-finetune, cloned at
exp7_bddx_lora/reference). Deviations from exp7 are marked EXP8 below and
justified in README.md.

Run with base miniconda python (/home/brandon/miniconda3/bin/python).
"""

from __future__ import annotations

import json

from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
CACHE_DIR = EXP_DIR / "cache"
CKPT_DIR = EXP_DIR / "checkpoints"
LOG_DIR = EXP_DIR / "logs"

EXP7_DIR = EXP_DIR.parent / "exp7_bddx_lora"
EXP6_DIR = EXP_DIR.parent / "exp6_detection_steered"
EXP5_DIR = EXP_DIR.parent / "exp5_qwen_reasoning"
REFERENCE_ROOT = EXP7_DIR / "reference" / "Qwen2.5-VL" / "qwen-vl-finetune"

# ---- Source data ----------------------------------------------------------
# BDD-X: reuse exp7's temporally-aligned SFT jsons verbatim (4,339 / 545).
BDDX_TRAIN_JSON = EXP7_DIR / "cache" / "bddx_sft_train.json"
BDDX_VAL_JSON = EXP7_DIR / "cache" / "bddx_sft_val.json"

# ROAD-Waymo: frames + frozen-detector boxes + GT from exp6's train dump.
ROAD_DET_PKL = EXP6_DIR / "cache" / "detections_train.pkl"
ROAD_FRAMES_DIR = "/data/datasets/ROAD_plusplus/rgb-images"
# Deployment-matched geometry (exp5): boxes are stored in 600x840 space.
ROAD_BOX_H, ROAD_BOX_W = 600, 840
MAX_BOXES_PER_FRAME = 40               # exp5 cap
IOU_MATCH_THRESH = 0.5                 # exp6 GT-assignment rule
# Frame subsets by sorted-key index modulo 12 (the sharding arithmetic already
# used for the zero-shot cache): train = shards {0,1} (same 9,596-frame subset
# the Stage-1 fusion pipeline caches), SFT-val = shard 11 (disjoint; the
# ROAD-Waymo *val split* stays untouched — it is the downstream test set).
ROAD_TRAIN_SHARDS = (0, 1)
ROAD_VAL_SHARD = 11
ROAD_NUM_SHARDS = 12
ROAD_VAL_N = 500                       # evenly subsampled from shard 11

# CoVLA (mini release: 50 scenes x 600 frames @ 20 Hz, per-frame captions).
COVLA_DIR = Path("/data/datasets/CoVLA/mini")
COVLA_CAPTIONS_DIR = COVLA_DIR / "captions"
COVLA_IMAGES_DIR = COVLA_DIR / "images"
COVLA_FRAME_STRIDE = 12                # every 0.6 s — captions barely change faster
COVLA_VAL_SCENES = 5                   # last 5 of the sorted scene list held out

# ---- Prepared SFT artifacts ----------------------------------------------
ROAD_IMG_DIR = CACHE_DIR / "road_frames"          # box-overlaid renders
COVLA_IMG_DIR = CACHE_DIR / "covla_frames"        # extracted pngs
ROAD_TRAIN_JSON = CACHE_DIR / "road_sft_train.json"
ROAD_VAL_JSON = CACHE_DIR / "road_sft_val.json"
COVLA_TRAIN_JSON = CACHE_DIR / "covla_sft_train.json"
COVLA_VAL_JSON = CACHE_DIR / "covla_sft_val.json"
# Combined files: block-concatenated [bddx | covla | road]. Interleaving is
# done by the round-robin SAMPLER at train time, not by file order.
JOINT_TRAIN_JSON = CACHE_DIR / "joint_sft_train.json"
JOINT_VAL_JSON = CACHE_DIR / "joint_sft_val.json"
JOINT_STATS_JSON = CACHE_DIR / "joint_sft_stats.json"
BLOCK_ORDER = ("bddx", "covla", "road")           # fixed block order in JOINT_*

# ---- Model ----------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# ---- LoRA (reference defaults, same as exp7) ------------------------------
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05

# ---- Trainer args (exp7 values unless marked EXP8) ------------------------
LR = 1e-5
PER_DEVICE_BS = 1
GRAD_ACCUM = 3         # EXP8: one sample from EACH corpus per optimizer step
                       # (exp7: 4). The round-robin sampler yields
                       # bddx,covla,road repeating, so every accumulation
                       # window contains exactly one of each.
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0
MODEL_MAX_LENGTH = 8192
MIN_PIXELS = 784
MAX_PIXELS = 501760    # EXP8: 640 visual tokens (exp7 kept the reference's
                       # 50,176 = 64 tokens). Per-box supervision needs the
                       # numbered box ids legible, and deployment (exp5
                       # qwen_infer) feeds ~600x840 ≈ 504k px frames — this
                       # matches training resolution to both.
DATA_FLATTEN = False   # EXP8: flatten packs samples across sequence — it
                       # would merge samples from different corpora into one
                       # forward pass and break the one-per-dataset step
                       # semantics. Padded collator + sdpa (exp7's proven
                       # fallback path) instead.
ATTN_IMPLEMENTATION = "sdpa"
NUM_EPOCHS = 3         # epoch = one pass over the LARGEST block (road, 9,596);
                       # smaller blocks cycle with per-epoch reshuffles.
LOGGING_STEPS = 10
SAVE_STRATEGY = "epoch"
EVAL_STRATEGY = "epoch"
PER_DEVICE_EVAL_BS = 1


# ---- Prompts / targets ----------------------------------------------------
# ROAD leg: exp5's deployment prompt (vlm_io.build_prompt) minus risk/rationale
# — those two fields have no ROAD GT and inventing targets would violate the
# no-synthetic-labels rule. A DIFFERENT prompt is used on purpose so the model
# does not learn to drop risk/rationale for the deployment prompt (BDD-X's leg
# supervises the justification style instead). Vocab lists and box rendering
# are byte-identical to deployment.
def build_road_prompt(n_boxes: int, agents: str, actions: str, locs: str) -> str:
    return f"""You are a driving-scene perception assistant. The image is a single frame from a Waymo autonomous-driving video. {n_boxes} candidate objects have been detected and drawn as numbered colored boxes (ids 0..{n_boxes - 1}). Some boxes may be false detections that enclose no real object.

For EACH numbered box, classify the object it encloses using ONLY the label vocabularies below. Use the exact label strings as written.

AGENT (choose exactly one, or null if the box is a false detection):
  {agents}

ACTION (choose one or more — what the object is doing; empty if agent is null):
  {actions}

LOCATION (choose zero or more — where it is in the road scene):
  {locs}

Rules:
- Output STRICT JSON only. No prose, no markdown fences.
- One JSON object per box, in a single JSON array, ordered by box id.
- Every box id 0..{n_boxes - 1} must appear exactly once.
- A box that does not tightly enclose a real object of an AGENT category gets "agent": null with empty actions and locations.

Output schema (array of {n_boxes} objects):
[
  {{"box_id": 0, "agent": "<agent or null>", "actions": ["<action>", ...], "locations": ["<loc>", ...]}},
  ...
]"""


def build_road_target(per_box: list) -> str:
    """per_box: [{'box_id','agent','actions','locations'}] with agent=None for
    unmatched (background) boxes. GT label names verbatim, strict JSON."""
    return json.dumps(per_box, ensure_ascii=False)


# CoVLA leg: plain_caption + risk (Approach-3 Task-3 field choice), exp5-style
# strict-JSON discipline. Annotator/auto-caption text goes in verbatim.
COVLA_USER_PROMPT = """You are a driving-scene reasoning assistant. The image is a front-camera frame from the ego vehicle, captured mid-drive.
Describe the driving scene (ego motion and salient objects), then state what the driver of the ego vehicle should be careful about.

Rules:
- Output STRICT JSON only. No prose, no markdown fences.

Output schema:
{"caption": "<short scene description>", "risk": "<what the driver should be careful about>"}"""


def build_covla_target(caption: str, risk: str) -> str:
    return json.dumps({"caption": caption, "risk": risk}, ensure_ascii=False)

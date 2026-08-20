"""Exp10 Tier-0 feature caching — frozen InternVideo2_CLIP_S RoI features.

Per frame: clean native frame → resize 224x224 → replicate x8 (static clip) →
CLIP_S vision tower → post-blocks token map [8, 256, 1024] → mean over T →
[16,16,1024] grid → RoI-average-pool per top-40 detector box (exp1 pooling
arithmetic, boxes normalized to [0,1] frame coords) → [n,1024] fp16.

Output: cache/roi_feats_clip_s_<split>.pt
  {"frames": [key...], "feats": {key: fp16 [n,1024]}, "n_boxes": {key: n},
   "meta": {...}}

Loading note: transformers 5.x forces meta-device init for this custom-code
model; we instantiate the class directly and load safetensors manually.

Usage: CUDA_VISIBLE_DEVICES=0 python -u cache_features.py --split train
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
from PIL import Image

EXP_DIR = Path(__file__).resolve().parent
CACHE_DIR = EXP_DIR / "cache"
EXP6_DIR = EXP_DIR.parent / "exp6_detection_steered"
ROAD_FRAMES_DIR = Path("/data/datasets/ROAD_plusplus/rgb-images")
BOX_H, BOX_W = 600, 840                 # detections pkl box space
MAX_BOXES = 40
TRAIN_SHARDS, NUM_SHARDS = (0, 1), 12   # exp8/exp9 subset arithmetic
REPO = "OpenGVLab/InternVideo2_CLIP_S"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_clip_s(device):
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    cfg = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
    cls = get_class_from_dynamic_module(
        "modeling_internvideo2encoder.InternVideo2_CLIP_small", REPO)
    m = cls(cfg)
    missing, unexpected = m.load_state_dict(
        load_file(hf_hub_download(REPO, "model.safetensors")), strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return m.half().to(device).eval()


def pool_roi(grid: torch.Tensor, box: np.ndarray) -> torch.Tensor:
    """grid [H',W',D]; box [4] normalized xyxy — exp1 _pool_one arithmetic."""
    H, W, _ = grid.shape
    x1, y1, x2, y2 = box.tolist()
    col_lo = max(0, int(x1 * W))
    col_hi = min(W, max(col_lo + 1, round(x2 * W + 0.5)))
    row_lo = max(0, int(y1 * H))
    row_hi = min(H, max(row_lo + 1, round(y2 * H + 0.5)))
    return grid[row_lo:row_hi, col_lo:col_hi, :].mean(dim=(0, 1))


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("train", "val"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    model = load_clip_s(device)
    vt = model.vision_encoder

    tokens = {}
    def _grab(_m, _i, o):
        tokens["blocks_in"] = o          # patch_embed output [B, T, 256, 1024]
    vt.patch_embed.register_forward_hook(_grab)
    # post-blocks tokens: hook the LAST transformer block's output
    vt.blocks[-1].register_forward_hook(lambda _m, _i, o: tokens.__setitem__(
        "post_blocks", o if isinstance(o, torch.Tensor) else o[0]))

    with open(EXP6_DIR / "cache" / f"detections_{args.split}.pkl", "rb") as f:
        payload = pickle.load(f)
    records = payload["records"]
    keys = sorted(records.keys())
    if args.split == "train":
        keys = [k for i, k in enumerate(keys) if i % NUM_SHARDS in TRAIN_SHARDS]
    if args.limit:
        keys = keys[: args.limit]
    print(f"[cache] {args.split}: {len(keys):,} frames", flush=True)

    feats, n_boxes_of = {}, {}
    t0 = time.time()
    grid_meta = None
    for ki, key in enumerate(keys):
        rec = records[key]
        boxes = rec["boxes"][:MAX_BOXES].astype(np.float32)
        if boxes.shape[0] == 0:
            continue
        vname, fid = key.rsplit("/", 1)
        img = Image.open(ROAD_FRAMES_DIR / vname / f"{int(fid):05d}.jpg").convert("RGB")
        arr = np.asarray(img.resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255.0
        arr = (arr - MEAN) / STD
        x = torch.from_numpy(arr).permute(2, 0, 1)                  # [3,224,224]
        x = x.unsqueeze(0).unsqueeze(0).repeat(1, 8, 1, 1, 1)       # [1,8,3,224,224]
        tokens.clear()
        model.encode_vision(x.half().to(device))
        tok = tokens["post_blocks"]                                  # [1, N, 1024]
        n_tok = tok.shape[1]
        n_spatial = 8 * 256
        # drop leading cls token(s) if present
        tok = tok[:, n_tok - n_spatial:, :]
        grid = tok.view(8, 16, 16, -1).float().mean(dim=0)           # [16,16,1024]
        if grid_meta is None:
            grid_meta = {"n_tokens": int(n_tok), "grid": [16, 16], "dim": int(grid.shape[-1])}
            print(f"[cache] token layout: {grid_meta}", flush=True)

        norm = boxes / np.array([BOX_W, BOX_H, BOX_W, BOX_H], dtype=np.float32)
        norm = np.clip(norm, 0.0, 1.0)
        f = torch.stack([pool_roi(grid, b) for b in norm]).half().cpu()
        feats[key] = f
        n_boxes_of[key] = int(boxes.shape[0])
        if (ki + 1) % 500 == 0:
            print(f"[cache] {ki + 1}/{len(keys)} | "
                  f"{(time.time() - t0) / (ki + 1):.3f}s/frame", flush=True)

    CACHE_DIR.mkdir(exist_ok=True)
    out = CACHE_DIR / f"roi_feats_clip_s_{args.split}.pt"
    torch.save({"frames": list(feats.keys()), "feats": feats,
                "n_boxes": n_boxes_of,
                "meta": {"encoder": REPO, "input": "clean-224-static8",
                         "pool": "roi-mean-postblocks-Tmean", **(grid_meta or {})}}, out)
    print(f"[cache] wrote {out} ({len(feats):,} frames, "
          f"{time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

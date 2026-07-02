"""Exp6 stage 1/4 — cache frozen 3D-RetinaNet detections + logits + GT.

Extends exp5's dump with the per-tube 184-dim raw classification logits
(RetinaNetWrapper's `logits_tube`), which the fusion head consumes. Runs the
frozen 17.76% detector over a split and stores, per *frame*:
  - boxes:  [K, 4]    top-K tube boxes, xyxy in 600x840, detector-score sorted
  - scores: [K]       per-tube mean agentness
  - logits: [K, 184]  raw flat classification logits at that frame (float16)
  - gt:     {boxes [N,4], agent/action/loc/duplex/triplet multi-hot} or None

Boxes are bit-identical to exp5's NMS-fixed dump (same frozen weights, same
conf/NMS/top-K path), so exp5's Qwen JSON cache maps onto these rows directly.

Usage:
  CUDA_VISIBLE_DEVICES=0 python -u dump_detections.py [--split val|train] [--limit N]
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

import config as C

# Reuse exp4's detector wrapper + dataloader transforms.
sys.path.insert(0, str(C.EXP4_DIR))
from model import RetinaNetWrapper                                  # noqa: E402
from dataloader import ROADWaymoDataset, clip_to_tensor, rescale_frame_targets  # noqa: E402

_HEADS = ("agent", "action", "loc", "duplex", "triplet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="First N clips (debug). 0 = all.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_path = Path(args.out) if args.out else C.CACHE_DIR / f"detections_{args.split}.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stride = C.TRAIN_STRIDE if args.split == "train" else C.VAL_STRIDE
    print(f"[dump] building dataset ({args.split}, stride={stride}) ...", flush=True)
    ds = ROADWaymoDataset(
        anno_file=C.ANNO_FILE, frames_dir=C.FRAMES_DIR,
        split=args.split, clip_len=C.CLIP_LEN, stride=stride,
    )
    n_clips = len(ds)
    print(f"[dump] {args.split} clips: {n_clips:,}", flush=True)

    print("[dump] loading frozen 3D-RetinaNet (17.76% ckpt) ...", flush=True)
    det = RetinaNetWrapper(
        ckpt_path=C.RETINANET_CKPT, top_k=C.RETINANET_TOPK_TUBES,
        d_retina=C.RETINANET_SPATIAL_DIM,
    ).to(device).eval()

    records: dict[str, dict] = {}
    t0 = time.time()
    with torch.no_grad():
        for i in range(n_clips):
            if args.limit and i >= args.limit:
                break
            vname, fids = ds.clips[i]
            pil_frames, frame_targets = ds[i]
            clip, h, w = clip_to_tensor(pil_frames)                 # [T,3,600,840]
            out = det(clip.unsqueeze(0).to(device))
            boxes = out["boxes"][0].float().cpu().numpy()           # [K,T,4]
            scores = out["scores"][0].float().cpu().numpy()         # [K]
            logits = out["logits_tube"][0].cpu().numpy()            # [K,T,184]
            gt_frames = rescale_frame_targets(frame_targets, h, w)  # GT in 600x840

            for t, fid in enumerate(fids):
                key = f"{vname}/{fid}"
                if key in records:                                  # frames repeat at stride<8·4
                    continue
                ft = gt_frames[t]
                gt = None
                if ft is not None:
                    gt = {"boxes": ft["boxes"].cpu().numpy().astype(np.float32)}
                    for hname in _HEADS:
                        gt[hname] = ft[hname].cpu().numpy().astype(np.float32)
                records[key] = {
                    "boxes": boxes[:, t, :].astype(np.float32),     # [K,4] 600x840
                    "scores": scores.astype(np.float32),            # [K]
                    "logits": logits[:, t, :].astype(np.float16),   # [K,184]
                    "gt": gt,
                }

            if (i + 1) % 25 == 0:
                el = time.time() - t0
                eta = el * (n_clips - i - 1) / max(i + 1, 1)
                print(f"[dump] {i+1}/{n_clips} clips  frames={len(records):,}  "
                      f"elapsed={el:.0f}s  ETA={eta/60:.0f}min", flush=True)

    payload = {
        "records": records,
        "labels": {
            "agent": ds.agent_labels, "action": ds.action_labels,
            "loc": ds.loc_labels, "duplex": ds.duplex_labels,
            "triplet": ds.triplet_labels,
        },
        "duplex_childs": ds.duplex_childs,
        "triplet_childs": ds.triplet_childs,
        "meta": {"split": args.split, "stride": stride,
                 "topk": C.RETINANET_TOPK_TUBES, "n_frames": len(records),
                 "has_logits": True},
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[dump] wrote {len(records):,} frames → {out_path}  "
          f"({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()

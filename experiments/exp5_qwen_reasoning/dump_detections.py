"""Exp5 stage 1/3 — cache frozen 3D-RetinaNet detections + GT for the val split.

For every clip in the baseline val protocol (stride=32), runs the frozen 17.76%
detector and stores, per *frame*:
  - boxes:  [K, 4]  top-K tube boxes, xyxy in the 600x840 detector-input space,
                    sorted by detector score (box id 0 = strongest).
  - scores: [K]     per-tube mean agentness (used as detection confidence in eval).
  - gt:     {boxes [N,4], agent/action/loc/duplex/triplet multi-hot} or None.

Output: cache/detections_<split>.pkl  (also embeds the label vocab so the Qwen
and eval stages never have to re-load the multi-GB annotation JSON).

Usage:
  CUDA_VISIBLE_DEVICES=0 python -u dump_detections.py [--split val] [--limit N]
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

    print(f"[dump] building dataset ({args.split}) ...", flush=True)
    ds = ROADWaymoDataset(
        anno_file=C.ANNO_FILE, frames_dir=C.FRAMES_DIR,
        split=args.split, clip_len=C.CLIP_LEN, stride=C.VAL_STRIDE,
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
            gt_frames = rescale_frame_targets(frame_targets, h, w)  # GT in 600x840

            for t, fid in enumerate(fids):
                key = f"{vname}/{fid}"
                if key in records:                                  # frames are unique at stride 32
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
        "meta": {"split": args.split, "stride": C.VAL_STRIDE,
                 "topk": C.RETINANET_TOPK_TUBES, "n_frames": len(records)},
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[dump] wrote {len(records):,} frames → {out_path}  "
          f"({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()

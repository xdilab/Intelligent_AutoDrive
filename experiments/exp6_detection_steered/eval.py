"""Exp6 eval — frame-level f-mAP of the fused head on the fixed test split.

Same protocol as exp5's eval (baseline evaluator, IoU=0.5, top-K detector
boxes as the row set) so all three numbers are apples-to-apples on identical
boxes:
  1. eval.py --detector-only     → RetinaNet logits over top-K (the control /
                                   floor; NOT the 17.76% all-anchor number)
  2. exp5 results_nms_fixed.json → zero-shot Qwen hard labels over top-K
  3. eval.py --ckpt ...          → detection-steered fusion (this experiment)

Scoring follows the baseline's val.py: each class row carries that class's
own sigmoid confidence (agentness included as class 0) — no score mixing.

Usage:
  python -u eval.py --ckpt checkpoints/fusion_ep020.pth [--split val] [--out results.json]
  python -u eval.py --detector-only [--split val]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import config as C
from model import DetectionSteeredFusion
from dataset import build_samples

sys.path.insert(0, C.BASELINE_ROOT)
import modules.evaluation as baseline_eval                          # noqa: E402
from modules.utils import get_individual_labels                     # noqa: E402

_HEADS = ("agent", "action", "loc", "duplex", "triplet")
_LABEL_TYPES = ("agentness", "agent", "action", "loc", "duplex", "triplet")


@torch.no_grad()
def _predict_probs(samples: dict, ckpt: str | None, device,
                   zero_lang: bool = False) -> np.ndarray:
    """[M, 184] per-sample class probabilities (fused, or raw detector)."""
    det = samples["det"]
    if ckpt is None:
        return det.sigmoid().numpy()
    model = DetectionSteeredFusion().to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[eval] loaded {ckpt} (epoch {state.get('epoch', '?')}, zero_lang={zero_lang})",
          flush=True)
    probs = np.empty((det.shape[0], C.NUM_CLASSES), dtype=np.float32)
    bs = 4096
    for s in range(0, det.shape[0], bs):
        sl = slice(s, s + bs)
        struct = samples["struct"][sl].to(device)
        rat = samples["rat"][sl].to(device)
        if zero_lang:                       # match training distribution of the zl cell
            struct = torch.zeros_like(struct)
            rat = torch.zeros_like(rat)
        logits = model(det[sl].to(device), struct, rat)
        probs[sl] = logits.sigmoid().cpu().numpy()
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--detector-only", action="store_true",
                    help="Score raw RetinaNet logits over the same top-K rows (control).")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0, help="First N frames (debug). 0 = all.")
    ap.add_argument("--zero-lang", action="store_true",
                    help="Zero the language tracks at predict time (for fusion_zl_* ckpts).")
    args = ap.parse_args()
    if not args.detector_only and not args.ckpt:
        ap.error("need --ckpt or --detector-only")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = build_samples(args.split, limit=args.limit)
    records, labels = samples["records"], samples["labels"]
    probs = _predict_probs(samples, None if args.detector_only else args.ckpt, device,
                           zero_lang=args.zero_lang)

    all_classes = [["agentness"], labels["agent"], labels["action"],
                   labels["loc"], labels["duplex"], labels["triplet"]]
    num_c = [len(c) for c in all_classes]                           # [1,10,22,16,49,86]
    nT = len(num_c)
    offsets = np.cumsum([0] + num_c)                                # flat → per-head slices

    gt_all = [[] for _ in range(nT)]
    det_all = [[[] for _ in range(num_c[t])] for t in range(nT)]

    frame_of, box_of = samples["frame_of"], samples["box_of"]
    t0 = time.time()
    for fi, key in enumerate(samples["frames"]):
        rec = records[key]
        mask = frame_of == fi
        bids = box_of[mask]
        p = probs[mask]                                             # [n,184]
        boxes = rec["boxes"][bids]                                  # [n,4]

        # ---- GT rows (mirrors exp5/eval_qwen.py, i.e. the baseline's val.py) ----
        gt = rec["gt"]
        if gt is None or gt["boxes"].shape[0] == 0:
            gt_boxes = np.zeros((0, 4), dtype=np.float32)
            mh = {h: np.zeros((0, num_c[i + 1]), dtype=np.float32) for i, h in enumerate(_HEADS)}
        else:
            gt_boxes = gt["boxes"].astype(np.float32)
            mh = {h: gt[h].astype(np.float32) for h in _HEADS}
        for nlt in range(nT):
            if nlt == 0:
                g = np.zeros((gt_boxes.shape[0], 5), dtype=np.float32)
                g[:, :4] = gt_boxes
            else:
                g = get_individual_labels(gt_boxes, mh[_HEADS[nlt - 1]]).astype(np.float32)
            gt_all[nlt].append(g)

        # ---- Detection rows: every box scored on every class with its own prob ----
        for nlt in range(nT):
            for c in range(num_c[nlt]):
                col = offsets[nlt] + c
                rows = np.concatenate([boxes, p[:, col : col + 1]], axis=1).astype(np.float32)
                det_all[nlt][c].append(rows)

        if (fi + 1) % 1000 == 0:
            print(f"[eval] accumulated {fi+1}/{len(samples['frames'])} frames "
                  f"{time.time()-t0:.0f}s", flush=True)

    mode = "detector-only control" if args.detector_only else f"fused ({args.ckpt})"
    print(f"[eval] running baseline f-mAP over {len(samples['frames']):,} frames — {mode}",
          flush=True)
    mAP, ap_all, ap_strs = baseline_eval.evaluate(gt_all, det_all, all_classes, iou_thresh=0.5)

    print(f"\n=== Summary f-mAP per label type ({mode}) ===")
    summary = {}
    for nlt in range(nT):
        summary[_LABEL_TYPES[nlt]] = float(mAP[nlt])
        print(f"  {_LABEL_TYPES[nlt]:>10s}: {mAP[nlt]:.4f}%")
    print("\n  (frozen 3D-RetinaNet all-anchor baseline = 17.76% agent f-mAP; "
          "compare fused vs --detector-only on these same top-K rows)")

    if args.out:
        out = {"mode": mode, "summary": summary,
               "per_class": {_LABEL_TYPES[t]: ap_strs[t] for t in range(nT)},
               "n_frames": len(samples["frames"])}
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"[eval] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

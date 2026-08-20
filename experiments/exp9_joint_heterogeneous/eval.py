"""Exp9 eval — f-mAP of the ROAD heads on the fixed val split, plus corrected
constraint-violation rates.

Protocol identical to the exp5/exp6/exp8 control family (baseline evaluator,
IoU=0.5, detector top-40 boxes as the row set) so exp9 numbers sit in the same
table as detector-only (agent 14.62 / action 12.16 / loc 11.76 / duplex 10.33 /
triplet 7.48). Row-building code mirrors exp6/eval.py; the only difference is
where the [n,184] probabilities come from: a heads forward pass over the frame
(single forward, no token generation — the cheap-eval property from DESIGN.md).

Box rendering for 'drawn' mode reuses exp5's deployment renderer
(vlm_io.draw_numbered_boxes) at native resolution — byte-identical to what the
cached-prompt lineage fed the VLM.

Violation rate: fraction of above-threshold co-predictions (per box, argmax-
free multi-hot at conf>t) outside the CORRECTED valid sets. Not comparable to
any pre-2026-08-17 violation number (those used the fossil childs arrays).

Usage:
  python -u eval.py --ckpt checkpoints/r1_ep003 [--out results_r1.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
from PIL import Image

import config as C
from dataset import RoadDataset
from model import Exp9Model

sys.path.insert(0, str(C.EXP5_DIR))
import vlm_io                                                        # noqa: E402

sys.path.insert(0, "/data/repos/PedestrianIntent++/ROAD_plus_plus_Baseline")
import modules.evaluation as baseline_eval                           # noqa: E402
from modules.utils import get_individual_labels                      # noqa: E402

_HEADS = ("agent", "action", "loc", "duplex", "triplet")
_LABEL_TYPES = ("agentness", "agent", "action", "loc", "duplex", "triplet")


def _native_image(key: str, boxes_600: np.ndarray) -> tuple[Image.Image, np.ndarray]:
    """Native frame (+ drawn boxes in 'drawn' mode) and boxes normalized [0,1]."""
    vname, fid = key.rsplit("/", 1)
    img = Image.open(Path(C.ROAD_FRAMES_DIR) / vname / f"{int(fid):05d}.jpg").convert("RGB")
    nw, nh = img.size
    sx, sy = nw / C.ROAD_BOX_W, nh / C.ROAD_BOX_H
    boxes_native = np.stack([boxes_600[:, 0] * sx, boxes_600[:, 1] * sy,
                             boxes_600[:, 2] * sx, boxes_600[:, 3] * sy], axis=1)
    if C.ROAD_INPUT_MODE == "drawn":
        img = vlm_io.draw_numbered_boxes(img, boxes_native.tolist())
    boxes_norm = boxes_600 / np.array(
        [C.ROAD_BOX_W, C.ROAD_BOX_H, C.ROAD_BOX_W, C.ROAD_BOX_H], dtype=np.float32)
    return img, np.clip(boxes_norm, 0.0, 1.0)


def _violations(probs: np.ndarray, thresh: float = 0.5) -> dict:
    """Corrected-set violation rates over per-box multi-hot co-predictions."""
    spec = json.loads(C.CONSTRAINTS_JSON.read_text())
    valid_d = set(map(tuple, spec["duplex_childs_derived"]))
    valid_t = set(map(tuple, spec["triplet_childs_derived"]))
    a_off, c_off, l_off = 1, 1 + C.N_AGENTS, 1 + C.N_AGENTS + C.N_ACTIONS
    d_cop = d_vio = t_cop = t_vio = 0
    for p in probs:
        ags = np.nonzero(p[a_off:a_off + C.N_AGENTS] > thresh)[0]
        acs = np.nonzero(p[c_off:c_off + C.N_ACTIONS] > thresh)[0]
        lcs = np.nonzero(p[l_off:l_off + C.N_LOCS] > thresh)[0]
        for a in ags:
            for c in acs:
                d_cop += 1
                d_vio += (int(a), int(c)) not in valid_d
                for l in lcs:
                    t_cop += 1
                    t_vio += (int(a), int(c), int(l)) not in valid_t
    return {"duplex_viol_pct": round(100 * d_vio / max(d_cop, 1), 2),
            "triplet_viol_pct": round(100 * t_vio / max(t_cop, 1), 2),
            "duplex_copreds": d_cop, "triplet_copreds": t_cop,
            "conf_thresh": thresh, "constraint_set": "corrected (49/86)"}


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Exp9Model(device)
    model.load_head_and_adapter(Path(args.ckpt))
    model.vlm.eval()
    model.head.eval()

    ds = RoadDataset(det_pkl=C.ROAD_DET_VAL_PKL, shards=None, mode="clean")
    keys = ds.keys[: args.limit] if args.limit else ds.keys
    records, labels = ds.records, ds.labels

    all_classes = [["agentness"], labels["agent"], labels["action"],
                   labels["loc"], labels["duplex"], labels["triplet"]]
    num_c = [len(c) for c in all_classes]
    nT = len(num_c)
    offsets = np.cumsum([0] + num_c)

    gt_all = [[] for _ in range(nT)]
    det_all = [[[] for _ in range(num_c[t])] for t in range(nT)]
    probs_accum = []
    t0 = time.time()

    for fi, key in enumerate(keys):
        rec = records[key]
        boxes = rec["boxes"][: C.MAX_BOXES_PER_FRAME].astype(np.float32)
        if boxes.shape[0] == 0:
            continue
        img, boxes_norm = _native_image(key, boxes)
        logits = model.road_forward(img, torch.from_numpy(boxes_norm))
        p = torch.sigmoid(logits.float()).cpu().numpy()               # [n,184]
        probs_accum.append(p)

        gt = rec["gt"]
        if gt is None or gt["boxes"].shape[0] == 0:
            gt_boxes = np.zeros((0, 4), dtype=np.float32)
            mh = {h: np.zeros((0, num_c[i + 1]), dtype=np.float32)
                  for i, h in enumerate(_HEADS)}
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
        for nlt in range(nT):
            for c in range(num_c[nlt]):
                col = offsets[nlt] + c
                det_all[nlt][c].append(
                    np.concatenate([boxes, p[:, col:col + 1]], axis=1).astype(np.float32))

        if (fi + 1) % 200 == 0:
            print(f"[eval] {fi + 1}/{len(keys)} frames | "
                  f"{(time.time() - t0) / (fi + 1):.2f}s/frame", flush=True)

    print(f"[eval] running baseline f-mAP over {len(keys):,} frames", flush=True)
    mAP, _, ap_strs = baseline_eval.evaluate(gt_all, det_all, all_classes, iou_thresh=0.5)

    print(f"\n=== Summary f-mAP per label type (exp9 heads, {args.ckpt}) ===")
    summary = {}
    for nlt in range(nT):
        summary[_LABEL_TYPES[nlt]] = float(mAP[nlt])
        print(f"  {_LABEL_TYPES[nlt]:>10s}: {mAP[nlt]:.4f}%")

    vio = _violations(np.concatenate(probs_accum))
    print(f"\n  corrected-set violations: duplex {vio['duplex_viol_pct']}% "
          f"({vio['duplex_copreds']:,} copreds) | triplet {vio['triplet_viol_pct']}% "
          f"({vio['triplet_copreds']:,})")
    print("  (top-40 protocol: compare vs exp6 detector-only control, "
          "NOT the 17.76% all-anchor number)")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"ckpt": args.ckpt, "summary": summary, "violations": vio,
             "per_class": {_LABEL_TYPES[t]: ap_strs[t] for t in range(nT)},
             "n_frames": len(keys)}, indent=2))
        print(f"[eval] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

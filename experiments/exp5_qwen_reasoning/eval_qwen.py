"""Exp5 stage 3/3 — frame-level f-mAP of vanilla Qwen2.5-VL over RetinaNet boxes.

Maps each cached Qwen per-box classification back onto its detector box
(RetinaNet box score = detection confidence) and runs the *baseline's own*
evaluator (modules.evaluation.evaluate, IoU=0.5) for every label head:
agentness / agent / action / loc / duplex / triplet.

duplex & triplet are derived (valid-by-construction) from Qwen's agent x action
x location, parsed from the label strings exactly as the dataset builds GT.

This is the untrained-VLM floor — scored over the top-K detector boxes, not all
anchors, so it is NOT directly the 17.76% all-anchor baseline.

Usage:
  python -u eval_qwen.py [--split val] [--out results.json] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

import config as C

sys.path.insert(0, C.BASELINE_ROOT)
import modules.evaluation as baseline_eval                          # noqa: E402
from modules.utils import get_individual_labels                     # noqa: E402

_HEADS = ("agent", "action", "loc", "duplex", "triplet")
_LABEL_TYPES = ("agentness", "agent", "action", "loc", "duplex", "triplet")


def _build_duplex_lookup(duplex_labels, agent_labels, action_labels):
    """'Car-MovAway' → idx, parsed greedily (agent prefix, then action)."""
    lut = {}
    for idx, s in enumerate(duplex_labels):
        for ag in agent_labels:
            if not s.startswith(ag + "-"):
                continue
            act = s[len(ag) + 1:]
            if act in action_labels:
                lut[(ag, act)] = idx
            break
    return lut


def _build_triplet_lookup(triplet_labels, agent_labels, action_labels, loc_labels):
    loc_set = set(loc_labels)
    lut = {}
    for idx, s in enumerate(triplet_labels):
        for ag in agent_labels:
            if not s.startswith(ag + "-"):
                continue
            rest = s[len(ag) + 1:]
            for act in action_labels:
                if not rest.startswith(act + "-"):
                    continue
                loc = rest[len(act) + 1:]
                if loc in loc_set:
                    lut[(ag, act, loc)] = idx
                break
            break
    return lut


def _qwen_cache_path(root: Path, key: str) -> Path:
    vname, fid = key.rsplit("/", 1)
    return root / vname / f"{int(fid):05d}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--detections", default=None)
    ap.add_argument("--qwen-cache", default=None,
                    help="Qwen JSON cache dir (default: plain zero-shot cache; "
                         "pass cache/qwen_steered for the detection-steered run).")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    qwen_root = Path(args.qwen_cache) if args.qwen_cache else C.QWEN_CACHE_DIR
    print(f"[eval] qwen cache: {qwen_root}", flush=True)

    det_path = Path(args.detections) if args.detections else C.CACHE_DIR / f"detections_{args.split}.pkl"
    with open(det_path, "rb") as f:
        payload = pickle.load(f)
    records = payload["records"]
    L = payload["labels"]
    agent_labels, action_labels = L["agent"], L["action"]
    loc_labels, duplex_labels, triplet_labels = L["loc"], L["duplex"], L["triplet"]

    agent_idx = {n: i for i, n in enumerate(agent_labels)}
    action_idx = {n: i for i, n in enumerate(action_labels)}
    loc_idx = {n: i for i, n in enumerate(loc_labels)}
    duplex_lut = _build_duplex_lookup(duplex_labels, agent_labels, action_labels)
    triplet_lut = _build_triplet_lookup(triplet_labels, agent_labels, action_labels, loc_labels)

    all_classes = [["agentness"], agent_labels, action_labels,
                   loc_labels, duplex_labels, triplet_labels]
    num_c = [len(c) for c in all_classes]                           # [1,10,22,16,49,86]
    nT = len(num_c)

    gt_all = [[] for _ in range(nT)]
    det_all = [[[] for _ in range(num_c[t])] for t in range(nT)]

    keys = sorted(records.keys())
    n_eval = n_missing = n_parsed_boxes = 0
    t0 = time.time()
    for ki, key in enumerate(keys):
        if args.limit and n_eval >= args.limit:
            break
        cpath = _qwen_cache_path(qwen_root, key)
        if not cpath.exists():
            n_missing += 1
            continue
        try:
            qc = json.loads(cpath.read_text())
        except Exception:
            n_missing += 1
            continue
        rec = records[key]
        n_boxes = int(qc.get("n_boxes", 0))
        boxes = rec["boxes"][:n_boxes]                              # [n,4] 600x840
        scores = rec["scores"][:n_boxes]                           # [n]
        parsed = qc.get("parsed", []) or []

        # ---- GT for this frame (mirrors baseline eval.py) ----
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

        # ---- Detections from Qwen (per class rows; score = box score) ----
        fdet = [[[] for _ in range(num_c[t])] for t in range(nT)]
        for i in range(n_boxes):
            row = [float(boxes[i, 0]), float(boxes[i, 1]),
                   float(boxes[i, 2]), float(boxes[i, 3]), float(scores[i])]
            fdet[0][0].append(row)                                 # agentness: every box
            p = parsed[i] if i < len(parsed) else None
            if not p:
                continue
            n_parsed_boxes += 1
            ag = p.get("agent")
            acts = p.get("actions") or []
            locs = p.get("locations") or []
            if ag in agent_idx:
                fdet[1][agent_idx[ag]].append(row)
            for a in acts:
                if a in action_idx:
                    fdet[2][action_idx[a]].append(row)
            for l in locs:
                if l in loc_idx:
                    fdet[3][loc_idx[l]].append(row)
            if ag is not None:
                for a in acts:
                    d = duplex_lut.get((ag, a))
                    if d is not None:
                        fdet[4][d].append(row)
                    for l in locs:
                        tr = triplet_lut.get((ag, a, l))
                        if tr is not None:
                            fdet[5][tr].append(row)
        for nlt in range(nT):
            for c in range(num_c[nlt]):
                rows = fdet[nlt][c]
                det_all[nlt][c].append(
                    np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 5), np.float32)
                )
        n_eval += 1
        if n_eval % 200 == 0:
            print(f"[eval] accumulated {n_eval} frames "
                  f"(missing-qwen={n_missing}) {time.time()-t0:.0f}s", flush=True)

    print(f"[eval] frames evaluated={n_eval}  missing-qwen={n_missing}  "
          f"parsed-boxes={n_parsed_boxes}", flush=True)
    if n_eval == 0:
        print("[eval] nothing to evaluate — run qwen_infer.py first.")
        return

    print("[eval] running baseline f-mAP ...", flush=True)
    mAP, ap_all, ap_strs = baseline_eval.evaluate(gt_all, det_all, all_classes, iou_thresh=0.5)

    print("\n=== Summary f-mAP per label type (vanilla Qwen2.5-VL over RetinaNet boxes) ===")
    summary = {}
    for nlt in range(nT):
        summary[_LABEL_TYPES[nlt]] = float(mAP[nlt])
        print(f"  {_LABEL_TYPES[nlt]:>10s}: {mAP[nlt]:.4f}%")
    print(f"\n  (frozen 3D-RetinaNet all-anchor baseline = 17.76% agent f-mAP)")

    if args.out:
        out = {"summary": summary,
               "per_class": {_LABEL_TYPES[t]: ap_strs[t] for t in range(nT)},
               "n_frames": n_eval, "n_missing_qwen": n_missing,
               "meta": payload.get("meta", {})}
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"[eval] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

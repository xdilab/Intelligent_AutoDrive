"""Exp6 stage 3/4 — assemble (frame, box) fusion samples from the caches.

One sample = one detector box on one frame:
  det_logits [184]  raw RetinaNet flat logits (from detections_<split>.pkl)
  struct     [54]   vectorized Qwen structured fields (agent/actions/locations/
                    risk + parsed flag), layout in config.STRUCT_DIM comment
  rationale  [1153] frozen SigLIP text embedding of the rationale sentence
                    (+ trailing has-rationale flag; zeros+0 when absent)
  target     [184]  GT flat multi-hot via IoU>=0.5 greedy-argmax match of the
                    detector box to GT boxes (agentness=1 when matched), zeros
                    for unmatched (background) — same assignment role as exp4's
                    losses.py matching, applied per frame instead of per tube.

Only boxes Qwen actually classified (id < n_boxes, capped at 40) become
samples; frames with no Qwen cache entry are skipped and counted.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

import config as C

_HEADS = ("agent", "action", "loc", "duplex", "triplet")


def _qwen_cache_path(key: str) -> Path:
    vname, fid = key.rsplit("/", 1)
    return C.QWEN_CACHE_DIR / vname / f"{int(fid):05d}.json"


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of a [n,4] vs b [m,4], xyxy."""
    area_a = np.maximum(a[:, 2] - a[:, 0], 0) * np.maximum(a[:, 3] - a[:, 1], 0)
    area_b = np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def _struct_vector(p: Optional[dict], agent_idx: Dict[str, int],
                   action_idx: Dict[str, int], loc_idx: Dict[str, int]) -> np.ndarray:
    """Vectorize one normalized Qwen parse (exp5 vlm_io schema) → [STRUCT_DIM]."""
    v = np.zeros(C.STRUCT_DIM, dtype=np.float32)
    o = 0
    if p is None:                                       # unparsed box: only nulls set
        v[o + C.N_AGENTS] = 1.0                         # agent-null
        v[C.STRUCT_DIM - 2] = 1.0                       # risk-null
        return v
    ag = p.get("agent")
    if ag in agent_idx:
        v[o + agent_idx[ag]] = 1.0
    else:
        v[o + C.N_AGENTS] = 1.0                         # agent-null
    o += C.N_AGENTS + 1
    for a in p.get("actions") or []:
        if a in action_idx:
            v[o + action_idx[a]] = 1.0
    o += C.N_ACTIONS
    for l in p.get("locations") or []:
        if l in loc_idx:
            v[o + loc_idx[l]] = 1.0
    o += C.N_LOCS
    risk = p.get("risk")
    if risk in C.RISK_LEVELS:
        v[o + C.RISK_LEVELS.index(risk)] = 1.0
    else:
        v[o + 3] = 1.0                                  # risk-null
    o += 4
    v[o] = 1.0                                          # parsed flag
    return v


def build_samples(split: str, limit: int = 0) -> dict:
    """Load caches and materialize all fusion samples for `split`.

    Returns dict with stacked arrays (float16 where cached as such) plus the
    per-frame records/labels needed by eval.py to rebuild detection rows.
    """
    det_path = C.CACHE_DIR / f"detections_{split}.pkl"
    with open(det_path, "rb") as f:
        payload = pickle.load(f)
    records, labels = payload["records"], payload["labels"]
    assert payload["meta"].get("has_logits"), (
        f"{det_path} has no per-tube logits — re-run exp6/dump_detections.py"
    )

    rat_path = C.CACHE_DIR / f"rationale_{split}.pkl"
    with open(rat_path, "rb") as f:
        rat = pickle.load(f)["emb"]

    agent_idx = {n: i for i, n in enumerate(labels["agent"])}
    action_idx = {n: i for i, n in enumerate(labels["action"])}
    loc_idx = {n: i for i, n in enumerate(labels["loc"])}

    keys = sorted(records.keys())
    if limit:
        keys = keys[:limit]

    det_rows: List[np.ndarray] = []
    struct_rows: List[np.ndarray] = []
    rat_rows: List[np.ndarray] = []
    tgt_rows: List[np.ndarray] = []
    frame_of: List[int] = []            # sample → index into `frames`
    box_of: List[int] = []              # sample → box id within its frame
    frames: List[str] = []              # frame keys that produced samples
    n_no_qwen = 0

    for key in keys:
        cpath = _qwen_cache_path(key)
        if not cpath.exists():
            n_no_qwen += 1
            continue
        try:
            qc = json.loads(cpath.read_text())
        except Exception:
            n_no_qwen += 1
            continue
        rec = records[key]
        n_boxes = min(int(qc.get("n_boxes", 0)), C.MAX_BOXES_PER_FRAME)
        if n_boxes == 0:
            continue
        parsed = qc.get("parsed") or []
        boxes = rec["boxes"][:n_boxes]                              # [n,4]
        logits = rec["logits"][:n_boxes].astype(np.float32)         # [n,184]

        # ---- targets via IoU matching ----
        tgts = np.zeros((n_boxes, C.NUM_CLASSES), dtype=np.float32)
        gt = rec["gt"]
        if gt is not None and gt["boxes"].shape[0] > 0:
            iou = _iou_matrix(boxes, gt["boxes"])                   # [n,N]
            best = iou.argmax(axis=1)
            matched = iou[np.arange(n_boxes), best] >= C.IOU_MATCH_THRESH
            for i in np.nonzero(matched)[0]:
                g = best[i]
                tgts[i, 0] = 1.0
                off = 1
                for h in _HEADS:
                    mh = gt[h][g]
                    tgts[i, off : off + mh.shape[0]] = mh
                    off += mh.shape[0]

        frame_i = len(frames)
        frames.append(key)
        rat_frame = rat.get(key, {})
        for i in range(n_boxes):
            p = parsed[i] if i < len(parsed) else None
            det_rows.append(logits[i])
            struct_rows.append(_struct_vector(p, agent_idx, action_idx, loc_idx))
            r = np.zeros(C.RATIONALE_DIM + 1, dtype=np.float32)
            if i in rat_frame:
                r[: C.RATIONALE_DIM] = rat_frame[i].astype(np.float32)
                r[-1] = 1.0                                         # has-rationale
            rat_rows.append(r)
            tgt_rows.append(tgts[i])
            frame_of.append(frame_i)
            box_of.append(i)

    print(f"[data] {split}: {len(det_rows):,} samples from {len(frames):,} frames "
          f"(skipped {n_no_qwen} frames with no Qwen cache)", flush=True)
    return {
        "det": torch.from_numpy(np.stack(det_rows)),
        "struct": torch.from_numpy(np.stack(struct_rows)),
        "rat": torch.from_numpy(np.stack(rat_rows)),
        "target": torch.from_numpy(np.stack(tgt_rows)),
        "frame_of": np.asarray(frame_of),
        "box_of": np.asarray(box_of),
        "frames": frames,
        "records": records,
        "labels": labels,
    }


class FusionSampleDataset(Dataset):
    """Thin index view over build_samples() output."""

    def __init__(self, samples: dict):
        self.s = samples

    def __len__(self) -> int:
        return self.s["det"].shape[0]

    def __getitem__(self, i: int):
        return (self.s["det"][i], self.s["struct"][i],
                self.s["rat"][i], self.s["target"][i])

"""Exp9 datasets — one ROAD box-supervision corpus + language SFT corpora.

RoadDataset   one item = one frame: (image, boxes_norm [n,4], targets [n,184]).
              Boxes are the frozen detector's top-40 (exp6 detections pkl,
              600x840 space → normalized); targets via IoU≥0.5 greedy-argmax
              against GT with all-zeros rows for unmatched boxes (exp6
              dataset.py logic, same constants). Frame subset = sorted-key
              shards {0,1} of 12 — the same 9,596 frames as exp8/Stage-1.
              'drawn' mode reads exp8's cached box-overlay renders
              (<video>_<fid>.jpg); 'clean' reads the raw dataset frame.

LangDataset   one item = (image_path, user_text, target_text) from the
              exp7/exp8 SFT JSONs (schema: {image, conversations:[human,gpt]};
              the human value carries an '<image>\n' prefix to strip).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import config as C


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Verbatim from exp6/dataset.py."""
    area_a = np.maximum(a[:, 2] - a[:, 0], 0) * np.maximum(a[:, 3] - a[:, 1], 0)
    area_b = np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


_HEADS = ("agent", "action", "loc", "duplex", "triplet")


def flat_targets(boxes: np.ndarray, gt: dict | None) -> np.ndarray:
    """[n,184] multi-hot via IoU≥0.5 greedy-argmax; zeros for unmatched
    (exp6 dataset.py target construction, unchanged)."""
    n = boxes.shape[0]
    tgts = np.zeros((n, C.NUM_CLASSES), dtype=np.float32)
    if gt is not None and gt["boxes"].shape[0] > 0:
        iou = _iou_matrix(boxes, gt["boxes"])
        best = iou.argmax(axis=1)
        matched = iou[np.arange(n), best] >= C.IOU_MATCH_THRESH
        for i in np.nonzero(matched)[0]:
            g = best[i]
            tgts[i, 0] = 1.0
            off = 1
            for h in _HEADS:
                mh = gt[h][g]
                tgts[i, off: off + mh.shape[0]] = mh
                off += mh.shape[0]
    return tgts


class RoadDataset:
    def __init__(self, det_pkl: Path = C.ROAD_DET_TRAIN_PKL,
                 shards=C.ROAD_TRAIN_SHARDS, mode: str = C.ROAD_INPUT_MODE):
        with open(det_pkl, "rb") as f:
            payload = pickle.load(f)
        self.records = payload["records"]
        self.labels = payload["labels"]
        keys = sorted(self.records.keys())
        if shards is not None:
            keys = [k for i, k in enumerate(keys)
                    if i % C.ROAD_NUM_SHARDS in shards]
        self.keys = keys
        self.mode = mode
        print(f"[data] road: {len(self.keys):,} frames "
              f"(shards={shards}, mode={mode})", flush=True)

    def __len__(self):
        return len(self.keys)

    def _image(self, key: str) -> Image.Image:
        vname, fid = key.rsplit("/", 1)
        if self.mode == "drawn":
            p = C.ROAD_DRAWN_DIR / f"{vname}_{int(fid)}.jpg"
            assert p.exists(), (
                f"{p} missing — exp8 render not found for {key}; "
                "check ROAD_DRAWN_DIR naming or use ROAD_INPUT_MODE='clean'"
            )
            return Image.open(p).convert("RGB")
        return Image.open(
            Path(C.ROAD_FRAMES_DIR) / vname / f"{int(fid):05d}.jpg"
        ).convert("RGB")

    def __getitem__(self, i: int):
        key = self.keys[i]
        rec = self.records[key]
        boxes = rec["boxes"][: C.MAX_BOXES_PER_FRAME].astype(np.float32)  # 600x840
        tgts = flat_targets(boxes, rec["gt"])
        boxes_norm = boxes / np.array(
            [C.ROAD_BOX_W, C.ROAD_BOX_H, C.ROAD_BOX_W, C.ROAD_BOX_H],
            dtype=np.float32)
        return (self._image(key),
                torch.from_numpy(np.clip(boxes_norm, 0.0, 1.0)),
                torch.from_numpy(tgts))


class LangDataset:
    def __init__(self, sft_json: Path, name: str):
        items = json.loads(Path(sft_json).read_text())
        self.items = items
        self.name = name
        print(f"[data] {name}: {len(items):,} SFT samples", flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        it = self.items[i]
        user = it["conversations"][0]["value"]
        if user.startswith("<image>"):
            user = user[len("<image>"):].lstrip("\n")
        return it["image"], user, it["conversations"][1]["value"]

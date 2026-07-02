"""Exp4 dataloader.

Re-uses exp1_road_r's ROADWaymoDataset (clip-of-PIL-frames + per-frame label dicts)
unchanged. This module adds:
  - clip_to_tensor(pil_frames, short_side, max_size): ResizeClip_Fixed +
    ToTensor + ImageNet-normalize  -> [T, 3, H, W].
  - rescale_frame_targets(frame_targets, orig_size, new_size): scale the GT
    boxes alongside the image resize.
  - exp4_collate_fn(batch): wraps a single sample into (clip [1, T, 3, H, W],
    frame_targets).
  - build_dataloader(split, ...): convenience factory.

ROAD-Waymo native resolution is ~1280×1920 (Waymo standard). Frozen RetinaNet
was trained at MIN_SIZE=600, MAX_SIZE=840; we match that here to stay in-distribution.
MoonViT will get the same input and use its own 2D RoPE to handle the resolution.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

import config as C


# Bootstrap import: ROADWaymoDataset lives in exp1_road_r/dataset.py.
_EXP1_DIR = Path("/data/repos/ROAD_Reason/experiments/exp1_road_r")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_exp1_dataset = _load_module("exp4_exp1_dataset", _EXP1_DIR / "dataset.py")
_ROADWaymoDatasetBase = _exp1_dataset.ROADWaymoDataset


class ROADWaymoDataset(_ROADWaymoDatasetBase):
    """Overrides exp1's _parse_frame to fix annotation-id remapping.

    The ROAD-Waymo annotation JSON stores `*_ids` indexing into the MASTER
    label lists `all_*_labels` (agent: 11, duplex: 152, triplet: 1620), not
    the curated `*_labels` (agent: 10, duplex: 49, triplet: 86) the model
    predicts over. The baseline does this remap via `filter_labels`; exp1's
    original parser just clamps `0 <= i < n_used`, which (a) silently drops
    out-of-range ids and (b) mis-labels in-range ids when positions don't
    align (duplex aligns only on positions 0–10; triplet never aligns).

    Concrete effect verified on a full-val baseline eval:
        duplex f-mAP fell from published 13.44% to 3.27% under the buggy GT.
        triplet f-mAP came in at 8.77% vs published 9.17% (closer because the
        triplet is computed from agent×action×loc cross-product not from
        triplet_ids; only the agent OthTL drop hurts it).

    This subclass remaps by NAME (the safe path the baseline uses).
    """

    def _build_clip_list(self, split, stride, seed):
        """Match baseline's val-time clip construction: start from END of video
        and decrement by stride. This catches the tail of each video that
        exp1's forward-walking misses; reproduces baseline's 1188 val clips
        (vs exp1's 1158 — 30 video-tail clips were missed)."""
        import random as _random
        clips = []
        for vname, vdata in self.db.items():
            split_ids = vdata.get("split_ids", [])
            if isinstance(split_ids, str):
                split_ids = [split_ids]
            is_train = "train" in split_ids or ("all" in split_ids and "val" not in split_ids)
            is_val   = "val" in split_ids
            if split == "train" and not is_train: continue
            if split == "val" and not is_val: continue

            # Use `numf` (full video length per the JSON metadata), NOT
            # len(frames). 25 of 198 val videos have numf > len(frames) — the
            # frames dict is incomplete for the video tail. Baseline reads
            # frame images from disk for ALL raw frame nums regardless of JSON
            # coverage; tail frames produce "no GT" entries which become FPs
            # in eval (apples to apples with baseline scoring). Frame IDs are
            # 1-indexed in the JSON / on disk.
            numf = vdata.get("numf", 0)
            if numf < self.clip_len:
                continue
            for start in range(numf - self.clip_len, 1, -stride):
                # frame_num goes from `start` to `start + clip_len - 1` (1-indexed).
                clip_fids = [str(start + i) for i in range(self.clip_len)]
                clips.append((vname, clip_fids))
        rng = _random.Random(seed)
        rng.shuffle(clips)
        return clips

    def __init__(self, *args, **kwargs):
        import json as _json
        super().__init__(*args, **kwargs)
        anno_file = kwargs.get("anno_file") or args[0]
        with open(anno_file) as f:
            data = _json.load(f)
        # Master lists (the index space the annotations actually use).
        self._all_agent_labels  = data["all_agent_labels"]
        self._all_duplex_labels = data["all_duplex_labels"]
        # action and loc use position-aligned `used` and `all` lists — no remap.
        # Triplet is computed from agent×action×loc cross-product so it
        # inherits the agent remap automatically.

        # Name → curated-index lookups.
        self._used_agent_idx  = {n: i for i, n in enumerate(self.agent_labels)}
        self._used_duplex_idx = {n: i for i, n in enumerate(self.duplex_labels)}

    def _parse_frame(self, frame_data):
        boxes_list = []
        agent_list, action_list, loc_list, duplex_list, triplet_list = [], [], [], [], []

        for anno in frame_data.get("annos", {}).values():
            if not isinstance(anno, dict):
                continue
            box = anno.get("box")
            if box is None or len(box) != 4:
                continue
            boxes_list.append(box)

            # ---- Remap by name into the curated-index space ----
            raw_agent_ids  = anno.get("agent_ids",  []) or []
            raw_action_ids = anno.get("action_ids", []) or []
            raw_loc_ids    = anno.get("loc_ids",    []) or []
            raw_duplex_ids = anno.get("duplex_ids", []) or []

            agent_ids = [
                self._used_agent_idx[self._all_agent_labels[i]]
                for i in raw_agent_ids
                if 0 <= i < len(self._all_agent_labels)
                and self._all_agent_labels[i] in self._used_agent_idx
            ]
            action_ids = [i for i in raw_action_ids if 0 <= i < len(self.action_labels)]
            loc_ids    = [i for i in raw_loc_ids    if 0 <= i < len(self.loc_labels)]
            duplex_ids = [
                self._used_duplex_idx[self._all_duplex_labels[i]]
                for i in raw_duplex_ids
                if 0 <= i < len(self._all_duplex_labels)
                and self._all_duplex_labels[i] in self._used_duplex_idx
            ]

            agent_mh   = torch.zeros(len(self.agent_labels))
            action_mh  = torch.zeros(len(self.action_labels))
            loc_mh     = torch.zeros(len(self.loc_labels))
            duplex_mh  = torch.zeros(len(self.duplex_labels))
            triplet_mh = torch.zeros(len(self.triplet_labels))

            for i in agent_ids:  agent_mh[i]  = 1.0
            for i in action_ids: action_mh[i] = 1.0
            for i in loc_ids:    loc_mh[i]    = 1.0
            for i in duplex_ids: duplex_mh[i] = 1.0

            for a_i in agent_ids:
                for ac_i in action_ids:
                    for l_i in loc_ids:
                        key = (
                            self.agent_labels[a_i],
                            self.action_labels[ac_i],
                            self.loc_labels[l_i],
                        )
                        t_i = self._triplet_lookup.get(key)
                        if t_i is not None:
                            triplet_mh[t_i] = 1.0

            agent_list.append(agent_mh)
            action_list.append(action_mh)
            loc_list.append(loc_mh)
            duplex_list.append(duplex_mh)
            triplet_list.append(triplet_mh)

        if not boxes_list:
            return None
        return {
            "boxes":   torch.tensor(boxes_list, dtype=torch.float32),
            "agent":   torch.stack(agent_list),
            "action":  torch.stack(action_list),
            "loc":     torch.stack(loc_list),
            "duplex":  torch.stack(duplex_list),
            "triplet": torch.stack(triplet_list),
        }


# ImageNet normalization (matches RetinaNet baseline's args.MEANS/STDS).
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def clip_to_tensor(pil_frames: List[Image.Image],
                   short_side: int = C.VAL_SHORT_SIDE,
                   max_size:   int = C.VAL_MAX_SIZE) -> Tuple[torch.Tensor, int, int]:
    """
    PIL frames → tensor clip [T, 3, H, W], **fixed** 600×840 (h, w) to match the
    baseline RetinaNet's training distribution exactly.

    baseline/data/transforms.py::ResizeClip_Fixed resizes every frame to
    F.resize(img, (MIN_SIZE=600, MAX_SIZE=840)), which torchvision interprets
    as (h, w) — distorting aspect ratio. ROAD-Waymo native is 1280×1920 (h, w);
    after this resize all clips are 600×840.

    Returns (clip, H=600, W=840).
    """
    assert len(pil_frames) > 0
    h, w = short_side, max_size
    tensors = []
    for f in pil_frames:
        f = TF.resize(f, [h, w], antialias=True)                 # distorts aspect ratio
        t = TF.to_tensor(f)
        t = TF.normalize(t, _IMAGENET_MEAN, _IMAGENET_STD)
        tensors.append(t)
    clip = torch.stack(tensors, dim=0)                           # [T, 3, H, W]
    return clip, h, w


def rescale_frame_targets(frame_targets: List[Optional[dict]],
                          h: int, w: int) -> List[Optional[dict]]:
    """Convert each per-frame box tensor from normalized [0,1] xyxy to pixel xyxy
    in the *resized* image space. ROAD-Waymo stores boxes as fractions of the
    original frame; after resize we need pixel coords to match the model's
    pred-box convention (pixel xyxy in the resized clip)."""
    scale = torch.tensor([w, h, w, h], dtype=torch.float32)
    out: List[Optional[dict]] = []
    for ft in frame_targets:
        if ft is None:
            out.append(None)
            continue
        new = dict(ft)
        new["boxes"] = ft["boxes"] * scale
        out.append(new)
    return out


def exp4_collate_fn(batch):
    """
    DataLoader collate for batch_size=1 (matches exp2f/2g convention).

    batch is a list of 1 sample = (pil_frames, frame_targets).
    Returns (clip [1, T, 3, H, W], frame_targets) — frame_targets is the rescaled
    list, ready to feed into greedy_group_tubes().
    """
    assert len(batch) == 1
    pil_frames, frame_targets = batch[0]
    clip, h, w = clip_to_tensor(pil_frames)
    rescaled = rescale_frame_targets(frame_targets, h, w)
    return clip.unsqueeze(0), rescaled


def build_dataloader(split: str = "train", batch_size: int = 1,
                     num_workers: int = 4, shuffle: Optional[bool] = None) -> DataLoader:
    """
    Convenience factory. split ∈ {"train", "val"}.
    Shuffle defaults to (split == "train").
    """
    if shuffle is None:
        shuffle = (split == "train")
    ds = ROADWaymoDataset(
        anno_file=C.ANNO_FILE,
        frames_dir=C.FRAMES_DIR,
        split=split,
        clip_len=C.CLIP_LEN,
        stride=C.CLIP_STRIDE,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=exp4_collate_fn,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


if __name__ == "__main__":
    # Quick smoke for the dataloader itself.
    loader = build_dataloader(split="val", num_workers=0)
    print(f"[dataloader smoke] val clips: {len(loader.dataset)}")
    clip, frame_targets = next(iter(loader))
    print(f"[dataloader smoke] clip: {tuple(clip.shape)} {clip.dtype}")
    print(f"[dataloader smoke] frame_targets: len={len(frame_targets)}")
    n_with_boxes = sum(1 for ft in frame_targets if ft is not None)
    print(f"[dataloader smoke] frames with GT boxes: {n_with_boxes}/{len(frame_targets)}")
    for i, ft in enumerate(frame_targets):
        if ft is not None:
            print(f"  frame {i}: boxes={tuple(ft['boxes'].shape)}, "
                  f"first_box={ft['boxes'][0].tolist()}")
            break

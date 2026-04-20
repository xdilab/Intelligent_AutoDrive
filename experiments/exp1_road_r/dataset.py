"""
ROAD-Waymo clip dataset for Experiment 1.

Each sample is CLIP_LEN consecutive annotated frames from one video.
Returns PIL frames + per-frame GT annotation dicts (boxes + multi-hot labels).

Label encoding
--------------
agent:   [10]  multi-hot, from anno['agent_ids']
action:  [22]  multi-hot, from anno['action_ids']
loc:     [16]  multi-hot, from anno['loc_ids']
duplex:  [49]  multi-hot, from anno['duplex_ids']
triplet: [86]  multi-hot, constructed from (agent_name, action_name, loc_name) lookup

Note on triplet_ids in the annotation JSON: the raw values are not 0-indexed into
triplet_labels (they span a much larger space). We reconstruct triplet targets by
matching (agent_name, action_name, loc_name) against triplet_labels strings.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


class ROADWaymoDataset(Dataset):
    """
    Samples T consecutive annotated frames from ROAD-Waymo.

    Args:
        anno_file:  path to road_waymo_trainval_v1.1.json
        frames_dir: path to rgb-images/
        split:      'train' or 'val'
        clip_len:   number of frames per sample
        stride:     step between clip start indices (controls overlap)
        seed:       random seed for shuffling
    """

    def __init__(
        self,
        anno_file: str,
        frames_dir: str,
        split: str = "train",
        clip_len: int = 8,
        stride: int = 4,
        seed: int = 42,
    ):
        with open(anno_file) as f:
            data = json.load(f)

        self.db             = data["db"]
        self.agent_labels   = data["agent_labels"]    # 10
        self.action_labels  = data["action_labels"]   # 22
        self.loc_labels     = data["loc_labels"]      # 16
        self.duplex_labels  = data["duplex_labels"]   # 49
        self.triplet_labels = data["triplet_labels"]  # 86
        self.duplex_childs  = data["duplex_childs"]   # list of [agent_idx, action_idx]
        self.triplet_childs = data["triplet_childs"]  # list of [agent_idx, action_idx, loc_idx]

        self.frames_dir = Path(frames_dir)
        self.clip_len   = clip_len
        self.split      = split

        # (agent_name, action_name, loc_name) → triplet index (0-85)
        self._triplet_lookup = self._build_triplet_lookup()

        # list of (video_name, [frame_id_0, ..., frame_id_{T-1}])
        self.clips = self._build_clip_list(split, stride, seed)

    # ── Lookup construction ────────────────────────────────────────────────────

    def _build_triplet_lookup(self) -> Dict[Tuple[str, str, str], int]:
        """
        Parse triplet_labels strings (e.g. 'Ped-MovAway-LftPav') into
        (agent_name, action_name, loc_name) tuples and map to 0-based index.

        Greedy left-to-right matching handles multi-word label names such as
        'XingFmLft', 'OutgoBusLane', etc.
        """
        loc_set = set(self.loc_labels)
        lookup: Dict[Tuple[str, str, str], int] = {}

        for idx, triplet_str in enumerate(self.triplet_labels):
            matched = False
            for agt in self.agent_labels:
                prefix_a = agt + "-"
                if not triplet_str.startswith(prefix_a):
                    continue
                rest = triplet_str[len(prefix_a):]
                for act in self.action_labels:
                    prefix_ac = act + "-"
                    if not rest.startswith(prefix_ac):
                        continue
                    loc = rest[len(prefix_ac):]
                    if loc in loc_set:
                        lookup[(agt, act, loc)] = idx
                        matched = True
                        break
                if matched:
                    break

        return lookup

    # ── Clip list construction ─────────────────────────────────────────────────

    def _build_clip_list(
        self,
        split: str,
        stride: int,
        seed: int,
    ) -> List[Tuple[str, List[str]]]:
        clips = []

        for vname, vdata in self.db.items():
            split_ids = vdata.get("split_ids", [])
            if isinstance(split_ids, str):
                split_ids = [split_ids]

            # ROAD-Waymo marks every video 'all'; train/val split determined by split_ids list.
            # Videos with only 'all' (no 'train'/'val') go to train by default.
            is_train = "train" in split_ids or ("all" in split_ids and "val" not in split_ids)
            is_val   = "val"   in split_ids

            if split == "train" and not is_train:
                continue
            if split == "val" and not is_val:
                continue

            frames = vdata.get("frames", {})
            annotated_fids = sorted(
                (fid for fid, fd in frames.items() if fd.get("annotated", 0) == 1),
                key=lambda x: int(x),
            )

            if len(annotated_fids) < self.clip_len:
                continue

            for start in range(0, len(annotated_fids) - self.clip_len + 1, stride):
                clip_fids = annotated_fids[start : start + self.clip_len]
                clips.append((vname, clip_fids))

        rng = random.Random(seed)
        rng.shuffle(clips)
        return clips

    # ── Per-frame annotation parsing ───────────────────────────────────────────

    def _parse_frame(self, frame_data: dict) -> Optional[Dict[str, torch.Tensor]]:
        """
        Convert a frame's annotation dict to per-agent label tensors.

        Returns None if the frame has no valid annotations (no boxes).
        Returns dict with:
            boxes:   [n, 4] float32   normalized [x1, y1, x2, y2]
            agent:   [n, 10] float32  multi-hot
            action:  [n, 22] float32  multi-hot
            loc:     [n, 16] float32  multi-hot
            duplex:  [n, 49] float32  multi-hot
            triplet: [n, 86] float32  multi-hot
        """
        boxes_list    = []
        agent_list    = []
        action_list   = []
        loc_list      = []
        duplex_list   = []
        triplet_list  = []

        for anno in frame_data.get("annos", {}).values():
            if not isinstance(anno, dict):
                continue
            box = anno.get("box")
            if box is None or len(box) != 4:
                continue

            boxes_list.append(box)

            agent_ids  = [i for i in anno.get("agent_ids",  []) if 0 <= i < 10]
            action_ids = [i for i in anno.get("action_ids", []) if 0 <= i < 22]
            loc_ids    = [i for i in anno.get("loc_ids",    []) if 0 <= i < 16]
            duplex_ids = [i for i in anno.get("duplex_ids", []) if 0 <= i < 49]

            agent_mh   = torch.zeros(10)
            action_mh  = torch.zeros(22)
            loc_mh     = torch.zeros(16)
            duplex_mh  = torch.zeros(49)
            triplet_mh = torch.zeros(86)

            for i in agent_ids:  agent_mh[i]  = 1.0
            for i in action_ids: action_mh[i] = 1.0
            for i in loc_ids:    loc_mh[i]    = 1.0
            for i in duplex_ids: duplex_mh[i] = 1.0

            # Triplet: all valid (agent, action, loc) cross-products present in lookup
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

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int):
        """
        Returns:
            pil_frames:    list of T PIL.Image.Image (RGB)
            frame_targets: list of T Optional[dict] — None for frames with no GT boxes
        """
        vname, fids = self.clips[idx]
        vdata = self.db[vname]
        frames_db = vdata["frames"]

        pil_frames    = []
        frame_targets = []

        for fid in fids:
            img_path = self.frames_dir / vname / f"{int(fid):05d}.jpg"
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                # Fallback: black frame (shouldn't happen with a complete dataset)
                w, h = 640, 480
                img = Image.new("RGB", (w, h))

            pil_frames.append(img)

            fdata  = frames_db.get(fid, {})
            target = self._parse_frame(fdata)
            frame_targets.append(target)

        return pil_frames, frame_targets

"""Exp8 stage 1 — build the three SFT corpora + combined joint files.

  road:  frames from exp6's detections_train.pkl, shards {0,1} of 12 (train)
         and shard 11 (SFT-val, 500 evenly subsampled). Each sample: the
         native frame with the frozen detector's top-40 boxes drawn (exact
         exp5 deployment rendering), target = GT agent/actions/locations per
         box via IoU>=0.5 greedy-argmax assignment (exp6 rule); unmatched
         boxes get agent null.
  covla: CoVLA-mini per-frame captions, stride 12 (0.6 s), last 5 scenes held
         out as SFT-val. Target = {caption: plain_caption, risk} verbatim.
  bddx:  exp7's temporally-aligned jsons reused verbatim (4,339 / 545).

Outputs (cache/): road_sft_{train,val}.json, covla_sft_{train,val}.json,
joint_sft_{train,val}.json (block order bddx|covla|road), joint_sft_stats.json
(block sizes — read by train_joint.py's round-robin sampler).

Usage:
  /home/brandon/miniconda3/bin/python -u data_prep.py [--overwrite] [--workers N]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import tarfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

import config as C

sys.path.insert(0, str(C.EXP5_DIR))
import vlm_io  # noqa: E402  (box rendering — byte-identical to deployment)


# --------------------------------------------------------------------------- #
# ROAD-Waymo                                                                  #
# --------------------------------------------------------------------------- #
def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Same as exp6/dataset.py — [n,4]x[m,4] xyxy IoU."""
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _road_frame_path(key: str) -> Path:
    vname, fid = key.rsplit("/", 1)
    return Path(C.ROAD_FRAMES_DIR) / vname / f"{int(fid):05d}.jpg"


def _render_road_frame(job: tuple) -> tuple:
    """Worker: draw boxes on the native frame, save jpg. Returns (key, ok)."""
    key, boxes_600, out_path = job
    fpath = _road_frame_path(key)
    if not fpath.exists():
        return key, False
    img = Image.open(fpath).convert("RGB")
    nw, nh = img.size
    sx, sy = nw / C.ROAD_BOX_W, nh / C.ROAD_BOX_H          # 600x840 -> native
    boxes_native = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in boxes_600]
    drawn = vlm_io.draw_numbered_boxes(img, boxes_native)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    drawn.save(out_path, quality=95)
    return key, True


def _road_targets(rec: dict, labels: dict, n_boxes: int) -> list:
    """Per-box GT via IoU>=0.5 greedy-argmax (exp6 assignment, name-level)."""
    boxes = rec["boxes"][:n_boxes]
    out = []
    gt = rec["gt"]
    matched = np.zeros(n_boxes, dtype=bool)
    best = np.zeros(n_boxes, dtype=int)
    if gt is not None and gt["boxes"].shape[0] > 0:
        iou = _iou_matrix(boxes, gt["boxes"])
        best = iou.argmax(axis=1)
        matched = iou[np.arange(n_boxes), best] >= C.IOU_MATCH_THRESH
    for i in range(n_boxes):
        if matched[i]:
            g = int(best[i])
            agent_ids = np.nonzero(gt["agent"][g])[0]
            out.append({
                "box_id": i,
                "agent": labels["agent"][int(agent_ids[0])] if len(agent_ids) else None,
                "actions": [labels["action"][j] for j in np.nonzero(gt["action"][g])[0]],
                "locations": [labels["loc"][j] for j in np.nonzero(gt["loc"][g])[0]],
            })
        else:
            out.append({"box_id": i, "agent": None, "actions": [], "locations": []})
    return out


def build_road(workers: int) -> dict:
    print(f"[prep:road] loading {C.ROAD_DET_PKL.name} ...")
    with open(C.ROAD_DET_PKL, "rb") as f:
        payload = pickle.load(f)
    records, labels = payload["records"], payload["labels"]
    keys = sorted(records.keys())
    agents = ", ".join(labels["agent"])
    actions = ", ".join(labels["action"])
    locs = ", ".join(labels["loc"])

    splits = {}
    train_keys = [k for i, k in enumerate(keys) if i % C.ROAD_NUM_SHARDS in C.ROAD_TRAIN_SHARDS]
    val_pool = [k for i, k in enumerate(keys) if i % C.ROAD_NUM_SHARDS == C.ROAD_VAL_SHARD]
    step = max(1, len(val_pool) // C.ROAD_VAL_N)
    splits["train"] = (train_keys, C.ROAD_TRAIN_JSON)
    splits["val"] = (val_pool[::step][: C.ROAD_VAL_N], C.ROAD_VAL_JSON)

    stats = {}
    for split, (skeys, out_json) in splits.items():
        img_dir = C.ROAD_IMG_DIR / split
        jobs, meta = [], []
        for key in skeys:
            rec = records[key]
            boxes = rec["boxes"][: C.MAX_BOXES_PER_FRAME]
            n_boxes = int(boxes.shape[0])
            if n_boxes == 0:
                continue
            out_path = img_dir / (key.replace("/", "_") + ".jpg")
            jobs.append((key, boxes, out_path))
            meta.append((key, n_boxes, out_path))
        print(f"[prep:road] {split}: rendering {len(jobs):,} frames "
              f"({workers} workers) ...")
        with Pool(workers) as pool:
            ok = dict(pool.map(_render_road_frame, jobs, chunksize=32))
        samples = []
        n_missing = 0
        for key, n_boxes, out_path in meta:
            if not ok.get(key):
                n_missing += 1
                continue
            samples.append({
                "image": str(out_path),
                "conversations": [
                    {"from": "human",
                     "value": "<image>\n" + C.build_road_prompt(n_boxes, agents, actions, locs)},
                    {"from": "gpt",
                     "value": C.build_road_target(
                         _road_targets(records[key], labels, n_boxes))},
                ],
                "road_key": key,
            })
        with open(out_json, "w") as f:
            json.dump(samples, f, ensure_ascii=False)
        stats[split] = {"samples": len(samples), "missing_frames": n_missing}
        print(f"[prep:road] {split}: {len(samples):,} samples "
              f"({n_missing} frames missing on disk) -> {out_json.name}")
    return stats


# --------------------------------------------------------------------------- #
# CoVLA                                                                       #
# --------------------------------------------------------------------------- #
def _read_caption_records(path: Path) -> list:
    """Captions are concatenated JSON objects (not line-delimited)."""
    raw = path.read_text()
    dec = json.JSONDecoder()
    recs, i = [], 0
    while i < len(raw):
        while i < len(raw) and raw[i] in " \n\r\t":
            i += 1
        if i >= len(raw):
            break
        obj, i = dec.raw_decode(raw, i)
        recs.append(obj)
    return recs


def build_covla() -> dict:
    scenes = sorted(p.stem.replace(".jsonl", "") for p in C.COVLA_CAPTIONS_DIR.glob("*.jsonl"))
    assert scenes, f"no caption files under {C.COVLA_CAPTIONS_DIR}"
    split_scenes = {"train": scenes[: -C.COVLA_VAL_SCENES],
                    "val": scenes[-C.COVLA_VAL_SCENES:]}
    stats = {}
    for split, snames in split_scenes.items():
        out_json = {"train": C.COVLA_TRAIN_JSON, "val": C.COVLA_VAL_JSON}[split]
        samples = []
        for scene in snames:
            caps = _read_caption_records(C.COVLA_CAPTIONS_DIR / f"{scene}.jsonl")
            tar_path = C.COVLA_IMAGES_DIR / f"{scene}.tar.gz"
            assert tar_path.exists(), f"missing {tar_path}"
            wanted = {f"images/{scene}/{i:04d}.png": i
                      for i in range(0, len(caps), C.COVLA_FRAME_STRIDE)}
            out_dir = C.COVLA_IMG_DIR / scene
            todo = {n: i for n, i in wanted.items()
                    if not (out_dir / Path(n).name).exists()}
            if todo:
                out_dir.mkdir(parents=True, exist_ok=True)
                with tarfile.open(tar_path, "r:gz") as tf:
                    for member in tf:
                        if member.name in todo:
                            with tf.extractfile(member) as src:
                                (out_dir / Path(member.name).name).write_bytes(src.read())
            n_bad = 0
            for name, idx in sorted(wanted.items(), key=lambda kv: kv[1]):
                img_path = out_dir / Path(name).name
                rec = caps[idx]
                cap, risk = rec.get("plain_caption", ""), rec.get("risk", "")
                if not img_path.exists() or not (cap and risk):
                    n_bad += 1
                    continue
                samples.append({
                    "image": str(img_path),
                    "conversations": [
                        {"from": "human", "value": "<image>\n" + C.COVLA_USER_PROMPT},
                        {"from": "gpt", "value": C.build_covla_target(cap, risk)},
                    ],
                    "covla_scene": scene,
                    "covla_frame": idx,
                })
            if n_bad:
                print(f"[prep:covla] {scene}: dropped {n_bad} frames "
                      f"(missing image or empty caption/risk)")
        with open(out_json, "w") as f:
            json.dump(samples, f, ensure_ascii=False)
        stats[split] = {"samples": len(samples), "scenes": len(snames)}
        print(f"[prep:covla] {split}: {len(samples):,} samples "
              f"from {len(snames)} scenes -> {out_json.name}")
    return stats


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    done = all(p.exists() for p in (C.JOINT_TRAIN_JSON, C.JOINT_VAL_JSON, C.JOINT_STATS_JSON))
    if done and not args.overwrite:
        raise SystemExit("[prep] joint files exist — use --overwrite to rebuild")

    for p in (C.BDDX_TRAIN_JSON, C.BDDX_VAL_JSON):
        assert p.exists(), f"{p} missing — run exp7 data_prep.py first"

    road_stats = build_road(args.workers)
    covla_stats = build_covla()

    # Combined block-concatenated files, order per C.BLOCK_ORDER.
    stats = {"block_order": list(C.BLOCK_ORDER), "road": road_stats, "covla": covla_stats}
    for split, out in (("train", C.JOINT_TRAIN_JSON), ("val", C.JOINT_VAL_JSON)):
        blocks = {
            "bddx": {"train": C.BDDX_TRAIN_JSON, "val": C.BDDX_VAL_JSON},
            "covla": {"train": C.COVLA_TRAIN_JSON, "val": C.COVLA_VAL_JSON},
            "road": {"train": C.ROAD_TRAIN_JSON, "val": C.ROAD_VAL_JSON},
        }
        combined, sizes = [], {}
        for name in C.BLOCK_ORDER:
            block = json.loads(blocks[name][split].read_text())
            sizes[name] = len(block)
            combined.extend(block)
        with open(out, "w") as f:
            json.dump(combined, f, ensure_ascii=False)
        stats[f"{split}_block_sizes"] = sizes
        print(f"[prep] joint {split}: {sizes} -> {out.name}")

    with open(C.JOINT_STATS_JSON, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[prep] stats -> {C.JOINT_STATS_JSON.name}")
    print("[prep] done")


if __name__ == "__main__":
    main()

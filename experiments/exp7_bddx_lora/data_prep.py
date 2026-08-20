"""Exp7 stage 1 — build the BDD-X SFT dataset from local files.

Input (all local, nothing downloaded):
  /data/datasets/BDD-X/BDD-X-Annotations_v1.csv   wide CSV, 15 action slots/row
  /data/datasets/BDD-X/{train,val}.txt            split stems ("{n}_{stem}")
  /data/datasets/bdd100k-yolopx/images/{train,val,test}/<stem>.jpg
                                                  BDD100K t=10 s still frames

Filter (both must hold — no synthetic pairing):
  1. the segment has non-empty action AND justification text;
  2. the segment's [start, end] window contains t=10 s, the timestamp of the
     one locally-available frame, so image and text describe the same moment.

Output: annotation JSON in the reference repo's LLaVA-style format
(qwen-vl-finetune qwenvl/data/data_processor.py::_build_messages):
  [{"image": "<abs path>", "conversations":
      [{"from": "human", "value": "<image>\\n<prompt>"},
       {"from": "gpt",   "value": "{\\"action\\": ..., \\"justification\\": ...}"}]}]

Cached: skips a split whose JSON already exists (use --overwrite to rebuild).

Usage:
  /home/brandon/miniconda3/bin/python -u data_prep.py [--overwrite] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import config as C


def _load_split_stems(split_file: Path) -> set[str]:
    """Split lines look like '1_06d501fd-a9ffc960' — strip the '{n}_' prefix."""
    stems = set()
    with open(split_file) as f:
        for line in f:
            line = line.strip()
            if line:
                stems.add("_".join(line.split("_")[1:]))
    return stems


def _index_images() -> dict[str, str]:
    """stem -> absolute jpg path across all BDD100K image dirs."""
    idx: dict[str, str] = {}
    for d in C.BDD100K_IMG_DIRS:
        if not d.is_dir():
            continue
        for fn in d.iterdir():
            if fn.suffix == ".jpg":
                idx.setdefault(fn.stem, str(fn))
    return idx


def build_samples(split: str, limit: int = 0) -> tuple[list[dict], dict]:
    """Parse the CSV and return (samples, stats) for one split."""
    stems = _load_split_stems(C.BDDX_SPLIT_FILES[split])
    img_idx = _index_images()

    samples: list[dict] = []
    stats = {
        "split": split,
        "videos_in_split": len(stems),
        "videos_seen_in_csv": 0,
        "videos_with_image": 0,
        "segments_total": 0,
        "segments_no_image": 0,
        "segments_bad_time": 0,
        "segments_not_covering_t10": 0,
        "samples": 0,
    }

    with open(C.BDDX_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url = row.get("Input.Video", "").strip()
            if not url:
                continue
            stem = Path(url).stem
            if stem not in stems:
                continue
            stats["videos_seen_in_csv"] += 1
            img_path = img_idx.get(stem)
            if img_path:
                stats["videos_with_image"] += 1

            for i in range(1, 16):
                action = row.get(f"Answer.{i}action", "").strip()
                justification = row.get(f"Answer.{i}justification", "").strip()
                if not (action and justification):
                    continue
                stats["segments_total"] += 1
                if img_path is None:
                    stats["segments_no_image"] += 1
                    continue
                try:
                    t0 = float(row.get(f"Answer.{i}start", ""))
                    t1 = float(row.get(f"Answer.{i}end", ""))
                except ValueError:
                    stats["segments_bad_time"] += 1
                    continue
                if not (t0 <= C.FRAME_TIME_S <= t1):
                    stats["segments_not_covering_t10"] += 1
                    continue
                samples.append(
                    {
                        "image": img_path,
                        "conversations": [
                            {"from": "human", "value": "<image>\n" + C.USER_PROMPT},
                            {"from": "gpt", "value": C.build_target(action, justification)},
                        ],
                        # bookkeeping fields (ignored by the reference loader)
                        "bddx_video": stem,
                        "bddx_segment": [t0, t1],
                    }
                )
                if limit and len(samples) >= limit:
                    stats["samples"] = len(samples)
                    return samples, stats

    stats["samples"] = len(samples)
    return samples, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Cap samples per split (debug). 0 = all.")
    args = ap.parse_args()

    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_stats = {}
    outs = {"train": C.SFT_TRAIN_JSON, "val": C.SFT_VAL_JSON}
    for split, out in outs.items():
        if out.exists() and not args.overwrite:
            print(f"[prep] {out.name} exists — skipping (use --overwrite to rebuild)")
            continue
        print(f"[prep] building {split} ...")
        samples, stats = build_samples(split, limit=args.limit)
        if not samples:
            raise SystemExit(f"[prep] FATAL: 0 samples for split '{split}' — check paths in config.py")
        with open(out, "w") as f:
            json.dump(samples, f, ensure_ascii=False, indent=1)
        all_stats[split] = stats
        print(f"[prep] {split}: {stats['samples']:,} samples "
              f"({stats['videos_with_image']}/{stats['videos_seen_in_csv']} videos have a frame; "
              f"dropped: {stats['segments_no_image']} no-image, "
              f"{stats['segments_not_covering_t10']} outside t=10s window, "
              f"{stats['segments_bad_time']} bad timestamps) -> {out}")

    if all_stats:
        with open(C.SFT_STATS_JSON, "w") as f:
            json.dump(all_stats, f, indent=2)
        print(f"[prep] stats -> {C.SFT_STATS_JSON}")
    print("[prep] done")


if __name__ == "__main__":
    main()

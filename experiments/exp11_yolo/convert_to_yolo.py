"""ROAD-Waymo -> Ultralytics YOLO dataset conversion (10-class agent detection).

Source: road_waymo_trainval_v1.1.json (same file as exp1..exp9) + rgb-images.
Output: /data/datasets/road_waymo_yolo/
  images/{train,val}/<video>_<frame:05d>.jpg   (symlinks into rgb-images)
  labels/{train,val}/<video>_<frame:05d>.txt   (class cx cy w h, normalized)
  data.yaml

Class ids follow the benchmark `agent_labels` order (Ped=0 ... TL=9).
`agent_ids` index `all_agent_labels` (11 entries incl. OthTL) -- remapped by
NAME to the 10-class vocab; names outside it are dropped (0 boxes in v1.1).
Annotated frames with zero annos get an empty label file (background negatives).
Boxes in the JSON are already normalized xyxy.
"""

import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

ANNO = "/data/datasets/ROAD_plusplus/road_waymo_trainval_v1.1.json"
FRAMES = Path("/data/datasets/ROAD_plusplus/rgb-images")
OUT = Path("/data/datasets/road_waymo_yolo")

d = json.load(open(ANNO))
AGENTS = d["agent_labels"]            # 10-class benchmark vocab
ALL_AGENTS = d["all_agent_labels"]    # 11-entry list that agent_ids index
NAME2CLS = {n: i for i, n in enumerate(AGENTS)}

for sp in ("train", "val"):
    (OUT / "images" / sp).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / sp).mkdir(parents=True, exist_ok=True)

stats = {"frames": 0, "boxes": 0, "empty": 0, "no_jpg": 0,
         "dropped_name": 0, "degenerate": 0}

for vname, vid in sorted(d["db"].items()):
    split = "train" if "train" in vid["split_ids"] else "val"
    for fkey, fr in vid["frames"].items():
        if not fr.get("annotated"):
            continue
        jpg = FRAMES / vname / f"{int(fkey):05d}.jpg"
        if not jpg.exists():
            stats["no_jpg"] += 1
            continue
        stem = f"{vname}_{int(fkey):05d}"
        lines = []
        for a in fr.get("annos", {}).values():
            x1, y1, x2, y2 = a["box"]
            x1 = min(max(x1, 0.0), 1.0); y1 = min(max(y1, 0.0), 1.0)
            x2 = min(max(x2, 0.0), 1.0); y2 = min(max(y2, 0.0), 1.0)
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                stats["degenerate"] += 1
                continue
            cx, cy = x1 + w / 2, y1 + h / 2
            for aid in a["agent_ids"]:
                name = ALL_AGENTS[aid]
                cls = NAME2CLS.get(name)
                if cls is None:
                    stats["dropped_name"] += 1
                    continue
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                stats["boxes"] += 1
        (OUT / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""))
        link = OUT / "images" / split / f"{stem}.jpg"
        if not link.exists():
            os.symlink(jpg, link)
        stats["frames"] += 1
        if not lines:
            stats["empty"] += 1
        if stats["frames"] % 10000 == 0:
            print(f"[convert] {stats['frames']:,} frames | "
                  f"{stats['boxes']:,} boxes", flush=True)

yaml = OUT / "data.yaml"
yaml.write_text(
    f"path: {OUT}\ntrain: images/train\nval: images/val\n\nnames:\n"
    + "".join(f"  {i}: {n}\n" for i, n in enumerate(AGENTS)))
print(f"[convert] DONE {stats}")
print(f"[convert] wrote {yaml}")

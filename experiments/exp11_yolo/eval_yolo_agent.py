"""Score a YOLO checkpoint's agent f-mAP under the baseline evaluator.

Protocol: FULL val split (every annotated frame), full-candidate detections
(conf 0.001, NMS iou 0.7, max_det 300 — the confirmed new-row protocol),
baseline evaluate() at IoU=0.5, normalized coords for both GT and dets.
Comparators: baseline 3D-RetinaNet 17.76 agent f-mAP (all-anchor, full val),
ECCV'24 report YOLOv8 reference 31.6 (their internal 75/25 split — approximate
external anchor, not identical rows).

Runs on GPU 0 alongside training: batch 1, hard memory cap so an OOM kills
this process, not the training run.

Usage: python -u eval_yolo_agent.py --ckpt runs/.../best.pt --out out.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

ANNO = "/data/datasets/ROAD_plusplus/road_waymo_yolo_anno_cache.npz"  # unused; kept simple
JSON_PATH = "/data/datasets/ROAD_plusplus/road_waymo_trainval_v1.1.json"
IMG_DIR = Path("/data/datasets/road_waymo_yolo/images/val")

sys.path.insert(0, "/data/repos/PedestrianIntent++/ROAD_plus_plus_Baseline")
import modules.evaluation as baseline_eval  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", default=None)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--memfrac", type=float, default=0.07)
ap.add_argument("--dump", default=None, help="pkl path to save per-frame full-candidate dets")
args = ap.parse_args()

torch.cuda.set_per_process_memory_fraction(args.memfrac, 0)

import json as _j
d = _j.load(open(JSON_PATH))
AGENTS = d["agent_labels"]; ALL = d["all_agent_labels"]
NAME2CLS = {n: i for i, n in enumerate(AGENTS)}

# GT per val frame, normalized xyxy + agent class rows
gt_of = {}
for vname, vid in d["db"].items():
    if "val" not in vid["split_ids"]:
        continue
    for fk, fr in vid["frames"].items():
        if not fr.get("annotated"):
            continue
        rows = []
        for a in fr.get("annos", {}).values():
            x1, y1, x2, y2 = [min(max(z, 0.0), 1.0) for z in a["box"]]
            if x2 <= x1 or y2 <= y1:
                continue
            for aid in a["agent_ids"]:
                c = NAME2CLS.get(ALL[aid])
                if c is not None:
                    rows.append([x1, y1, x2, y2, c])
        gt_of[f"{vname}_{int(fk):05d}"] = np.array(rows, dtype=np.float32).reshape(-1, 5)

stems = sorted(gt_of)
if args.limit:
    stems = stems[: args.limit]
print(f"[eval] {len(stems):,} val frames | ckpt {args.ckpt}", flush=True)

from ultralytics import YOLO
model = YOLO(args.ckpt)

gt_all = [[]]
det_all = [[[] for _ in range(len(AGENTS))]]
dump = {} if args.dump else None
t0 = time.time()
for i, stem in enumerate(stems):
    r = model.predict(IMG_DIR / f"{stem}.jpg", imgsz=1280, conf=0.001, iou=0.7,
                      max_det=300, device=0, half=True, verbose=False)[0]
    xy = r.boxes.xyxyn.cpu().numpy().astype(np.float32)
    cls = r.boxes.cls.cpu().numpy().astype(int)
    conf = r.boxes.conf.cpu().numpy().astype(np.float32)
    if dump is not None:
        dump[stem] = {"boxes_xyxyn": xy, "cls": cls, "conf": conf}
    g = gt_of[stem]
    gt5 = np.zeros((g.shape[0], 5), dtype=np.float32)
    gt5[:, :4] = g[:, :4]; gt5[:, 4] = g[:, 4]
    gt_all[0].append(gt5)
    for c in range(len(AGENTS)):
        m = cls == c
        det_all[0][c].append(
            np.concatenate([xy[m], conf[m, None]], axis=1).astype(np.float32))
    if (i + 1) % 2000 == 0:
        print(f"[eval] {i + 1}/{len(stems)} | {(time.time()-t0)/(i+1):.3f}s/frame",
              flush=True)

print(f"[eval] running baseline f-mAP over {len(stems):,} frames", flush=True)
mAP, _, ap_strs = baseline_eval.evaluate(gt_all, det_all, [AGENTS], iou_thresh=0.5)
print(f"\n=== agent f-mAP: {mAP[0]:.4f}% ===")
for s in ap_strs[0]:
    print("  ", s)
if args.dump:
    import pickle
    with open(args.dump, "wb") as f:
        pickle.dump({"records": dump, "meta": {"ckpt": args.ckpt,
            "protocol": "full-candidate conf0.001 iou0.7 maxdet300 imgsz1280"}}, f)
    print(f"[eval] dumped {len(dump):,} frames -> {args.dump}", flush=True)
if args.out:
    Path(args.out).write_text(json.dumps(
        {"ckpt": args.ckpt, "agent_fmap": float(mAP[0]),
         "per_class": ap_strs[0], "n_frames": len(stems),
         "protocol": "full-val full-candidate conf0.001 maxdet300 iou0.5"}, indent=2))
    print(f"[eval] wrote {args.out}", flush=True)

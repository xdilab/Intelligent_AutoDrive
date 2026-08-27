"""Dump YOLOv8x best.pt full-candidate detections on the TRAIN split.
Input for junk-negative head retraining. No scoring — dump only."""
import pickle, sys, time
from pathlib import Path
import numpy as np
import torch
sys.stdout.reconfigure(line_buffering=True)
torch.cuda.set_per_process_memory_fraction(0.16, 0)
E = Path("/data/repos/ROAD_Reason/experiments/exp11_yolo")
IMG = Path("/data/datasets/road_waymo_yolo/images/train")
from ultralytics import YOLO
model = YOLO(str(E / "runs/v8x_agent_1280_eccv/weights/best.pt"))
stems = sorted(p.stem for p in IMG.iterdir())
stems = [s2 for i, s2 in enumerate(stems) if i % 12 in (0, 1)]   # exp8/9 shard subset
print(f"[dump] {len(stems):,} train frames", flush=True)
dump = {}
t0 = time.time()
for i, stem in enumerate(stems):
    r = model.predict(IMG / f"{stem}.jpg", imgsz=1280, conf=0.001, iou=0.7,
                      max_det=300, device=0, half=True, verbose=False)[0]
    dump[stem] = {"boxes_xyxyn": r.boxes.xyxyn.cpu().numpy().astype(np.float32),
                  "cls": r.boxes.cls.cpu().numpy().astype(np.int16),
                  "conf": r.boxes.conf.cpu().numpy().astype(np.float32)}
    if (i + 1) % 2000 == 0:
        print(f"[dump] {i+1}/{len(stems)} | {(time.time()-t0)/(i+1):.3f}s/frame", flush=True)
        with open(E / "dets_v8x_best_train_fullcand.pkl.partial", "wb") as f:
            pickle.dump({"records": dump, "meta": {"partial": True}}, f)
with open(E / "dets_v8x_best_train_fullcand.pkl", "wb") as f:
    pickle.dump({"records": dump, "meta": {"protocol": "full-candidate train"}}, f)
print(f"[dump] wrote {len(dump):,} frames ({time.time()-t0:.0f}s)", flush=True)

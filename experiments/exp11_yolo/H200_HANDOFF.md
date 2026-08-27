# Exp11 H200 handoff — YOLO26x on ROAD-Waymo (10-class agent)

Plan: v8x anchor finishes on the A6000 rig; YOLO26x trains on the 2x H200 host.
Identical data and recipe so the v8 -> 26 comparison is clean.

## 1. Data onto the H200 host
- Copy `rgb-images/` from the dataset host (`/data/datasets/ROAD_plusplus/rgb-images`, 798 video dirs of jpgs).
- Copy `road_waymo_yolo_labels.tar.gz` (labels + data.yaml, this dir) and extract to `<ROOT>/road_waymo_yolo/`.
- Rebuild image symlinks (or rerun `convert_to_yolo.py` with paths edited — it regenerates everything from the JSON in minutes):
  `python3 relink_images.py <ROOT>/road_waymo_yolo <path-to-rgb-images>`  (see below)
- Edit `data.yaml` `path:` to the new root.

## 2. Environment
- `pip install -U ultralytics` — MUST be the current release; YOLO26 configs are absent
  from 8.3.226 (verified 2026-08-20). Confirm with:
  `python3 -c "from ultralytics import YOLO; YOLO('yolo26x.pt')"`

## 3. Train (ECCV Track-1 recipe, same as v8x run)
- `python3 train_v8x.py --model yolo26x.pt --name yolo26x_agent_1280_eccv --device 0,1`
- Recipe knobs are baked into the script: SGD lr0=0.005 -> 0.0005 linear, batch 32,
  close_mosaic=5, epochs 50, imgsz 1280. Do not change any knob — the v8/26 pair
  must differ only in the model.
- Log line-buffered to a file; `tail -f` to monitor.

## 4. Ship back
- `runs/yolo26x_agent_1280_eccv/weights/best.pt` + `results.csv` back to the rig for
  the full-candidate val dump and f-mAP scoring (evaluator lives here, not on H200).

## Deviation log (A6000 rig)
- 2026-08-24: YOLO26x at batch 32 / imgsz 1280 OOMs on 48GB A6000s (needs ~47.5+GB/GPU
  at 16 per GPU; v8x fit at 44-47GB). Fallback batch 24 ALSO OOMed (TaskAlignedAssigner: dual NMS-free assignment x 33.6K anchors x mosaic-stacked dense GT). Final: batch 16 (8/GPU, accumulate x4 -> effective 64). Ultralytics auto-accumulates
  toward nominal batch 64, so effective batch stays comparable (24x3=72 vs 32x2=64),
  but per-GPU batch-norm statistics differ from the v8x run — a forced deviation, not
  a chosen one. On H200s (141GB), batch 32 fits; prefer it there for exact recipe match.
- 2026-08-26: YOLO26x run stopped at epoch 26/50 by decision (October defense
  timeline; its best checkpoint ep2 = 23.30 agent f-mAP already scored; curve
  in steady over-training decline, best.pt fitness-locked). The aug-off-phase
  behavior on 26 remains unmeasured - rerun on H200s if it ever matters.

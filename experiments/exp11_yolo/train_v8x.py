"""Exp11 leg 1 — YOLOv8x 10-class agent detector on ROAD-Waymo.

Recipe replicates the ECCV 2024 Track-1 report (wiki papers/eccv24-track1.md):
SGD, 30 epochs @ LR 0.005 + 20 @ 0.0005 (approximated as one 50-epoch linear
decay 0.005 -> 0.0005), batch 32, augmentation disabled last 5 epochs.
Documented deviations: imgsz=1280 (theirs unrecorded; 43% of boxes <8px at
640 per box-size stats 2026-08-20), SGD momentum left at Ultralytics default
(theirs unrecorded).
"""
import argparse, sys
sys.stdout.reconfigure(line_buffering=True)
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--model", default="yolov8x.pt")
p.add_argument("--imgsz", type=int, default=1280)
p.add_argument("--epochs", type=int, default=50)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--device", default="0,1")
p.add_argument("--name", default="v8x_agent_1280_eccv")
p.add_argument("--fraction", type=float, default=1.0)
a = p.parse_args()

YOLO(a.model).train(
    data="/data/datasets/road_waymo_yolo/data.yaml",
    imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, device=a.device,
    optimizer="SGD", lr0=0.005, lrf=0.1, close_mosaic=5,
    fraction=a.fraction, name=a.name,
    project="/data/repos/ROAD_Reason/experiments/exp11_yolo/runs",
    exist_ok=True)

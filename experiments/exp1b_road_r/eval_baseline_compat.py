#!/usr/bin/env python3
"""
Evaluate Exp1b with the ROAD-Waymo 3D-RetinaNet baseline frame-mAP code.

This script does two things:

1. Runs Exp1b frame-by-frame to produce per-frame detections in the same
   pickle schema expected by the baseline repo's `evaluate_frames(...)`.
2. Calls the baseline evaluator directly so the metric path matches the
   3D-RetinaNet baseline as closely as possible.

Important note:
- This is a compatibility evaluation path.
- It uses the baseline repo's frame-level IoU-based evaluator.
- It is therefore the correct path for apples-to-apples comparison against
  `analysis/baseline_val_metrics.csv`.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP1B_DIR = Path(__file__).resolve().parent
EXP1_DIR = REPO_ROOT / "experiments" / "exp1_road_r"
BASELINE_ROOT = Path("/data/repos/PedestrianIntent++/ROAD_plus_plus_Baseline")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(EXP1B_DIR))
sys.path.append(str(EXP1_DIR))

exp1_train = _load_module("exp1_train_for_compat", EXP1_DIR / "train.py")

import config as C  # noqa: E402
from model import QwenROADModel  # noqa: E402

preprocess_clip = exp1_train.preprocess_clip


sys.path.insert(0, str(BASELINE_ROOT))
from modules.evaluation import evaluate_frames  # noqa: E402
import data.datasets as baseline_datasets  # noqa: E402


HEAD_SIZES = {
    "agent": 10,
    "action": 22,
    "loc": 16,
    "duplex": 49,
    "triplet": 86,
}


def is_in_subset(split_ids, subset: str) -> bool:
    if isinstance(split_ids, str):
        split_ids = [split_ids]
    return subset in split_ids


def decode_boxes(
    ltrb: torch.Tensor,
    frame_shape: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """
    Decode FCOS-style [l, t, r, b] distances to normalized [x1, y1, x2, y2].
    """
    h_prime, w_prime = frame_shape
    rows = torch.arange(h_prime, dtype=torch.float32, device=device)
    cols = torch.arange(w_prime, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
    cy = ((grid_y + 0.5) / h_prime).reshape(-1)
    cx = ((grid_x + 0.5) / w_prime).reshape(-1)

    x1 = (cx - ltrb[:, 0]).clamp(0.0, 1.0)
    y1 = (cy - ltrb[:, 1]).clamp(0.0, 1.0)
    x2 = (cx + ltrb[:, 2]).clamp(0.0, 1.0)
    y2 = (cy + ltrb[:, 3]).clamp(0.0, 1.0)
    return torch.stack([x1, y1, x2, y2], dim=1)


def postprocess_single_frame(
    preds: dict,
    agentness_threshold: float,
    nms_iou_threshold: float,
    device: torch.device,
) -> dict:
    """
    Postprocess a single-frame prediction dict from Exp1b.
    """
    frame_shape = preds["frame_shapes"][0]
    agentness = preds["agentness"][:, 0]
    fg_mask = agentness >= agentness_threshold

    if not fg_mask.any():
        return {
            "boxes": torch.zeros(0, 4, device=device),
            "agentness": torch.zeros(0, device=device),
            "agent": torch.zeros(0, 10, device=device),
            "action": torch.zeros(0, 22, device=device),
            "loc": torch.zeros(0, 16, device=device),
            "duplex": torch.zeros(0, 49, device=device),
            "triplet": torch.zeros(0, 86, device=device),
        }

    all_boxes = decode_boxes(preds["box"], frame_shape, device)
    boxes = all_boxes[fg_mask]
    scores = agentness[fg_mask]

    try:
        from torchvision.ops import nms as torchvision_nms

        keep = torchvision_nms(boxes, scores, nms_iou_threshold)
    except ImportError:
        keep = torch.arange(boxes.shape[0], device=device)

    fg_indices = torch.where(fg_mask)[0]
    kept_global = fg_indices[keep]

    result = {
        "boxes": boxes[keep],
        "agentness": scores[keep],
    }
    for head in ("agent", "action", "loc", "duplex", "triplet"):
        result[head] = preds[head][kept_global]
    return result


def boxes_to_baseline_scale(boxes: torch.Tensor, width: int, height: int) -> np.ndarray:
    """
    Baseline evaluator expects absolute pixel coordinates.
    """
    out = boxes.detach().cpu().numpy().copy()
    out[:, 0] *= width
    out[:, 2] *= width
    out[:, 1] *= height
    out[:, 3] *= height
    return out.astype(np.float32)


def empty_class_arrays(n_classes: int) -> List[np.ndarray]:
    return [np.zeros((0, 5), dtype=np.float32) for _ in range(n_classes)]


def frame_to_detection_lists(
    dets: dict,
    width: int,
    height: int,
    class_threshold: float,
) -> dict:
    """
    Convert Exp1b detections into the baseline evaluator's per-frame format.
    """
    if dets["boxes"].shape[0] == 0:
        return {
            "agent_ness": [np.zeros((0, 5), dtype=np.float32)],
            "agent": empty_class_arrays(10),
            "action": empty_class_arrays(22),
            "loc": empty_class_arrays(16),
            "duplex": empty_class_arrays(49),
            "triplet": empty_class_arrays(86),
        }

    boxes = boxes_to_baseline_scale(dets["boxes"], width, height)
    n_det = boxes.shape[0]

    frame_det = {}
    agentness = dets["agentness"].detach().cpu().numpy().reshape(n_det, 1)
    frame_det["agent_ness"] = [np.hstack([boxes, agentness]).astype(np.float32)]

    for head, n_classes in HEAD_SIZES.items():
        scores = dets[head].detach().cpu().numpy()
        per_class = []
        for cid in range(n_classes):
            cls_scores = scores[:, cid]
            keep = cls_scores > class_threshold
            if np.any(keep):
                per_class.append(
                    np.hstack([boxes[keep], cls_scores[keep, None]]).astype(np.float32)
                )
            else:
                per_class.append(np.zeros((0, 5), dtype=np.float32))
        frame_det[head] = per_class

    return frame_det


def build_detection_pickle(
    model,
    processor,
    anno_file: Path,
    frames_dir: Path,
    subset: str,
    device: torch.device,
    dtype: torch.dtype,
    agentness_threshold: float,
    nms_iou_threshold: float,
    class_threshold: float,
) -> dict:
    with open(anno_file) as f:
        anno_data = json.load(f)

    detections = {
        "agent_ness": {},
        "agent": {},
        "action": {},
        "loc": {},
        "duplex": {},
        "triplet": {},
    }

    annotated_frames = 0
    used_frames = 0
    total_kept = 0
    t0 = time.time()

    model.eval()
    with torch.no_grad():
        for videoname, vdata in anno_data["db"].items():
            if not is_in_subset(vdata.get("split_ids", []), subset):
                continue

            for frame_id, frame in vdata.get("frames", {}).items():
                if frame.get("annotated", 0) != 1:
                    continue
                annotated_frames += 1

                frame_name = f"{int(frame_id):05d}"
                frame_key = videoname + frame_name
                img_path = frames_dir / videoname / f"{frame_name}.jpg"

                if not img_path.exists():
                    continue

                image = Image.open(img_path).convert("RGB")
                width, height = image.size
                pixel_values, image_grid_thw = preprocess_clip([image], processor, device, dtype)
                preds = model(pixel_values, image_grid_thw)
                dets = postprocess_single_frame(
                    preds,
                    agentness_threshold=agentness_threshold,
                    nms_iou_threshold=nms_iou_threshold,
                    device=device,
                )

                total_kept += int(dets["boxes"].shape[0])
                frame_det = frame_to_detection_lists(
                    dets,
                    width=width,
                    height=height,
                    class_threshold=class_threshold,
                )

                for label_type, value in frame_det.items():
                    detections[label_type][frame_key] = value

                used_frames += 1
                if used_frames % 500 == 0:
                    elapsed = time.time() - t0
                    print(
                        f"  processed {used_frames}/{annotated_frames} annotated frames "
                        f"| avg kept/frame={total_kept / max(used_frames, 1):.2f} "
                        f"| {elapsed:.0f}s"
                    )

    print(
        f"Finished export: {used_frames} frames | "
        f"avg kept/frame={total_kept / max(used_frames, 1):.2f}"
    )
    return detections


def summarize_results(results: dict) -> dict:
    summary = {}
    for label_type in ("agent_ness", "agent", "action", "loc", "duplex", "triplet"):
        if label_type not in results:
            continue
        entry = results[label_type]
        summary[label_type] = {
            "mAP": round(float(entry["mAP"]), 6),
            "mR": round(float(entry["mR"]), 6),
            "ap_all": [round(float(x), 6) for x in entry["ap_all"]],
        }
    return summary


RESULTS_CSV = Path(__file__).resolve().parents[2] / "results" / "val_metrics.csv"
CSV_FIELDS = [
    "model", "source", "status", "epoch", "metric", "split", "iou",
    "agent_ness", "agent", "action", "loc", "duplex", "triplet",
]


def write_to_csv(
    csv_path: Path,
    model_name: str,
    source: str,
    status: str,
    epoch: int,
    metric: str,
    split: str,
    iou: float | str,
    summary: dict,
) -> None:
    """
    Append (or update) one row in the shared results CSV.

    Uniqueness key: (model, epoch, metric, split). If a row with the same key
    already exists it is replaced in-place; otherwise the row is appended.
    """
    new_row = {
        "model":  model_name,
        "source": source,
        "status": status,
        "epoch":  epoch,
        "metric": metric,
        "split":  split,
        "iou":    iou,
        "agent_ness": summary.get("agent_ness", {}).get("mAP", ""),
        "agent":      summary.get("agent",      {}).get("mAP", ""),
        "action":     summary.get("action",     {}).get("mAP", ""),
        "loc":        summary.get("loc",        {}).get("mAP", ""),
        "duplex":     summary.get("duplex",     {}).get("mAP", ""),
        "triplet":    summary.get("triplet",    {}).get("mAP", ""),
    }

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    key = lambda r: (r["model"], str(r["epoch"]), r["metric"], r["split"])
    new_key = key(new_row)

    if csv_path.exists():
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

    replaced = False
    for i, r in enumerate(rows):
        if key(r) == new_key:
            rows[i] = new_row
            replaced = True
            break
    if not replaced:
        rows.append(new_row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    action = "Updated" if replaced else "Appended"
    print(f"{action} results in {csv_path}")


def infer_frame_size(anno_file: Path, frames_dir: Path, subset: str) -> tuple[int, int]:
    with open(anno_file) as f:
        anno_data = json.load(f)

    for videoname, vdata in anno_data["db"].items():
        if not is_in_subset(vdata.get("split_ids", []), subset):
            continue
        for frame_id, frame in vdata.get("frames", {}).items():
            if frame.get("annotated", 0) != 1:
                continue
            frame_name = f"{int(frame_id):05d}"
            img_path = frames_dir / videoname / f"{frame_name}.jpg"
            if img_path.exists():
                with Image.open(img_path) as img:
                    width, height = img.size
                return width, height

    raise FileNotFoundError(f"Could not infer frame size from subset={subset}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",              default=C.MODEL_ID)
    parser.add_argument("--ckpt",               default=str(Path(C.CKPT_DIR) / "best.pt"))
    parser.add_argument("--anno",               default=C.ANNO_FILE)
    parser.add_argument("--frames",             default=C.FRAMES_DIR)
    parser.add_argument("--subset",             default="val")
    parser.add_argument("--det-pkl",            default=str(Path(C.LOG_DIR) / "baseline_compat_dets.pkl"))
    parser.add_argument("--out",                default=str(Path(C.LOG_DIR) / "baseline_compat_results.json"))
    parser.add_argument("--agentness-threshold",type=float, default=C.AGENTNESS_THRESHOLD)
    parser.add_argument("--class-threshold",    type=float, default=0.025)
    parser.add_argument("--nms-iou-threshold",  type=float, default=C.NMS_IOU_THRESHOLD)
    parser.add_argument("--eval-iou",           type=float, default=0.5)
    parser.add_argument("--csv",                default=str(RESULTS_CSV),
                        help="Path to shared results CSV (default: results/val_metrics.csv)")
    parser.add_argument("--model-name",         default=None,
                        help="Row label for the CSV (default: Exp1b-QwenViT-FCOS-Godel)")
    parser.add_argument("--source",             default="novel",
                        help="Source column value in CSV (default: novel)")
    parser.add_argument("--status",             default="done",
                        help="Status column value in CSV (default: done)")
    parser.add_argument("--no-csv",             action="store_true",
                        help="Skip writing to the results CSV")
    parser.add_argument("--sync-sharepoint",    action="store_true",
                        help="Push results/val_metrics.csv to OneDrive after writing CSV")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"Device: {device} | dtype: {dtype}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Subset: {args.subset}")

    processor = AutoProcessor.from_pretrained(args.model)

    model = QwenROADModel(
        model_id=args.model,
        d_model=C.VIT_DIM,
        freeze_vit=True,
    )
    model.add_lora(
        r=C.LORA_R,
        lora_alpha=C.LORA_ALPHA,
        lora_dropout=C.LORA_DROPOUT,
        target_modules=C.LORA_TARGET_MODULES,
        n_layers=C.LORA_N_LAYERS,
    )
    model = model.to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded epoch {ckpt.get('epoch', '?')} (best action mAP={ckpt.get('best_map', '?')})")

    detections = build_detection_pickle(
        model=model,
        processor=processor,
        anno_file=Path(args.anno),
        frames_dir=Path(args.frames),
        subset=args.subset,
        device=device,
        dtype=dtype,
        agentness_threshold=args.agentness_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        class_threshold=args.class_threshold,
    )

    det_pkl = Path(args.det_pkl)
    det_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(det_pkl, "wb") as f:
        pickle.dump(detections, f)
    print(f"Saved baseline-compatible detections to {det_pkl}")

    width, height = infer_frame_size(Path(args.anno), Path(args.frames), args.subset)
    baseline_datasets.g_w = height
    baseline_datasets.g_h = width

    results = evaluate_frames(
        args.anno,
        str(det_pkl),
        args.subset,
        wh=[height, width],
        iou_thresh=args.eval_iou,
        dataset="road_waymo",
    )

    summary = summarize_results(results)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "checkpoint": args.ckpt,
                "subset": args.subset,
                "eval_iou": args.eval_iou,
                "agentness_threshold": args.agentness_threshold,
                "class_threshold": args.class_threshold,
                "nms_iou_threshold": args.nms_iou_threshold,
                "summary": summary,
            },
            f,
            indent=2,
        )

    print("\nBaseline-compatible frame-mAP summary")
    for key in ("agent_ness", "agent", "action", "loc", "duplex", "triplet"):
        if key in summary:
            print(f"  {key:10s} mAP={summary[key]['mAP']:.4f}  mR={summary[key]['mR']:.4f}")
    print(f"\nSaved results to {out_path}")

    if not args.no_csv:
        import torch as _torch
        ckpt_data = _torch.load(args.ckpt, map_location="cpu", weights_only=True) \
            if Path(args.ckpt).exists() else {}
        epoch = ckpt_data.get("epoch", "?")
        model_name = args.model_name or "Exp1b-QwenViT-FCOS-Godel"
        write_to_csv(
            csv_path=Path(args.csv),
            model_name=model_name,
            source=args.source,
            status=args.status,
            epoch=epoch,
            metric="f-mAP",
            split=args.subset,
            iou=args.eval_iou,
            summary=summary,
        )

    if args.sync_sharepoint:
        import subprocess, sys as _sys
        sync_script = Path(__file__).resolve().parents[2] / "results" / "sync_to_sharepoint.py"
        subprocess.run([_sys.executable, str(sync_script)], check=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]
EXP1_DIR = REPO_ROOT / "experiments" / "exp1_road_r"
BASELINE_ROOT = Path("/data/repos/PedestrianIntent++/ROAD_plus_plus_Baseline")

if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXP1_DIR) not in sys.path:
    sys.path.insert(0, str(EXP1_DIR))

import config as C
from losses import greedy_group_tubes
from matcher import box_iou
from model import DETRROADModel


def _load_module(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def preprocess_clip(pil_frames, processor, device, dtype):
    inputs = processor.image_processor(
        images=pil_frames,
        return_tensors="pt",
        min_pixels=C.MIN_PIXELS,
        max_pixels=C.MAX_PIXELS,
    )
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    image_grid_thw = inputs["image_grid_thw"].to(device=device)
    return pixel_values, image_grid_thw

sys.path.insert(0, str(BASELINE_ROOT))
from modules.evaluation import evaluate_frames  # noqa: E402
import data.datasets as baseline_datasets  # noqa: E402


def is_in_subset(split_ids, subset: str) -> bool:
    if isinstance(split_ids, str):
        split_ids = [split_ids]
    return subset in split_ids


def to_xyxy(boxes_cxcywh: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes_cxcywh.unbind(-1)
    x1 = (cx - 0.5 * w).clamp(0.0, 1.0)
    y1 = (cy - 0.5 * h).clamp(0.0, 1.0)
    x2 = (cx + 0.5 * w).clamp(0.0, 1.0)
    y2 = (cy + 0.5 * h).clamp(0.0, 1.0)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def frame_to_detection_lists(
    boxes: torch.Tensor,
    probs: dict,
    width: int,
    height: int,
    class_threshold: float,
):
    def empty(n: int):
        return [np.zeros((0, 5), dtype=np.float32) for _ in range(n)]

    if boxes.shape[0] == 0:
        return {
            "agent_ness": [np.zeros((0, 5), dtype=np.float32)],
            "agent": empty(C.N_AGENTS),
            "action": empty(C.N_ACTIONS),
            "loc": empty(C.N_LOCS),
            "duplex": empty(C.N_DUPLEXES),
            "triplet": empty(C.N_TRIPLETS),
        }

    boxes_np = boxes.detach().cpu().numpy().copy()
    boxes_np[:, 0] *= width
    boxes_np[:, 2] *= width
    boxes_np[:, 1] *= height
    boxes_np[:, 3] *= height

    conf = probs["agent"].max(dim=1).values.detach().cpu().numpy()[:, None]
    out = {"agent_ness": [np.hstack([boxes_np, conf]).astype(np.float32)]}

    for head, n_classes in C.HEAD_SIZES.items():
        scores = (probs["agent"].max(dim=1).values.unsqueeze(1) * probs[head]).detach().cpu().numpy()
        per_class = []
        for cid in range(n_classes):
            cls_scores = scores[:, cid]
            keep = cls_scores > class_threshold
            if keep.any():
                per_class.append(np.hstack([boxes_np[keep], cls_scores[keep, None]]).astype(np.float32))
            else:
                per_class.append(np.zeros((0, 5), dtype=np.float32))
        out[head] = per_class
    return out


def summarize_results(results: dict) -> dict:
    summary = {}
    for label_type in ("agent_ness", "agent", "action", "loc", "duplex", "triplet"):
        if label_type in results:
            summary[label_type] = {
                "mAP": round(float(results[label_type]["mAP"]), 6),
                "mR": round(float(results[label_type]["mR"]), 6),
            }
    return summary


def tube_iou(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    overlap = pred_mask & gt_mask
    if not overlap.any():
        return 0.0
    ious = []
    for t in torch.where(overlap)[0]:
        ious.append(float(box_iou(pred_boxes[t : t + 1], gt_boxes[t : t + 1]).item()))
    return float(sum(ious) / max(len(ious), 1))


def approximate_video_eval(model, processor, anno_file: Path, frames_dir: Path, subset: str, device, dtype, threshold: float):
    with open(anno_file) as f:
        data = json.load(f)

    ap_like_scores = []
    for videoname, vdata in data["db"].items():
        if not is_in_subset(vdata.get("split_ids", []), subset):
            continue
        frame_ids = sorted(
            [fid for fid, fd in vdata.get("frames", {}).items() if fd.get("annotated", 0) == 1],
            key=lambda x: int(x),
        )
        if len(frame_ids) < C.CLIP_LEN:
            continue
        clip_ids = frame_ids[: C.CLIP_LEN]
        pil_frames = []
        frame_targets = []
        for fid in clip_ids:
            img_path = frames_dir / videoname / f"{int(fid):05d}.jpg"
            pil_frames.append(Image.open(img_path).convert("RGB"))
            annos = vdata["frames"][fid].get("annos", {})
            parsed = []
            for anno in annos.values():
                if isinstance(anno, dict) and anno.get("box") is not None:
                    parsed.append(anno)
            frame_targets.append(parsed if parsed else None)

        pixel_values, image_grid_thw = preprocess_clip(pil_frames, processor, device, dtype)
        outputs = model(pixel_values, image_grid_thw)
        probs = {k: v.sigmoid() for k, v in outputs["pred_logits"].items()}
        keep = probs["agent"].max(dim=1).values > threshold
        if not keep.any():
            continue

        pred_boxes = to_xyxy(outputs["pred_boxes"][keep])
        pred_mask = torch.ones(pred_boxes.shape[:2], dtype=torch.bool, device=pred_boxes.device)

        parsed_targets = []
        for annos in frame_targets:
            if annos is None:
                parsed_targets.append(None)
                continue
            boxes = torch.tensor([anno["box"] for anno in annos], dtype=torch.float32, device=device)
            agent = torch.zeros(len(annos), C.N_AGENTS, device=device)
            action = torch.zeros(len(annos), C.N_ACTIONS, device=device)
            loc = torch.zeros(len(annos), C.N_LOCS, device=device)
            duplex = torch.zeros(len(annos), C.N_DUPLEXES, device=device)
            triplet = torch.zeros(len(annos), C.N_TRIPLETS, device=device)
            parsed_targets.append({"boxes": boxes, "agent": agent, "action": action, "loc": loc, "duplex": duplex, "triplet": triplet})
        gt_tubes = greedy_group_tubes(parsed_targets, iou_thresh=C.TUBE_LINK_IOU)
        for tube in gt_tubes:
            best = 0.0
            for q in range(pred_boxes.shape[0]):
                best = max(best, tube_iou(pred_boxes[q], tube["boxes"], pred_mask[q], tube["box_mask"]))
            ap_like_scores.append(best)

    return {
        "mean_best_tube_iou": round(float(sum(ap_like_scores) / max(len(ap_like_scores), 1)), 6),
        "n_tubes": len(ap_like_scores),
        "note": "Approximate tube metric for early experiment iteration; replace with official video AP when available.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=C.MODEL_ID)
    parser.add_argument("--ckpt", default=str(Path(C.CKPT_DIR) / "best.pt"))
    parser.add_argument("--anno", default=C.ANNO_FILE)
    parser.add_argument("--frames", default=C.FRAMES_DIR)
    parser.add_argument("--subset", default="val")
    parser.add_argument("--mode", choices=["frame", "video"], default="frame")
    parser.add_argument("--out", default=None)
    parser.add_argument("--det-pkl", default=str(Path(C.LOG_DIR) / "baseline_compat_dets.pkl"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(args.model)

    model = DETRROADModel(
        model_id=args.model,
        vit_dim=C.VIT_DIM,
        d_model=C.D_MODEL,
        head_sizes=C.HEAD_SIZES,
        clip_len=C.CLIP_LEN,
        num_queries=C.NUM_QUERIES,
        num_decoder_layers=C.NUM_DECODER_LAYERS,
        nhead=C.NHEAD,
        dim_ffn=C.DIM_FFN,
        dropout=C.DROPOUT,
        freeze_vit=True,
    )
    model.add_lora(
        r=C.LORA_R,
        lora_alpha=C.LORA_ALPHA,
        lora_dropout=C.LORA_DROPOUT,
        target_modules=C.LORA_TARGET_MODULES,
        n_layers=C.LORA_N_LAYERS,
    )
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(device)
    model.eval()

    if args.mode == "video":
        summary = approximate_video_eval(
            model, processor, Path(args.anno), Path(args.frames), args.subset, device, dtype, C.CONFIDENCE_THRESHOLD
        )
        out_path = Path(args.out or (Path(C.LOG_DIR) / "eval_vmap.json"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))
        return

    with open(args.anno) as f:
        anno_data = json.load(f)

    detections = {k: {} for k in ["agent_ness", "agent", "action", "loc", "duplex", "triplet"]}
    frame_size = None
    with torch.no_grad():
        for videoname, vdata in anno_data["db"].items():
            if not is_in_subset(vdata.get("split_ids", []), args.subset):
                continue
            for frame_id, frame in vdata.get("frames", {}).items():
                if frame.get("annotated", 0) != 1:
                    continue
                fname = f"{int(frame_id):05d}"
                img_path = Path(args.frames) / videoname / f"{fname}.jpg"
                if not img_path.exists():
                    continue
                image = Image.open(img_path).convert("RGB")
                width, height = image.size
                frame_size = (height, width)
                pixel_values, image_grid_thw = preprocess_clip([image], processor, device, dtype)
                outputs = model(pixel_values, image_grid_thw)
                probs = {k: v.sigmoid() for k, v in outputs["pred_logits"].items()}
                conf = probs["agent"].max(dim=1).values
                keep = conf > C.CONFIDENCE_THRESHOLD
                boxes = to_xyxy(outputs["pred_boxes"][keep, 0])
                probs_kept = {k: v[keep] for k, v in probs.items()}
                det = frame_to_detection_lists(boxes, probs_kept, width, height, C.CLASS_THRESHOLD)
                frame_key = videoname + fname
                for head, value in det.items():
                    detections[head][frame_key] = value

    det_pkl = Path(args.det_pkl)
    det_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(det_pkl, "wb") as f:
        pickle.dump(detections, f)

    assert frame_size is not None, "No annotated frames found for baseline-compatible eval"
    height, width = frame_size
    baseline_datasets.g_w = height
    baseline_datasets.g_h = width
    results = evaluate_frames(
        args.anno,
        str(det_pkl),
        args.subset,
        wh=[height, width],
        iou_thresh=0.5,
        dataset="road_waymo",
    )
    summary = summarize_results(results)
    out_path = Path(args.out or (Path(C.LOG_DIR) / "eval_fmap.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

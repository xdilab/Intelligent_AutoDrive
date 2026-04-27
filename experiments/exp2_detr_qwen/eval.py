#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]
EXP1_DIR = REPO_ROOT / "experiments" / "exp1_road_r"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))
if str(EXP1_DIR) not in sys.path:
    sys.path.append(str(EXP1_DIR))

import config as C
from losses import SetCriterion, compute_class_alphas, load_constraint_children
from matcher import HungarianMatcher
from model import DETRROADModel


def _load_module(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

exp1_dataset = _load_module("exp1_dataset_for_exp2_eval", EXP1_DIR / "dataset.py")

ROADWaymoDataset = exp1_dataset.ROADWaymoDataset


def collate_fn(batch):
    assert len(batch) == 1
    return batch[0]


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


def average_precision(scores: torch.Tensor, targets: torch.Tensor) -> float:
    scores = scores.detach().cpu()
    targets = targets.detach().cpu().bool()
    n_pos = int(targets.sum())
    if n_pos == 0:
        return float("nan")

    order = torch.argsort(scores, descending=True)
    targets = targets[order]
    tp = targets.float().cumsum(0)
    fp = (~targets).float().cumsum(0)
    precision = tp / (tp + fp).clamp(min=1e-6)
    return float((precision[targets].sum() / n_pos).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=C.MODEL_ID)
    parser.add_argument("--ckpt", default=str(Path(C.CKPT_DIR) / "best.pt"))
    parser.add_argument("--anno", default=C.ANNO_FILE)
    parser.add_argument("--frames", default=C.FRAMES_DIR)
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    ds = ROADWaymoDataset(args.anno, args.frames, split=args.split, clip_len=C.CLIP_LEN, stride=C.CLIP_STRIDE)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
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

    matcher = HungarianMatcher(C.COST_CLASS, C.COST_BBOX, C.COST_GIOU)
    class_alphas = compute_class_alphas(args.anno)
    constraint_data = load_constraint_children(args.anno)
    criterion = SetCriterion(matcher, class_alphas=class_alphas, **constraint_data).to(device)

    action_scores = []
    action_targets = []

    with torch.no_grad():
        for pil_frames, frame_targets in loader:
            pixel_values, image_grid_thw = preprocess_clip(pil_frames, processor, device, dtype)
            outputs = model(pixel_values, image_grid_thw)
            from losses import greedy_group_tubes
            gt_tubes = greedy_group_tubes(frame_targets, iou_thresh=C.TUBE_LINK_IOU)
            matched_pred, matched_gt = matcher(outputs["pred_boxes"], outputs["pred_logits"], gt_tubes)
            if len(matched_pred) == 0:
                continue
            probs = outputs["pred_logits"]["action"][matched_pred].sigmoid()
            gts = torch.stack([gt_tubes[int(j)]["labels"]["action"] for j in matched_gt], dim=0).to(probs.device)
            action_scores.append(probs)
            action_targets.append(gts)

    if not action_scores:
        print("No matched queries found.")
        return

    action_scores = torch.cat(action_scores, dim=0)
    action_targets = torch.cat(action_targets, dim=0)
    aps = []
    for c in range(action_scores.shape[1]):
        ap = average_precision(action_scores[:, c], action_targets[:, c])
        if ap == ap:
            aps.append(ap)

    summary = {
        "matched_action_map": round(sum(aps) / max(len(aps), 1), 6),
        "n_samples": int(action_scores.shape[0]),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Exp2b training: EfficientNet-FPN + Deformable DETR with VLM semantic fusion."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import torch
from torch.optim import AdamW
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
from losses import SetCriterion, compute_class_alphas, greedy_group_tubes, load_constraint_children
from matcher import HungarianMatcher
from model import EfficientNetFPNDETRModel


def _load_module(name: str, path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


exp1_dataset = _load_module("exp1_dataset_for_exp2b", EXP1_DIR / "dataset.py")
ROADWaymoDataset = exp1_dataset.ROADWaymoDataset


def collate_fn(batch):
    assert len(batch) == 1
    return batch[0]


def preprocess_clip(pil_frames, processor, device, dtype):
    """Convert PIL frames to Qwen ViT tensor format."""
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


def load_exp2_warmstart(model: EfficientNetFPNDETRModel, ckpt_path: str, device: torch.device) -> None:
    """Transfer compatible weights from exp2 checkpoint.

    Transfers: Qwen ViT + LoRA, FeatureProjection, classification heads,
    agentness head, box head MLP, temporal embeddings.
    Does NOT transfer: DETR decoder layers (incompatible with deformable attn).
    """
    ckpt_file = Path(ckpt_path)
    if not ckpt_file.exists():
        print(f"Warm-start skipped: {ckpt_file} not found")
        return

    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)

    # Build mapping from exp2 keys → exp2b keys
    key_map = {}
    for k in state:
        new_k = None
        if k.startswith("vit."):
            # vit.visual.* → vlm_fusion.vit.visual.*
            new_k = "vlm_fusion." + k
        elif k.startswith("projection."):
            # projection.* → vlm_fusion.feat_proj.*
            new_k = k.replace("projection.", "vlm_fusion.feat_proj.")
        elif k.startswith("cls_heads."):
            new_k = k  # same path
        elif k == "agentness_head.weight" or k == "agentness_head.bias":
            new_k = k  # same path
        # decoder.box_head.* and decoder.temporal_embed.* from exp2 are
        # incompatible: exp2b uses per-layer box_heads (ModuleList) and
        # temporal attention inside decoder layers. Skip these.

        if new_k is not None:
            key_map[k] = new_k

    # Filter to keys that exist in new model with matching shapes
    new_state = model.state_dict()
    transfer = {}
    skipped = []
    for old_k, new_k in key_map.items():
        if new_k in new_state and state[old_k].shape == new_state[new_k].shape:
            transfer[new_k] = state[old_k]
        else:
            skipped.append((old_k, new_k))

    missing, unexpected = model.load_state_dict(transfer, strict=False)
    print(
        f"Warm-started from exp2 {ckpt_file.name} | "
        f"transferred={len(transfer)} keys | "
        f"skipped={len(skipped)} (shape mismatch or missing)"
    )


def build_optimizer(model: EfficientNetFPNDETRModel):
    """Four-group AdamW: LoRA, backbone, decoder, heads."""
    param_groups = [
        {"params": model.lora_parameters(),     "lr": C.LR_LORA},
        {"params": model.backbone_parameters(),  "lr": C.LR_BACKBONE},
        {"params": model.decoder_parameters(),   "lr": C.LR_DECODER},
        {"params": model.head_parameters(),      "lr": C.LR_HEADS},
    ]
    return AdamW(param_groups, weight_decay=C.WEIGHT_DECAY)


_LR_GROUPS = [C.LR_LORA, C.LR_BACKBONE, C.LR_DECODER, C.LR_HEADS]


def set_warmup_lr(optimizer, step: int, warmup_steps: int):
    if step >= warmup_steps:
        return
    frac = float(step + 1) / max(warmup_steps, 1)
    for group, base_lr in zip(optimizer.param_groups, _LR_GROUPS):
        group["lr"] = base_lr * frac


def set_cosine_lr(optimizer, epoch: int, total_epochs: int):
    cos = 0.5 * (1.0 + math.cos(math.pi * epoch / max(total_epochs, 1)))
    min_scale = 0.1
    for group, base_lr in zip(optimizer.param_groups, _LR_GROUPS):
        group["lr"] = base_lr * (min_scale + (1.0 - min_scale) * cos)


def validate(model, loader, processor, criterion, matcher, device, dtype):
    model.eval()
    totals = {
        "L_total": 0.0, "L_cls": 0.0, "L_bbox": 0.0,
        "L_giou": 0.0, "L_tnorm": 0.0, "L_agentness": 0.0, "L_aux": 0.0,
    }
    n = 0
    action_scores = []
    action_targets = []

    with torch.no_grad():
        for pil_frames, frame_targets in loader:
            pixel_values, image_grid_thw = preprocess_clip(pil_frames, processor, device, dtype)
            outputs = model(pil_frames, pixel_values, image_grid_thw)
            loss, log = criterion(outputs, frame_targets)
            for k in totals:
                totals[k] += log[k]
            n += 1

            gt_tubes = greedy_group_tubes(frame_targets, iou_thresh=C.TUBE_LINK_IOU)
            matched_pred, matched_gt = matcher(
                outputs["pred_boxes"], outputs["pred_logits"], gt_tubes
            )
            if len(matched_pred) == 0:
                continue
            probs = outputs["pred_logits"]["action"][matched_pred].sigmoid()
            gts = torch.stack(
                [gt_tubes[int(j)]["labels"]["action"] for j in matched_gt], dim=0
            ).to(probs.device)
            action_scores.append(probs)
            action_targets.append(gts)

    if n == 0:
        return {"L_total": float("nan"), "matched_action_map": float("nan")}

    metrics = {k: v / n for k, v in totals.items()}
    if action_scores:
        scores = torch.cat(action_scores, dim=0)
        targets = torch.cat(action_targets, dim=0)
        aps = []
        for c in range(scores.shape[1]):
            ap = average_precision(scores[:, c], targets[:, c])
            if ap == ap:
                aps.append(ap)
        metrics["matched_action_map"] = sum(aps) / max(len(aps), 1)
    else:
        metrics["matched_action_map"] = 0.0
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=C.MODEL_ID)
    parser.add_argument("--anno", default=C.ANNO_FILE)
    parser.add_argument("--frames", default=C.FRAMES_DIR)
    parser.add_argument("--epochs", type=int, default=C.MAX_EPOCHS)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--warmstart", default=C.EXP2_CKPT,
                        help="Exp2 checkpoint for warm-starting compatible weights")
    parser.add_argument("--ckpt-dir", default=C.CKPT_DIR)
    parser.add_argument("--log-dir", default=C.LOG_DIR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    print(f"Device: {device} | dtype: {dtype}")

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_dir) / "metrics.jsonl"

    train_ds = ROADWaymoDataset(
        args.anno, args.frames, split="train", clip_len=C.CLIP_LEN, stride=C.CLIP_STRIDE
    )
    val_ds = ROADWaymoDataset(
        args.anno, args.frames, split="val", clip_len=C.CLIP_LEN, stride=C.CLIP_STRIDE
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    print(f"train clips: {len(train_ds):,} | val clips: {len(val_ds):,}")

    processor = AutoProcessor.from_pretrained(args.model)

    print("Building model...")
    model = EfficientNetFPNDETRModel(
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
        n_deform_points=C.N_DEFORM_POINTS,
        backbone_name=C.BACKBONE,
        backbone_freeze_blocks=C.BACKBONE_FREEZE_BLOCKS,
        fpn_in_channels=C.FPN_IN_CHANNELS,
        gate_init=C.VLM_GATE_INIT,
        lora_r=C.LORA_R,
        lora_alpha=C.LORA_ALPHA,
        lora_dropout=C.LORA_DROPOUT,
        lora_target_modules=C.LORA_TARGET_MODULES,
        lora_n_layers=C.LORA_N_LAYERS,
    )
    model = model.to(device)

    # Warm-start from exp2 checkpoint (ViT+LoRA, projection, cls heads, box head)
    if args.warmstart and not args.resume:
        load_exp2_warmstart(model, args.warmstart, device)

    # Print param counts
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {n_total:,} | Trainable: {n_train:,}")

    class_alphas = compute_class_alphas(args.anno)
    matcher = HungarianMatcher(C.COST_CLASS, C.COST_BBOX, C.COST_GIOU)
    constraint_data = load_constraint_children(args.anno)
    criterion = SetCriterion(
        matcher, class_alphas=class_alphas, **constraint_data
    ).to(device)
    optimizer = build_optimizer(model)

    start_epoch = 1
    best_map = -1.0
    global_step = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_map = ckpt.get("best_map", -1.0)
        global_step = ckpt.get("global_step", 0)
        print(f"Resumed from epoch {ckpt['epoch']}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = {k: 0.0 for k in ["L_total", "L_cls", "L_bbox", "L_giou", "L_tnorm", "L_agentness", "L_aux"]}
        n_batches = 0
        t0 = time.time()
        print(f"Starting epoch {epoch}/{args.epochs} ...")

        for step, (pil_frames, frame_targets) in enumerate(train_loader, start=1):
            pixel_values, image_grid_thw = preprocess_clip(pil_frames, processor, device, dtype)
            outputs = model(pil_frames, pixel_values, image_grid_thw)
            loss, log = criterion(outputs, frame_targets)

            (loss / C.GRAD_ACCUM).backward()

            if step % C.GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], C.GRAD_CLIP
                )
                set_warmup_lr(optimizer, global_step, C.WARMUP_STEPS)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            for k in running:
                running[k] += log[k]
            n_batches += 1

            if step % 25 == 0:
                avg = {k: running[k] / max(n_batches, 1) for k in running}
                print(
                    f"  [train] ep{epoch}/{args.epochs} clip {step}/{len(train_loader)} | "
                    f"L={avg['L_total']:.4f} cls={avg['L_cls']:.4f} box={avg['L_bbox']:.4f} "
                    f"giou={avg['L_giou']:.4f} tnorm={avg['L_tnorm']:.4f} aux={avg['L_aux']:.4f}"
                )

        set_cosine_lr(optimizer, epoch, args.epochs)

        train_metrics = {k: v / max(n_batches, 1) for k, v in running.items()}
        val_metrics = validate(model, val_loader, processor, criterion, matcher, device, dtype)
        elapsed = time.time() - t0

        payload = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "elapsed_s": round(elapsed, 1),
            "global_step": global_step,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(payload) + "\n")

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train L={train_metrics['L_total']:.4f} | "
            f"val L={val_metrics['L_total']:.4f} | "
            f"val matched action mAP={val_metrics['matched_action_map']:.4f} | "
            f"{elapsed:.0f}s"
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_map": best_map,
            "global_step": global_step,
        }
        torch.save(ckpt, Path(args.ckpt_dir) / "latest.pt")

        if val_metrics["matched_action_map"] > best_map:
            best_map = val_metrics["matched_action_map"]
            ckpt["best_map"] = best_map
            torch.save(ckpt, Path(args.ckpt_dir) / "best.pt")
            print(f"  New best matched action mAP: {best_map:.4f}")


if __name__ == "__main__":
    main()

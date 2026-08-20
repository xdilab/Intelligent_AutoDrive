#!/usr/bin/env python3
"""Exp2f training: R50-FPN + Frozen-DETR encoder + CLIP ViT-L/14 + flat 184-dim head.

Same architecture as exp2e except the classification head:
  - exp2e: 6 separate heads (agentness:1, agent:10, action:22, loc:16, duplex:49, triplet:86)
  - exp2f: 1 flat nn.Linear(256, 184) with sigmoid + focal loss on ALL queries

Key fix: unmatched queries now receive explicit target=0 supervision on all 184 dims,
not just agentness. This should fix the score-localization decorrelation found in exp2e.
"""
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
from augmentations import ClipAugmentation
from losses import SetCriterion, compute_flat_alphas, greedy_group_tubes, load_constraint_children
from matcher import HungarianMatcher
from model import FrozenDETRModel


def _load_module(name: str, path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


exp1_dataset = _load_module("exp1_dataset_for_exp2f", EXP1_DIR / "dataset.py")
ROADWaymoDataset = exp1_dataset.ROADWaymoDataset


def collate_fn(batch):
    assert len(batch) == 1
    return batch[0]


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


def load_dino_coco_weights(model: FrozenDETRModel, ckpt_path: str, device: torch.device) -> None:
    """Transfer DINO COCO pretrained weights to R50 + encoder + decoder.

    Same as exp2e — transfers backbone, encoder, decoder, box heads.
    Skips class heads (91 COCO → our flat 184-dim head).
    """
    ckpt_file = Path(ckpt_path)
    if not ckpt_file.exists():
        print(f"DINO COCO weights skipped: {ckpt_file} not found")
        return

    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    dino_state = ckpt.get("model", ckpt)
    our_state = model.state_dict()

    transfer = {}

    # --- R50 Backbone ---
    for dk, dv in dino_state.items():
        if dk.startswith("backbone.0.body."):
            ok = dk.replace("backbone.0.body.", "backbone.body.")
            if ok in our_state and dv.shape == our_state[ok].shape:
                transfer[ok] = dv

    # --- Encoder ---
    for i in range(6):
        dino_prefix = f"transformer.encoder.layers.{i}"
        our_prefix = f"encoder.layers.{i}"
        for suffix in [
            "self_attn.sampling_offsets.weight", "self_attn.sampling_offsets.bias",
            "self_attn.attention_weights.weight", "self_attn.attention_weights.bias",
            "self_attn.value_proj.weight", "self_attn.value_proj.bias",
            "self_attn.output_proj.weight", "self_attn.output_proj.bias",
        ]:
            dk, ok = f"{dino_prefix}.{suffix}", f"{our_prefix}.{suffix}"
            if dk in dino_state and ok in our_state:
                if dino_state[dk].shape == our_state[ok].shape:
                    transfer[ok] = dino_state[dk]
        for suffix in ["linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias"]:
            dk, ok = f"{dino_prefix}.{suffix}", f"{our_prefix}.{suffix}"
            if dk in dino_state and ok in our_state:
                if dino_state[dk].shape == our_state[ok].shape:
                    transfer[ok] = dino_state[dk]
        for suffix in ["norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"]:
            dk, ok = f"{dino_prefix}.{suffix}", f"{our_prefix}.{suffix}"
            if dk in dino_state and ok in our_state:
                transfer[ok] = dino_state[dk]

    # --- Decoder layers (partial) ---
    for i in range(6):
        dino_prefix = f"transformer.decoder.layers.{i}"
        our_prefix = f"decoder.layers.{i}"
        for suffix in [
            "self_attn.in_proj_weight", "self_attn.in_proj_bias",
            "self_attn.out_proj.weight", "self_attn.out_proj.bias",
        ]:
            dk, ok = f"{dino_prefix}.{suffix}", f"{our_prefix}.{suffix}"
            if dk in dino_state and ok in our_state:
                if dino_state[dk].shape == our_state[ok].shape:
                    transfer[ok] = dino_state[dk]
        ffn_map = {
            "linear1.weight": "ffn.0.weight", "linear1.bias": "ffn.0.bias",
            "linear2.weight": "ffn.3.weight", "linear2.bias": "ffn.3.bias",
        }
        for dino_suffix, our_suffix in ffn_map.items():
            dk, ok = f"{dino_prefix}.{dino_suffix}", f"{our_prefix}.{our_suffix}"
            if dk in dino_state and ok in our_state:
                if dino_state[dk].shape == our_state[ok].shape:
                    transfer[ok] = dino_state[dk]
        norm_map = {"norm2": "norm1", "norm1": "norm2", "norm3": "norm4"}
        for dino_norm, our_norm in norm_map.items():
            for wb in ["weight", "bias"]:
                dk = f"{dino_prefix}.{dino_norm}.{wb}"
                ok = f"{our_prefix}.{our_norm}.{wb}"
                if dk in dino_state and ok in our_state:
                    transfer[ok] = dino_state[dk]

    # --- Box heads ---
    for i in range(6):
        for layer_idx in range(3):
            for wb in ["weight", "bias"]:
                dk = f"bbox_embed.{i}.layers.{layer_idx}.{wb}"
                ok = f"decoder.box_heads.{i}.layers.{layer_idx}.{wb}"
                if dk in dino_state and ok in our_state:
                    if dino_state[dk].shape == our_state[ok].shape:
                        transfer[ok] = dino_state[dk]

    model.load_state_dict(transfer, strict=False)
    n_backbone = sum(1 for k in transfer if k.startswith("backbone."))
    n_enc = sum(1 for k in transfer if k.startswith("encoder."))
    n_dec = sum(1 for k in transfer if k.startswith("decoder."))
    print(
        f"DINO COCO transfer from {ckpt_file.name} | "
        f"total={len(transfer)} keys | backbone={n_backbone} | encoder={n_enc} | decoder={n_dec}"
    )


def build_optimizer(model: FrozenDETRModel):
    """Four-group AdamW: backbone, encoder+decoder, deformable (0.1x), heads."""
    deform_params = model.deformable_parameters()
    lr_deform = C.LR_ENCODER_DECODER * C.LR_DEFORM_MULT
    param_groups = [
        {"params": model.backbone_parameters(),          "lr": C.LR_BACKBONE},
        {"params": model.encoder_decoder_parameters(),   "lr": C.LR_ENCODER_DECODER},
        {"params": deform_params,                        "lr": lr_deform},
        {"params": model.head_parameters(),              "lr": C.LR_HEADS},
    ]
    print(f"Optimizer groups: backbone={C.LR_BACKBONE}, enc/dec={C.LR_ENCODER_DECODER}, "
          f"deform={lr_deform} ({len(deform_params)} params), heads={C.LR_HEADS}")
    return AdamW(param_groups, weight_decay=C.WEIGHT_DECAY)


_LR_GROUPS = [C.LR_BACKBONE, C.LR_ENCODER_DECODER,
              C.LR_ENCODER_DECODER * C.LR_DEFORM_MULT, C.LR_HEADS]


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


def validate(model, loader, criterion, matcher, device):
    model.eval()
    totals = {
        "L_total": 0.0, "L_cls": 0.0, "L_bbox": 0.0,
        "L_giou": 0.0, "L_tnorm": 0.0, "L_aux": 0.0,
    }
    n = 0
    action_scores = []
    action_targets = []

    off = C.CLS_OFFSETS

    with torch.no_grad():
        for pil_frames, frame_targets in loader:
            outputs = model(pil_frames)
            loss, log = criterion(outputs, frame_targets)
            for k in totals:
                totals[k] += log.get(k, 0.0)
            n += 1

            gt_tubes = greedy_group_tubes(frame_targets, iou_thresh=C.TUBE_LINK_IOU)
            matched_pred, matched_gt = matcher(
                outputs["pred_boxes"], outputs["pred_logits"], gt_tubes
            )
            if len(matched_pred) == 0:
                continue

            # Slice action from flat 184-dim logits
            probs = outputs["pred_logits"][matched_pred].sigmoid()
            action_probs = probs[:, off["action"]:off["action"] + C.N_ACTIONS]
            gts = torch.stack(
                [gt_tubes[int(j)]["labels"]["action"] for j in matched_gt], dim=0
            ).to(action_probs.device)
            action_scores.append(action_probs)
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
    parser.add_argument("--anno", default=C.ANNO_FILE)
    parser.add_argument("--frames", default=C.FRAMES_DIR)
    parser.add_argument("--epochs", type=int, default=C.MAX_EPOCHS)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--warmstart-dino", default=C.DINO_COCO_CKPT,
                        help="DINO COCO checkpoint for R50+encoder+decoder pretrained weights")
    parser.add_argument("--ckpt-dir", default=C.CKPT_DIR)
    parser.add_argument("--log-dir", default=C.LOG_DIR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_dir) / "metrics.jsonl"

    train_augment = ClipAugmentation(train=True)
    print("Train augmentation: DETR-standard (flip + multi-scale [480..800] + random crop)")
    print(f"Val resize: short_side={C.VAL_SHORT_SIDE}, max_size={C.VAL_MAX_SIZE}")

    train_ds = ROADWaymoDataset(
        args.anno, args.frames, split="train", clip_len=C.CLIP_LEN, stride=C.CLIP_STRIDE
    )
    val_ds = ROADWaymoDataset(
        args.anno, args.frames, split="val", clip_len=C.CLIP_LEN, stride=C.CLIP_STRIDE
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    print(f"train clips: {len(train_ds):,} | val clips: {len(val_ds):,}")

    print("Building model (R50 + CLIP ViT-L/14 + Frozen-DETR + flat 184-dim head)...")
    model = FrozenDETRModel(
        clip_model=C.CLIP_MODEL,
        clip_cache_dir=C.CLIP_CACHE_DIR,
        clip_cls_dim=C.CLIP_CLS_DIM,
        clip_patch_dim=C.CLIP_PATCH_DIM,
        clip_grid=C.CLIP_GRID,
        d_model=C.D_MODEL,
        num_classes=C.NUM_CLASSES,
        clip_len=C.CLIP_LEN,
        num_queries=C.NUM_QUERIES,
        num_encoder_layers=C.NUM_ENCODER_LAYERS,
        num_decoder_layers=C.NUM_DECODER_LAYERS,
        nhead=C.NHEAD,
        dim_ffn=C.DIM_FFN,
        dropout=C.DROPOUT,
        n_deform_points=C.N_DEFORM_POINTS,
        encoder_n_levels=C.ENCODER_N_LEVELS,
        decoder_n_levels=C.FPN_LEVELS,
        fpn_in_channels=C.FPN_IN_CHANNELS,
        val_short_side=C.VAL_SHORT_SIDE,
        val_max_size=C.VAL_MAX_SIZE,
    )
    model = model.to(device)

    if not args.resume and args.warmstart_dino:
        load_dino_coco_weights(model, args.warmstart_dino, device)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = n_total - n_train
    print(f"Total params: {n_total:,} | Trainable: {n_train:,} | Frozen: {n_frozen:,}")

    flat_alphas = compute_flat_alphas(args.anno)
    matcher = HungarianMatcher(C.COST_CLASS, C.COST_BBOX, C.COST_GIOU)
    constraint_data = load_constraint_children(args.anno)
    criterion = SetCriterion(
        matcher, flat_alphas=flat_alphas, **constraint_data
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
        model.clip_encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        running = {k: 0.0 for k in ["L_total", "L_cls", "L_bbox", "L_giou", "L_tnorm", "L_aux"]}
        n_batches = 0
        t0 = time.time()
        print(f"Starting epoch {epoch}/{args.epochs} ...")

        for step, (pil_frames, frame_targets) in enumerate(train_loader, start=1):
            pil_frames, frame_targets = train_augment(pil_frames, frame_targets)
            outputs = model(pil_frames)
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
                running[k] += log.get(k, 0.0)
            n_batches += 1

            if step % 25 == 0:
                avg = {k: running[k] / max(n_batches, 1) for k in running}
                elapsed = time.time() - t0
                print(
                    f"  [train] ep{epoch}/{args.epochs} clip {step}/{len(train_loader)} | "
                    f"L={avg['L_total']:.4f} cls={avg['L_cls']:.4f} box={avg['L_bbox']:.4f} "
                    f"giou={avg['L_giou']:.4f} tnorm={avg['L_tnorm']:.4f} aux={avg['L_aux']:.4f} "
                    f"| {elapsed:.0f}s"
                )

        set_cosine_lr(optimizer, epoch, args.epochs)

        train_metrics = {k: v / max(n_batches, 1) for k, v in running.items()}
        val_metrics = validate(model, val_loader, criterion, matcher, device)
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

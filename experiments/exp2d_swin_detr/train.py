#!/usr/bin/env python3
"""Exp2d v2 training: Swin-L-FPN + Frozen-DETR encoder + CLIP ViT-L/14.

v2 changes (anti-overfitting):
  - DINO COCO pretrained encoder/decoder weights
  - DETR-standard augmentations (flip, multi-scale resize, random crop)
  - Swin-L drop path (stochastic depth 0.2)
  - Stronger weight decay (0.05)
  - DIM_FFN 1024 → 2048 (match DINO architecture)
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
from losses import SetCriterion, compute_class_alphas, greedy_group_tubes, load_constraint_children
from matcher import HungarianMatcher
from model import FrozenDETRModel


def _load_module(name: str, path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


exp1_dataset = _load_module("exp1_dataset_for_exp2d", EXP1_DIR / "dataset.py")
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


def load_exp2c_warmstart(model: FrozenDETRModel, ckpt_path: str, device: torch.device) -> None:
    """Transfer compatible weights from exp2c checkpoint.

    Transfers: CLIP encoder, patch_proj, encoder (6 layers), decoder (6 layers),
    cls_heads, agentness_head, query_embed, reference_points_head, box_heads,
    temporal embeddings.

    Does NOT transfer: backbone (EfficientNet -> Swin-L), FPN lateral/smooth convs
    (channel dims changed 40/112/320 -> 384/768/1536).

    Swin-L backbone gets its own ImageNet-22K pretrained weights via timm.
    """
    ckpt_file = Path(ckpt_path)
    if not ckpt_file.exists():
        print(f"Warm-start skipped: {ckpt_file} not found")
        return

    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)

    new_state = model.state_dict()
    transfer = {}
    skipped_shape = []
    skipped_missing = []

    for k, v in state.items():
        # Skip backbone keys (EfficientNet -> Swin-L, architecture changed)
        if k.startswith("backbone."):
            continue
        # Skip FPN keys (channel dims changed: 40/112/320 -> 384/768/1536)
        if k.startswith("fpn."):
            continue

        if k in new_state:
            if v.shape == new_state[k].shape:
                transfer[k] = v
            else:
                skipped_shape.append(k)
        else:
            skipped_missing.append(k)

    missing, unexpected = model.load_state_dict(transfer, strict=False)
    print(
        f"Warm-started from exp2c {ckpt_file.name} | "
        f"transferred={len(transfer)} keys | "
        f"skipped={len(skipped_shape)} (shape mismatch) + "
        f"{len(skipped_missing)} (key missing in new model)"
    )
    # Show what's randomly initialized (new components)
    new_keys = [k for k in missing if not k.startswith("clip_encoder.")]
    if new_keys:
        prefixes = sorted(set(k.split(".")[0] for k in new_keys))
        print(f"  Randomly initialized: {', '.join(prefixes)}")


def load_dino_coco_weights(model: FrozenDETRModel, ckpt_path: str, device: torch.device) -> None:
    """Transfer DINO COCO pretrained weights to our encoder/decoder.

    Source: dino_4scale_r50_1x_coco_checkpoint0011.pth (DINO R50, 12ep COCO, 51.9 AP)
    Architecture: d_model=256, nheads=8, enc/dec=6 layers, dim_feedforward=2048

    Transfers:
      - Encoder (6 layers): self_attn (MSDeformAttn) + FFN (256→2048→256) + norms
      - Decoder (6 layers): self_attn (MHA) + FFN (mapped linear1/2 → ffn.0/3) + norms
      - Box heads (6 per-layer MLPs): 256→256→4

    Skips:
      - Backbone (R50 → Swin-L)
      - FPN/input_proj (different channel dims)
      - Decoder cross_attn (4-level in DINO vs 3-level in ours)
      - Decoder temporal_attn, CLS injection (our additions, not in DINO)
      - Class heads (91 COCO classes → 5 ROAD heads)
    """
    ckpt_file = Path(ckpt_path)
    if not ckpt_file.exists():
        print(f"DINO COCO weights skipped: {ckpt_file} not found")
        return

    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    dino_state = ckpt.get("model", ckpt)
    our_state = model.state_dict()

    transfer = {}

    # --- Encoder: direct key mapping (transformer.encoder → encoder) ---
    for i in range(6):
        dino_prefix = f"transformer.encoder.layers.{i}"
        our_prefix = f"encoder.layers.{i}"
        # self_attn (MSDeformAttn — same structure)
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
        # FFN (linear1/linear2 — same naming)
        for suffix in ["linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias"]:
            dk, ok = f"{dino_prefix}.{suffix}", f"{our_prefix}.{suffix}"
            if dk in dino_state and ok in our_state:
                if dino_state[dk].shape == our_state[ok].shape:
                    transfer[ok] = dino_state[dk]
        # Norms
        for suffix in ["norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"]:
            dk, ok = f"{dino_prefix}.{suffix}", f"{our_prefix}.{suffix}"
            if dk in dino_state and ok in our_state:
                transfer[ok] = dino_state[dk]

    # --- Decoder layers: partial transfer ---
    for i in range(6):
        dino_prefix = f"transformer.decoder.layers.{i}"
        our_prefix = f"decoder.layers.{i}"
        # self_attn (nn.MultiheadAttention — same structure)
        for suffix in [
            "self_attn.in_proj_weight", "self_attn.in_proj_bias",
            "self_attn.out_proj.weight", "self_attn.out_proj.bias",
        ]:
            dk, ok = f"{dino_prefix}.{suffix}", f"{our_prefix}.{suffix}"
            if dk in dino_state and ok in our_state:
                if dino_state[dk].shape == our_state[ok].shape:
                    transfer[ok] = dino_state[dk]
        # FFN: DINO linear1/2 → our ffn.0/ffn.3 (Sequential: Linear, ReLU, Dropout, Linear)
        ffn_map = {
            "linear1.weight": "ffn.0.weight", "linear1.bias": "ffn.0.bias",
            "linear2.weight": "ffn.3.weight", "linear2.bias": "ffn.3.bias",
        }
        for dino_suffix, our_suffix in ffn_map.items():
            dk, ok = f"{dino_prefix}.{dino_suffix}", f"{our_prefix}.{our_suffix}"
            if dk in dino_state and ok in our_state:
                if dino_state[dk].shape == our_state[ok].shape:
                    transfer[ok] = dino_state[dk]
        # Norms: DINO norm2 = self_attn norm → our norm1
        #         DINO norm1 = cross_attn norm → our norm2
        #         DINO norm3 = FFN norm → our norm4
        norm_map = {"norm2": "norm1", "norm1": "norm2", "norm3": "norm4"}
        for dino_norm, our_norm in norm_map.items():
            for wb in ["weight", "bias"]:
                dk = f"{dino_prefix}.{dino_norm}.{wb}"
                ok = f"{our_prefix}.{our_norm}.{wb}"
                if dk in dino_state and ok in our_state:
                    transfer[ok] = dino_state[dk]

    # --- Box heads: DINO bbox_embed.{i} → our decoder.box_heads.{i} ---
    for i in range(6):
        for layer_idx in range(3):  # 3-layer MLP
            for wb in ["weight", "bias"]:
                dk = f"bbox_embed.{i}.layers.{layer_idx}.{wb}"
                ok = f"decoder.box_heads.{i}.layers.{layer_idx}.{wb}"
                if dk in dino_state and ok in our_state:
                    if dino_state[dk].shape == our_state[ok].shape:
                        transfer[ok] = dino_state[dk]

    # Apply transfer
    missing, unexpected = model.load_state_dict(transfer, strict=False)
    n_enc = sum(1 for k in transfer if k.startswith("encoder."))
    n_dec = sum(1 for k in transfer if k.startswith("decoder."))
    print(
        f"DINO COCO transfer from {ckpt_file.name} | "
        f"total={len(transfer)} keys | encoder={n_enc} | decoder={n_dec}"
    )
    # Show what was NOT transferred
    skipped_keys = set()
    for k in dino_state:
        mapped = False
        for ok in transfer:
            if ok in transfer:
                mapped = True
                break
        if not mapped and k.startswith("transformer.encoder") or k.startswith("transformer.decoder"):
            skipped_keys.add(k.split(".")[2])  # e.g. "layers" or "bbox_embed"
    print(f"  Encoder: {n_enc} keys transferred (self_attn + FFN + norms)")
    print(f"  Decoder: {n_dec} keys transferred (self_attn + FFN + norms + box_heads)")
    print(f"  Skipped: cross_attn (4-level→3-level), temporal_attn, CLS injection, class heads")


def load_exp2d_weights_only(model: FrozenDETRModel, ckpt_path: str, device: torch.device) -> None:
    """Transfer backbone/FPN/CLIP/decoder-custom weights from exp2d best checkpoint.

    Loads everything NOT covered by the DINO COCO transfer:
      - Swin-L backbone (trained on ROAD-Waymo for 2 epochs)
      - FPN (trained)
      - CLIP patch_proj (trained)
      - Decoder cross_attn (3-level, incompatible with DINO's 4-level)
      - Decoder temporal_attn + temporal_pos
      - Decoder CLS injection (image_query_proj, image_query_norm)
      - Classification heads + agentness
    """
    ckpt_file = Path(ckpt_path)
    if not ckpt_file.exists():
        print(f"Exp2d weights skipped: {ckpt_file} not found")
        return

    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    our_state = model.state_dict()

    # Only transfer keys that were NOT already set by DINO COCO transfer
    # Focus on: backbone, fpn, patch_proj, clip_encoder, decoder cross_attn/temporal/CLS, heads
    transfer_prefixes = [
        "backbone.", "fpn.", "patch_proj.", "clip_encoder.",
        "cls_heads.", "agentness_head.",
    ]
    # Decoder-specific keys not covered by DINO
    decoder_custom_suffixes = [
        "cross_attn.", "temporal_attn.", "temporal_pos", "norm3.",
    ]
    decoder_cls_prefixes = ["decoder.image_query_proj.", "decoder.image_query_norm."]

    transfer = {}
    skipped_shape = []

    for k, v in state.items():
        # Check if this key should be transferred
        should_transfer = False
        for prefix in transfer_prefixes:
            if k.startswith(prefix):
                should_transfer = True
                break
        if not should_transfer:
            for prefix in decoder_cls_prefixes:
                if k.startswith(prefix):
                    should_transfer = True
                    break
        if not should_transfer:
            for suffix in decoder_custom_suffixes:
                if suffix in k and k.startswith("decoder.layers."):
                    should_transfer = True
                    break

        if should_transfer and k in our_state:
            if v.shape == our_state[k].shape:
                transfer[k] = v
            else:
                skipped_shape.append(k)

    model.load_state_dict(transfer, strict=False)
    prefixes = sorted(set(k.split(".")[0] for k in transfer))
    print(
        f"Exp2d weights from {ckpt_file.name} | "
        f"transferred={len(transfer)} keys | "
        f"skipped={len(skipped_shape)} (shape mismatch) | "
        f"components: {', '.join(prefixes)}"
    )


def build_optimizer(model: FrozenDETRModel):
    """Three-group AdamW: backbone, encoder+decoder, heads."""
    param_groups = [
        {"params": model.backbone_parameters(),          "lr": C.LR_BACKBONE},
        {"params": model.encoder_decoder_parameters(),   "lr": C.LR_ENCODER_DECODER},
        {"params": model.head_parameters(),              "lr": C.LR_HEADS},
    ]
    return AdamW(param_groups, weight_decay=C.WEIGHT_DECAY)


_LR_GROUPS = [C.LR_BACKBONE, C.LR_ENCODER_DECODER, C.LR_HEADS]


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
        "L_giou": 0.0, "L_tnorm": 0.0, "L_agentness": 0.0, "L_aux": 0.0,
    }
    n = 0
    action_scores = []
    action_targets = []

    with torch.no_grad():
        for pil_frames, frame_targets in loader:
            outputs = model(pil_frames)
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
    parser.add_argument("--anno", default=C.ANNO_FILE)
    parser.add_argument("--frames", default=C.FRAMES_DIR)
    parser.add_argument("--epochs", type=int, default=C.MAX_EPOCHS)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--warmstart", default=None,
                        help="Exp2c checkpoint for warm-starting compatible weights (v1 mode)")
    parser.add_argument("--warmstart-dino", default=C.DINO_COCO_CKPT,
                        help="DINO COCO checkpoint for encoder/decoder pretrained weights")
    parser.add_argument("--warmstart-exp2d", default=C.EXP2D_BEST_CKPT,
                        help="Exp2d best checkpoint for backbone/FPN/custom decoder weights")
    parser.add_argument("--ckpt-dir", default=C.CKPT_DIR)
    parser.add_argument("--log-dir", default=C.LOG_DIR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_dir) / "metrics_v2.jsonl"

    # Augmentation (train only)
    train_augment = ClipAugmentation(train=True)
    print("Train augmentation: DETR-standard (flip + multi-scale resize + random crop)")

    train_ds = ROADWaymoDataset(
        args.anno, args.frames, split="train", clip_len=C.CLIP_LEN, stride=C.CLIP_STRIDE
    )
    val_ds = ROADWaymoDataset(
        args.anno, args.frames, split="val", clip_len=C.CLIP_LEN, stride=C.CLIP_STRIDE
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    print(f"train clips: {len(train_ds):,} | val clips: {len(val_ds):,}")

    print("Building model...")
    model = FrozenDETRModel(
        clip_model=C.CLIP_MODEL,
        clip_cache_dir=C.CLIP_CACHE_DIR,
        clip_cls_dim=C.CLIP_CLS_DIM,
        clip_patch_dim=C.CLIP_PATCH_DIM,
        clip_grid=C.CLIP_GRID,
        d_model=C.D_MODEL,
        head_sizes=C.HEAD_SIZES,
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
        backbone_name=C.BACKBONE,
        backbone_freeze_stages=C.BACKBONE_FREEZE_STAGES,
        fpn_in_channels=C.FPN_IN_CHANNELS,
        input_size=C.INPUT_SIZE,
        drop_path_rate=C.DROP_PATH_RATE,
    )
    model = model.to(device)

    # Weight initialization (three modes, checked in order):
    # 1. --resume: full checkpoint restore (model + optimizer + epoch)
    # 2. --warmstart-dino + --warmstart-exp2d: DINO COCO encoder + exp2d backbone/custom
    # 3. --warmstart: legacy exp2c warm-start
    if not args.resume:
        if args.warmstart_dino:
            load_dino_coco_weights(model, args.warmstart_dino, device)
        if args.warmstart_exp2d:
            load_exp2d_weights_only(model, args.warmstart_exp2d, device)
        elif args.warmstart:
            load_exp2c_warmstart(model, args.warmstart, device)

    # Print param counts
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = n_total - n_train
    print(f"Total params: {n_total:,} | Trainable: {n_train:,} | Frozen: {n_frozen:,}")

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
        # Keep CLIP frozen during training
        model.clip_encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        running = {k: 0.0 for k in ["L_total", "L_cls", "L_bbox", "L_giou", "L_tnorm", "L_agentness", "L_aux"]}
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

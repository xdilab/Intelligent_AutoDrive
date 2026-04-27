#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

# line_buffering=True: every print() flushes immediately — essential when
# monitoring training via `tail -f train.log`. Without this, output is
# buffered by the OS and may only appear in large chunks.
sys.stdout.reconfigure(line_buffering=True)

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoProcessor

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]
EXP1_DIR = REPO_ROOT / "experiments" / "exp1_road_r"

# sys.path manipulation: Python won't find our local modules (config, losses, etc.)
# without adding the experiment directory to the module search path.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))
if str(EXP1_DIR) not in sys.path:
    sys.path.append(str(EXP1_DIR))

import config as C
from losses import SetCriterion, compute_class_alphas, greedy_group_tubes, load_constraint_children
from matcher import HungarianMatcher
from model import DETRROADModel


def _load_module(name: str, path: Path):
    """
    Dynamically import a Python file as a module without it needing to be a
    proper package. Used to import ROADWaymoDataset from exp1 without
    restructuring the repo as a package.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


exp1_dataset = _load_module("exp1_dataset_for_exp2", EXP1_DIR / "dataset.py")
ROADWaymoDataset = exp1_dataset.ROADWaymoDataset


def collate_fn(batch):
    """
    DataLoader collation for batch_size=1. Returns the single element directly
    instead of wrapping it in a list. DETR processes one clip at a time (batch=1)
    because: (a) clips have variable numbers of GT tubes, and (b) Hungarian
    matching is per-clip. Batching across clips would require padding and masking.
    """
    assert len(batch) == 1
    return batch[0]


def preprocess_clip(pil_frames, processor, device, dtype):
    """
    Convert a list of T PIL images into the tensor format Qwen's ViT expects.

    Qwen2.5-VL's image_processor does:
        1. Resize each frame to a grid of 14×14 patches (dynamic resolution)
        2. Apply 2×2 spatial merge to produce the smaller patch grid
        3. Pack all frames' patches into a single flat tensor

    Returns:
        pixel_values:   [N_patches_total, 1176] — all frames' patches concatenated
                        1176 = 14×14×6 (6 channels from the 2×2 merge)
        image_grid_thw: [1, 3] — (T, H_patches, W_patches) for this clip
                        Used by the ViT to reconstruct the spatial structure.

    min_pixels / max_pixels bound the resolution: too low = information loss,
    too high = OOM. 448×448 = one Qwen "standard" size.
    """
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
    """
    Computes per-class Average Precision (area under the precision-recall curve).

    Algorithm:
        1. Sort predictions by confidence (descending)
        2. Walk down the sorted list: at each position, compute precision and recall
        3. AP = mean precision at each recall point where a true positive is found

    This is the standard VOC/COCO 11-point AP formulation.

    Returns nan if there are no positive GT examples for this class (common for
    rare ROAD++ classes). The nan values are filtered out before averaging mAP.
    """
    scores = scores.detach().cpu()
    targets = targets.detach().cpu().bool()
    n_pos = int(targets.sum())
    if n_pos == 0:
        return float("nan")

    order = torch.argsort(scores, descending=True)
    targets = targets[order]         # reorder ground truth by confidence rank
    tp = targets.float().cumsum(0)   # running count of true positives
    fp = (~targets).float().cumsum(0)  # running count of false positives
    precision = tp / (tp + fp).clamp(min=1e-6)
    # Average precision at each TP rank (interpolated AP)
    return float((precision[targets].sum() / n_pos).item())


def load_exp1b_vit_warmstart(model: DETRROADModel, ckpt_path: str, device: torch.device) -> None:
    """
    Transfers only the ViT + LoRA weights from exp1b's checkpoint into exp2's model.

    Why warm-start from exp1b:
        Exp1b trained LoRA adapters on the ViT for 15 epochs on ROAD-Waymo frames.
        The ViT already learned ROAD-Waymo-specific visual features. Starting exp2
        from these weights means the ViT doesn't need to relearn basic visual
        adaptation from scratch — it can focus on producing better features for
        the new DETR decoder.

    Why only vit.* keys:
        Exp1b had a completely different downstream architecture (FCOS dense detection
        heads instead of DETR decoder). Those weights have different shapes and
        semantics — they cannot transfer. We take only the shared visual backbone.

    strict=False: necessary because model has many keys exp1b doesn't have
    (decoder, projection, cls_heads). Missing keys are expected and safe.
    """
    ckpt_file = Path(ckpt_path)
    if not ckpt_file.exists():
        print(f"Warm-start skipped: {ckpt_file} not found")
        return

    ckpt = torch.load(ckpt_file, map_location=device)
    state = ckpt.get("model", ckpt)  # handle both raw state_dict and checkpoint dicts
    vit_state = {k: v for k, v in state.items() if k.startswith("vit.")}
    missing, unexpected = model.load_state_dict(vit_state, strict=False)
    print(
        f"Warm-started vit.* from {ckpt_file.name} | "
        f"loaded={len(vit_state)} keys | missing={len(missing)} unexpected={len(unexpected)}"
    )


def build_optimizer(model: DETRROADModel):
    """
    Three-group AdamW with different LRs for different components.

    Rationale for different learning rates:
        LoRA (5e-5):    Small delta on top of frozen pre-trained weights.
                        Too large an LR would destabilise the pre-trained
                        representations that we specifically want to keep.
        Decoder (1e-4): Fresh random initialisation — needs a faster LR to
                        make meaningful progress early in training.
        Heads (1e-4):   Same logic — fresh classification heads need to learn quickly.

    AdamW vs Adam:
        AdamW decouples the weight decay from the gradient update (correct L2
        regularisation). Standard Adam applies weight decay incorrectly by
        multiplying the gradient, not the weight — AdamW fixes this.
    """
    param_groups = [
        {"params": model.lora_parameters(),    "lr": C.LR_LORA},
        {"params": model.decoder_parameters(), "lr": C.LR_DECODER},
        {"params": model.head_parameters(),    "lr": C.LR_HEADS},
    ]
    return AdamW(param_groups, weight_decay=C.WEIGHT_DECAY)


def set_warmup_lr(optimizer, step: int, warmup_steps: int):
    """
    Linear LR warmup for the first warmup_steps gradient steps.

    Why warmup:
        At the start of training, the randomly initialised decoder and heads
        produce arbitrary gradients. Immediately hitting these large gradients
        with full LR can push the LoRA weights far from their pre-trained values
        before the decoder has learned any coherent structure. Warmup ramps the
        LR linearly from 0 to the target, giving the decoder time to stabilise.
        500 warmup steps ≈ 2000 training clips (4 clips per gradient step).

    Called BEFORE optimizer.step() so the LR is correct for the current step.
    No-op after warmup_steps.
    """
    if step >= warmup_steps:
        return
    scales = [C.LR_LORA, C.LR_DECODER, C.LR_HEADS]
    frac = float(step + 1) / max(warmup_steps, 1)
    for group, base_lr in zip(optimizer.param_groups, scales):
        group["lr"] = base_lr * frac


def set_cosine_lr(optimizer, epoch: int, total_epochs: int):
    """
    Cosine annealing after each epoch (called at epoch end).

    Cosine schedule: LR decays from base_lr to 0.1 * base_lr following a cosine curve.
    The 10% floor (min_scale=0.1) prevents the LR from going to zero, which would
    stop learning entirely in the last few epochs.

    Why cosine vs step decay:
        Step decay (e.g., divide by 10 at epoch 10) can cause sharp loss spikes.
        Cosine annealing is smooth and empirically more stable for DETR-style training.
    """
    scales = [C.LR_LORA, C.LR_DECODER, C.LR_HEADS]
    cos = 0.5 * (1.0 + math.cos(math.pi * epoch / max(total_epochs, 1)))
    min_scale = 0.1
    for group, base_lr in zip(optimizer.param_groups, scales):
        group["lr"] = base_lr * (min_scale + (1.0 - min_scale) * cos)


def validate(model, loader, processor, criterion, matcher, device, dtype):
    """
    Validation loop: computes val loss AND matched-query action mAP.

    Val loss: average total loss over all val clips (teacher-forced, like training).
    Matched action mAP: after matching predictions to GT, compute average precision
    for each action class using only the matched queries. This is an internal metric —
    not comparable to the baseline's frame-mAP — but tracks whether the model is
    learning to classify actions correctly on its best detections.

    Why matched mAP instead of raw mAP:
        Raw mAP over all 100 queries would be dominated by the 90 unmatched queries
        with near-zero scores. The matched mAP measures classification quality on
        the detections that actually aligned to GT, which is a cleaner signal.

    Best checkpoint is saved based on matched_action_map (higher = better model).
    """
    model.eval()
    totals = {
        "L_total": 0.0, "L_cls": 0.0, "L_bbox": 0.0,
        "L_giou": 0.0, "L_tnorm": 0.0, "L_agentness": 0.0
    }
    n = 0
    action_scores = []
    action_targets = []

    with torch.no_grad():
        for pil_frames, frame_targets in loader:
            pixel_values, image_grid_thw = preprocess_clip(pil_frames, processor, device, dtype)
            outputs = model(pixel_values, image_grid_thw)
            loss, log = criterion(outputs, frame_targets)
            for k in totals:
                totals[k] += log[k]
            n += 1

            # Re-run matching to find matched queries for the mAP calculation.
            # This is redundant with the matching inside criterion, but criterion
            # doesn't expose matched_pred externally.
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
        scores = torch.cat(action_scores, dim=0)   # [N_total_matched, 22]
        targets = torch.cat(action_targets, dim=0)
        aps = []
        for c in range(scores.shape[1]):
            ap = average_precision(scores[:, c], targets[:, c])
            if ap == ap:  # filter nan (classes with no GT examples in val set)
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
    parser.add_argument("--ckpt-dir", default=C.CKPT_DIR)
    parser.add_argument("--log-dir", default=C.LOG_DIR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16  # ViT runs in bfloat16; decoder casts to float32 internally
    print(f"Device: {device} | dtype: {dtype}")

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_dir) / "metrics.jsonl"  # append-only, one JSON obj per epoch

    # ROADWaymoDataset returns (pil_frames, frame_targets) per clip.
    # clip_len=8: 8 frames per clip. stride=16: clips taken every 16 frames to
    # avoid too much overlap between adjacent training clips.
    train_ds = ROADWaymoDataset(
        args.anno, args.frames, split="train", clip_len=C.CLIP_LEN, stride=C.CLIP_STRIDE
    )
    val_ds = ROADWaymoDataset(
        args.anno, args.frames, split="val", clip_len=C.CLIP_LEN, stride=C.CLIP_STRIDE
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, collate_fn=collate_fn)
    print(f"train clips: {len(train_ds):,} | val clips: {len(val_ds):,}")

    processor = AutoProcessor.from_pretrained(args.model)

    model = DETRROADModel(
        model_id=args.model,
        vit_dim=C.VIT_DIM,           # 3584 — Qwen2.5-VL-7B ViT hidden dim
        d_model=C.D_MODEL,           # 256  — DETR decoder hidden dim
        head_sizes=C.HEAD_SIZES,     # {agent:10, action:22, loc:16, duplex:49, triplet:86}
        clip_len=C.CLIP_LEN,         # 8 frames
        num_queries=C.NUM_QUERIES,   # 100 learnable object queries
        num_decoder_layers=C.NUM_DECODER_LAYERS,  # 6
        nhead=C.NHEAD,               # 8 attention heads
        dim_ffn=C.DIM_FFN,           # 1024 — FFN hidden dim
        dropout=C.DROPOUT,           # 0.1
        freeze_vit=True,
    )
    model.add_lora(
        r=C.LORA_R,                          # 8
        lora_alpha=C.LORA_ALPHA,             # 16
        lora_dropout=C.LORA_DROPOUT,         # 0.05
        target_modules=C.LORA_TARGET_MODULES,  # ["qkv", "proj"]
        n_layers=C.LORA_N_LAYERS,            # 8 — first 8 ViT blocks
    )
    model = model.to(device)

    # Load ViT + LoRA weights from exp1b (15 epochs on ROAD-Waymo)
    load_exp1b_vit_warmstart(model, C.EXP1B_CKPT, device)

    # Per-class inverse-frequency alpha weights from training set statistics
    class_alphas = compute_class_alphas(args.anno)
    matcher = HungarianMatcher(C.COST_CLASS, C.COST_BBOX, C.COST_GIOU)
    constraint_data = load_constraint_children(args.anno)
    criterion = SetCriterion(
        matcher, class_alphas=class_alphas, **constraint_data
    ).to(device)
    optimizer = build_optimizer(model)

    start_epoch = 1
    best_map = -1.0
    global_step = 0  # counts gradient update steps (not clip iterations)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_map = ckpt.get("best_map", -1.0)
        global_step = ckpt.get("global_step", 0)
        print(f"Resumed from epoch {ckpt['epoch']}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)  # set_to_none=True: frees memory vs zero-filling
        running = {k: 0.0 for k in ["L_total", "L_cls", "L_bbox", "L_giou", "L_tnorm", "L_agentness"]}
        n_batches = 0
        t0 = time.time()
        print(f"Starting epoch {epoch}/{args.epochs} ...")

        for step, (pil_frames, frame_targets) in enumerate(train_loader, start=1):
            # --- Forward pass ---
            pixel_values, image_grid_thw = preprocess_clip(pil_frames, processor, device, dtype)
            outputs = model(pixel_values, image_grid_thw)
            loss, log = criterion(outputs, frame_targets)

            # --- Gradient accumulation ---
            # We process batch_size=1 clips but accumulate GRAD_ACCUM=4 clips before
            # a parameter update. Effective batch = 4, matching the original plan.
            # Dividing by GRAD_ACCUM ensures the accumulated gradient is the average
            # over 4 clips (not the sum, which would give 4× the effective LR).
            (loss / C.GRAD_ACCUM).backward()

            if step % C.GRAD_ACCUM == 0:
                # Gradient clipping: prevents the "exploding gradient" problem where
                # a single bad batch sends weights far off course. Clips the global
                # L2 norm of all gradients to 1.0.
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], C.GRAD_CLIP
                )
                # Apply warmup LR scaling BEFORE the step (overrides the base LR
                # with a fraction of it during the warmup period)
                set_warmup_lr(optimizer, global_step, C.WARMUP_STEPS)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            for k in running:
                running[k] += log[k]
            n_batches += 1

            # Progress print every 25 clips — gives enough resolution to spot
            # divergence early without flooding the log
            if step % 25 == 0:
                avg = {k: running[k] / max(n_batches, 1) for k in running}
                print(
                    f"  [train] ep{epoch}/{args.epochs} clip {step}/{len(train_loader)} | "
                    f"L={avg['L_total']:.4f} cls={avg['L_cls']:.4f} box={avg['L_bbox']:.4f} "
                    f"giou={avg['L_giou']:.4f} tnorm={avg['L_tnorm']:.4f}"
                )

        # Cosine LR decay at epoch end (after warmup phase is done)
        set_cosine_lr(optimizer, epoch, args.epochs)

        train_metrics = {k: v / max(n_batches, 1) for k, v in running.items()}
        val_metrics = validate(model, val_loader, processor, criterion, matcher, device, dtype)
        elapsed = time.time() - t0

        # Log all metrics as a JSON line for easy programmatic analysis later
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

        # Always save the latest checkpoint (for crash recovery / resuming)
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_map": best_map,
            "global_step": global_step,
        }
        torch.save(ckpt, Path(args.ckpt_dir) / "latest.pt")

        # Save best checkpoint based on validation matched action mAP
        if val_metrics["matched_action_map"] > best_map:
            best_map = val_metrics["matched_action_map"]
            ckpt["best_map"] = best_map
            torch.save(ckpt, Path(args.ckpt_dir) / "best.pt")
            print(f"  New best matched action mAP: {best_map:.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Experiment 1b training script — Qwen2.5-VL + LoRA + FCOS dense detection.

Usage
-----
# Standard run (warm-starts from Exp1 best.pt automatically):
python -u experiments/exp1b_road_r/train.py

# Override warm-start checkpoint:
python -u experiments/exp1b_road_r/train.py --warm-start path/to/ckpt.pt

# Skip warm-start (train from scratch):
python -u experiments/exp1b_road_r/train.py --no-warm-start

# Resume interrupted Exp1b run:
python -u experiments/exp1b_road_r/train.py --resume experiments/exp1b_road_r/checkpoints/latest.pt

What it does
------------
1. Scans train annotations to compute per-class focal loss α weights
2. Loads QwenROADModel with ViT frozen (for clean weight loading)
3. Warm-loads Exp1 best.pt — ViT + classification heads load; box/agentness heads fresh
4. Adds LoRA adapters to first 8 ViT blocks (base weights preserved)
5. Trains with two param groups: LoRA (lr=5e-5) and heads (lr=1e-4)
6. Per clip: FCOS-style GT assignment, then 4-term loss
7. Checkpoints by action head macro-mAP on foreground tokens
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoProcessor

_REPO_ROOT = str(Path(__file__).parent.parent.parent)
_EXP1B_DIR = str(Path(__file__).parent)
_EXP1_DIR  = str(Path(__file__).parent.parent / "exp1_road_r")

# Guarantee exp1b is first on sys.path
if _EXP1B_DIR in sys.path:
    sys.path.remove(_EXP1B_DIR)
sys.path.insert(0, _EXP1B_DIR)

for p in [_EXP1_DIR, _REPO_ROOT]:
    if p not in sys.path:
        sys.path.append(p)

import config as C
from assign import assign_tokens_to_gt, empty_assignment, merge_assignments
from dataset import ROADWaymoDataset
from model import QwenROADModel
from losses import ROADLoss, compute_class_alphas

# Reuse collate_fn and preprocess_clip from Exp1
import importlib.util as _ilu
_exp1_train_spec = _ilu.spec_from_file_location(
    "exp1_train", Path(_EXP1_DIR) / "train.py"
)
_exp1_train = _ilu.module_from_spec(_exp1_train_spec)
_exp1_train_spec.loader.exec_module(_exp1_train)
collate_fn      = _exp1_train.collate_fn
preprocess_clip = _exp1_train.preprocess_clip


# ── AP computation ─────────────────────────────────────────────────────────────

def compute_ap(scores: np.ndarray, labels: np.ndarray) -> float:
    if labels.sum() == 0:
        return float("nan")
    order  = np.argsort(-scores)
    labels = labels[order]
    tp     = np.cumsum(labels)
    fp     = np.cumsum(1 - labels)
    rec    = tp / max(labels.sum(), 1)
    prec   = tp / np.maximum(tp + fp, 1)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p   = prec[rec >= t]
        ap += p.max() if len(p) > 0 else 0.0
    return ap / 101


def action_macro_map(all_scores: np.ndarray, all_gt: np.ndarray) -> float:
    """Macro-AP over action classes with at least one GT positive."""
    n_classes = all_scores.shape[1]
    aps   = [compute_ap(all_scores[:, c], all_gt[:, c]) for c in range(n_classes)]
    valid = [ap for ap in aps if not np.isnan(ap)]
    return float(np.mean(valid)) if valid else 0.0


# ── GT assignment helper ───────────────────────────────────────────────────────

def build_assignment(frame_targets, frame_shapes, device):
    """
    Build merged FCOS assignment from per-frame targets and token grid shapes.

    Args:
        frame_targets: list of T Optional[dict] — None or {'boxes':..., 'agent':..., ...}
        frame_shapes:  list of T (H', W') tuples from preds["frame_shapes"]
        device:        torch device

    Returns:
        merged assignment dict (all tokens concatenated), or None if all frames empty
    """
    per_frame = []
    for target, (H_prime, W_prime) in zip(frame_targets, frame_shapes):
        n_tokens = H_prime * W_prime
        if target is None or target["boxes"].shape[0] == 0:
            per_frame.append(empty_assignment(n_tokens, device))
        else:
            gt_labels = {k: target[k] for k in ("agent", "action", "loc", "duplex", "triplet")}
            per_frame.append(assign_tokens_to_gt(
                H_prime, W_prime,
                target["boxes"], gt_labels, device,
            ))
    return merge_assignments(per_frame)


# ── One-epoch loop ─────────────────────────────────────────────────────────────

LOG_EVERY = 50


def run_epoch(
    model,
    loader,
    processor,
    loss_fn,
    optimizer,
    device,
    dtype,
    is_train:     bool,
    warmup_steps: int  = 0,
    global_step:  list = None,
    epoch:        int  = 0,
    n_epochs:     int  = 0,
):
    model.train(is_train)

    totals  = {"L_agentness": 0.0, "L_box": 0.0, "L_focal": 0.0,
               "L_tnorm": 0.0, "L_total": 0.0}
    n_clips = 0
    split   = "train" if is_train else "val"
    n_total = len(loader)
    t_epoch = time.time()
    total_fg = 0

    # Accumulate foreground action predictions for mAP (val only)
    action_scores_list = []
    action_gt_list     = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for pil_frames, frame_targets in loader:
            pixel_values, image_grid_thw = preprocess_clip(
                pil_frames, processor, device, dtype
            )

            preds = model(pixel_values, image_grid_thw)

            # FCOS GT assignment — no GT boxes needed in model forward
            assignment = build_assignment(
                frame_targets, preds["frame_shapes"], device
            )

            loss, log = loss_fn(preds, assignment)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                trainable = [p for p in model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, C.GRAD_CLIP)
                optimizer.step()

                if global_step is not None and global_step[0] < warmup_steps:
                    gs = global_step[0] + 1
                    global_step[0] = gs
                    frac = gs / warmup_steps
                    for pg in optimizer.param_groups:
                        pg["lr"] = pg["_base_lr"] * frac
            else:
                # Collect foreground token predictions for mAP
                is_fg = assignment["is_fg"]
                if is_fg.any():
                    action_scores_list.append(
                        preds["action"][is_fg].float().cpu().numpy()
                    )
                    action_gt_list.append(
                        assignment["action_target"][is_fg].cpu().numpy()
                    )

            for k in totals:
                totals[k] += log[k]
            n_clips  += 1
            total_fg += log["n_fg"]

            if n_clips % LOG_EVERY == 0:
                elapsed  = time.time() - t_epoch
                avg      = {k: v / n_clips for k, v in totals.items()}
                avg_fg   = total_fg / n_clips
                lr_lora  = optimizer.param_groups[0]["lr"] if is_train else 0.0
                lr_heads = optimizer.param_groups[1]["lr"] if is_train else 0.0
                print(
                    f"  [{split}] ep{epoch}/{n_epochs} "
                    f"clip {n_clips}/{n_total} | "
                    f"L={avg['L_total']:.4f} "
                    f"agn={avg['L_agentness']:.4f} "
                    f"box={avg['L_box']:.4f} "
                    f"focal={avg['L_focal']:.4f} "
                    f"tnorm={avg['L_tnorm']:.4f} | "
                    f"avg_fg={avg_fg:.1f} | "
                    f"lr_lora={lr_lora:.2e} lr_heads={lr_heads:.2e} | "
                    f"{elapsed:.0f}s elapsed"
                )

    if n_clips == 0:
        return {k: float("nan") for k in totals}, 0.0

    avg_metrics = {k: v / n_clips for k, v in totals.items()}

    map_score = 0.0
    if not is_train and action_scores_list:
        all_scores = np.concatenate(action_scores_list, axis=0)
        all_gt     = np.concatenate(action_gt_list,     axis=0)
        map_score  = action_macro_map(all_scores, all_gt)

    return avg_metrics, map_score


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default=C.MODEL_ID)
    parser.add_argument("--anno",         default=C.ANNO_FILE)
    parser.add_argument("--frames",       default=C.FRAMES_DIR)
    parser.add_argument("--epochs",       type=int,   default=C.MAX_EPOCHS)
    parser.add_argument("--lambda-tnorm", type=float, default=C.LAMBDA_TNORM,
                        dest="lambda_tnorm")
    parser.add_argument("--tnorm",        default=C.TNORM_TYPE,
                        choices=["lukasiewicz", "godel"])
    parser.add_argument("--ckpt-dir",     default=C.CKPT_DIR,   dest="ckpt_dir")
    parser.add_argument("--log-dir",      default=C.LOG_DIR,    dest="log_dir")
    parser.add_argument("--warm-start",   default=C.EXP1_CKPT,  dest="warm_start",
                        help="Exp1 checkpoint to warm-start from")
    parser.add_argument("--no-warm-start", action="store_true",  dest="no_warm_start")
    parser.add_argument("--resume",       default=None,
                        help="Resume interrupted Exp1b run from this checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16

    print(f"Device: {device} | dtype: {dtype}")
    print(f"Model:  {args.model}")

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_dir) / "metrics.jsonl"

    # ── Step 1: per-class α weights ────────────────────────────────────────────
    print("Computing per-class focal loss α weights …")
    alphas = compute_class_alphas(args.anno)

    # ── Step 2: datasets ───────────────────────────────────────────────────────
    print("Loading datasets …")
    train_ds = ROADWaymoDataset(
        args.anno, args.frames,
        split="train", clip_len=C.CLIP_LEN, stride=C.CLIP_STRIDE,
    )
    val_ds = ROADWaymoDataset(
        args.anno, args.frames,
        split="val", clip_len=C.CLIP_LEN, stride=C.CLIP_STRIDE,
    )
    print(f"  train: {len(train_ds):,} clips  |  val: {len(val_ds):,} clips")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, collate_fn=collate_fn)

    # ── Step 3: processor ─────────────────────────────────────────────────────
    print("Loading processor …")
    processor = AutoProcessor.from_pretrained(args.model)

    # ── Step 4: model — frozen ViT for clean weight loading ───────────────────
    print("Instantiating model (ViT frozen for weight loading) …")
    try:
        model = QwenROADModel(
            model_id   = args.model,
            d_model    = C.VIT_DIM,
            freeze_vit = True,
        )
    except Exception:
        print("ERROR during model init:")
        print(traceback.format_exc())
        raise

    # ── Step 5: warm-start / resume ────────────────────────────────────────────
    start_epoch = 1
    best_map    = 0.0
    global_step = [0]

    if args.resume:
        print("Adding LoRA before loading Exp1b resume checkpoint …")
        model.add_lora(
            r              = C.LORA_R,
            lora_alpha     = C.LORA_ALPHA,
            lora_dropout   = C.LORA_DROPOUT,
            target_modules = C.LORA_TARGET_MODULES,
            n_layers       = C.LORA_N_LAYERS,
        )
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"] + 1
        best_map    = ckpt.get("best_map", 0.0)
        global_step = [ckpt.get("global_step", 0)]
        print(f"  Resumed from epoch {ckpt['epoch']} (best action mAP: {best_map:.4f})")
    else:
        if not args.no_warm_start:
            warm_path = Path(args.warm_start)
            if warm_path.exists():
                print(f"Warm-starting from Exp1: {warm_path} …")
                ckpt = torch.load(warm_path, map_location="cpu")
                missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
                # Expected missing: heads.agentness.*, heads.box.*
                # Expected unexpected: roi_pool.*, tube_link.*
                fg_missing = [k for k in missing if "vit." not in k]
                print(f"  Warm-start complete (Exp1 epoch {ckpt.get('epoch', '?')})")
                if fg_missing:
                    print(f"  Fresh-init keys: {fg_missing}")
            else:
                print(f"  WARNING: warm-start checkpoint not found at {warm_path}")

        print("Adding LoRA adapters …")
        model.add_lora(
            r              = C.LORA_R,
            lora_alpha     = C.LORA_ALPHA,
            lora_dropout   = C.LORA_DROPOUT,
            target_modules = C.LORA_TARGET_MODULES,
            n_layers       = C.LORA_N_LAYERS,
        )

    print(f"Moving model to {device} …")
    model = model.to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total_p   = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_trainable:,} trainable / {n_total_p:,} total")

    # ── Step 6: loss function ──────────────────────────────────────────────────
    with open(args.anno) as f:
        anno_data = json.load(f)

    loss_fn = ROADLoss(
        duplex_childs   = anno_data["duplex_childs"],
        triplet_childs  = anno_data["triplet_childs"],
        lambda_tnorm    = args.lambda_tnorm,
        tnorm           = args.tnorm,
        gamma           = C.FOCAL_GAMMA,
        alphas          = alphas,
        lambda_box      = C.LAMBDA_BOX,
        agentness_gamma = C.AGENTNESS_GAMMA,
    ).to(device)

    # ── Step 7: optimizer ─────────────────────────────────────────────────────
    lora_params = model.vit.lora_parameters()
    head_params = model.head_parameters()

    print(f"Optimizer param groups:")
    print(f"  LoRA:  {sum(p.numel() for p in lora_params):,} params  lr={C.LR_LORA}")
    print(f"  Heads: {sum(p.numel() for p in head_params):,} params  lr={C.LR_HEADS}")

    optimizer = AdamW(
        [
            {"params": lora_params, "lr": C.LR_LORA,  "_base_lr": C.LR_LORA},
            {"params": head_params, "lr": C.LR_HEADS, "_base_lr": C.LR_HEADS},
        ],
        weight_decay = C.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    if args.resume:
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics, _ = run_epoch(
            model, train_loader, processor, loss_fn, optimizer,
            device, dtype, is_train=True,
            warmup_steps=C.WARMUP_STEPS, global_step=global_step,
            epoch=epoch, n_epochs=args.epochs,
        )
        val_metrics, val_map = run_epoch(
            model, val_loader, processor, loss_fn, optimizer,
            device, dtype, is_train=False,
            epoch=epoch, n_epochs=args.epochs,
        )

        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train L={train_metrics['L_total']:.4f} "
            f"(agn={train_metrics['L_agentness']:.4f} "
            f"box={train_metrics['L_box']:.4f} "
            f"focal={train_metrics['L_focal']:.4f} "
            f"tnorm={train_metrics['L_tnorm']:.4f}) | "
            f"val L={val_metrics['L_total']:.4f} | "
            f"val action mAP={val_map:.4f} | "
            f"{elapsed:.0f}s"
        )

        with open(log_path, "a") as f:
            f.write(json.dumps({
                "epoch":       epoch,
                "train":       train_metrics,
                "val":         val_metrics,
                "val_map":     round(val_map, 6),
                "lr_lora":     optimizer.param_groups[0]["lr"],
                "lr_heads":    optimizer.param_groups[1]["lr"],
                "elapsed_s":   round(elapsed, 1),
            }) + "\n")

        ckpt_state = {
            "epoch":       epoch,
            "model":       model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "scheduler":   scheduler.state_dict(),
            "best_map":    best_map,
            "global_step": global_step[0],
        }
        torch.save(ckpt_state, Path(args.ckpt_dir) / "latest.pt")

        if val_map > best_map:
            best_map = val_map
            ckpt_state["best_map"] = best_map
            torch.save(ckpt_state, Path(args.ckpt_dir) / "best.pt")
            print(f"  ★ New best action mAP: {best_map:.4f}")


if __name__ == "__main__":
    main()

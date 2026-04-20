#!/usr/bin/env python3
"""
Experiment 1 training script — ROAD-R with Qwen2.5-VL ViT encoder.

Usage
-----
python experiments/exp1_road_r/train.py
python experiments/exp1_road_r/train.py --epochs 30 --lr 2e-4
python experiments/exp1_road_r/train.py --model Qwen/Qwen2.5-VL-3B-Instruct  # lighter

What it does
------------
1. Loads ROAD-Waymo train/val splits from road_waymo_trainval_v1.1.json
2. Instantiates QwenROADModel (ViT frozen, tube-link + heads trainable)
3. Trains with BCE classification loss + Łukasiewicz t-norm constraint loss
4. Validates every epoch; saves best checkpoint by val total loss
5. Logs per-epoch metrics to logs/metrics.jsonl

Hardware note: The frozen Qwen2.5-VL-7B ViT requires ~16 GB VRAM in bfloat16.
For smaller GPUs, use --model Qwen/Qwen2.5-VL-3B-Instruct (~8 GB).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

# Force line-buffered stdout so tail -f sees output immediately
sys.stdout.reconfigure(line_buffering=True)

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoProcessor

import sys
_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import config as C
from dataset import ROADWaymoDataset
from model import QwenROADModel
from losses import ROADLoss


# ── Collate function ───────────────────────────────────────────────────────────

def collate_fn(batch):
    """Identity collate — returns the single-item batch as-is.

    Each item is (pil_frames: list[PIL], frame_targets: list[dict|None]).
    batch_size=1 is required because clips have variable annotation counts.
    """
    assert len(batch) == 1, "batch_size > 1 requires padding; use batch_size=1 for now"
    return batch[0]


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess_clip(pil_frames, processor, device, dtype):
    """
    Run T PIL frames through the Qwen image_processor.

    We treat each frame as an independent image (t=1 per row in grid_thw)
    so the ViT processes frames independently. Cross-frame context is added
    by TubeLinkingModule after ViT feature extraction.

    Returns:
        pixel_values:   [total_patches, D_in]  on device, dtype
        image_grid_thw: [T, 3]  on device (int64)
    """
    inputs = processor.image_processor(
        images=pil_frames,
        return_tensors="pt",
        min_pixels=C.MIN_PIXELS,
        max_pixels=C.MAX_PIXELS,
    )
    pixel_values   = inputs["pixel_values"].to(device=device, dtype=dtype)
    image_grid_thw = inputs["image_grid_thw"].to(device=device)
    return pixel_values, image_grid_thw


# ── Target stacking ────────────────────────────────────────────────────────────

def stack_targets(frame_targets, device):
    """
    Flatten per-frame GT annotations into a single stacked tensor dict.

    Returns dict of [total_agents, n_class] float32 tensors, or None if
    no frame in the clip has any annotations.
    """
    parts = {k: [] for k in ("agent", "action", "loc", "duplex", "triplet")}
    boxes_per_frame = []

    for t in frame_targets:
        if t is None:
            boxes_per_frame.append(None)
            continue
        boxes_per_frame.append(t["boxes"].to(device))
        for k in parts:
            parts[k].append(t[k].to(device))

    if not any(v for v in parts.values()):
        return None, boxes_per_frame

    stacked = {k: torch.cat(v, dim=0) for k, v in parts.items() if v}
    return stacked, boxes_per_frame


# ── One-epoch loop ─────────────────────────────────────────────────────────────

LOG_EVERY = 50   # print a progress line every this many clips


def run_epoch(
    model,
    loader,
    processor,
    loss_fn,
    optimizer,
    scheduler,
    device,
    dtype,
    is_train: bool,
    warmup_steps: int = 0,
    global_step: list = None,
    epoch: int = 0,
    n_epochs: int = 0,
):
    model.train(is_train)
    totals = {"L_cls": 0.0, "L_tnorm": 0.0, "L_total": 0.0}
    n_clips = 0
    split = "train" if is_train else "val"
    n_total = len(loader)
    t_epoch = time.time()

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for pil_frames, frame_targets in loader:
            # ── Preprocess ──────────────────────────────────────────────────
            pixel_values, image_grid_thw = preprocess_clip(
                pil_frames, processor, device, dtype
            )

            # ── GT boxes and stacked targets ────────────────────────────────
            targets, boxes_per_frame = stack_targets(frame_targets, device)
            if targets is None:
                continue

            # ── Forward ─────────────────────────────────────────────────────
            preds = model(pixel_values, image_grid_thw, boxes_per_frame)
            if preds is None:
                continue

            # ── Loss ────────────────────────────────────────────────────────
            loss, log = loss_fn(preds, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    C.GRAD_CLIP,
                )
                optimizer.step()

                # Linear warmup
                if global_step is not None and global_step[0] < warmup_steps:
                    gs = global_step[0] + 1
                    global_step[0] = gs
                    for pg in optimizer.param_groups:
                        pg["lr"] = C.LR * gs / warmup_steps

            for k in totals:
                totals[k] += log[k]
            n_clips += 1

            if n_clips % LOG_EVERY == 0:
                elapsed = time.time() - t_epoch
                avg = {k: v / n_clips for k, v in totals.items()}
                lr = optimizer.param_groups[0]["lr"] if is_train else 0.0
                print(
                    f"  [{split}] ep{epoch}/{n_epochs} "
                    f"clip {n_clips}/{n_total} | "
                    f"L={avg['L_total']:.4f} "
                    f"cls={avg['L_cls']:.4f} "
                    f"tnorm={avg['L_tnorm']:.4f} | "
                    f"lr={lr:.2e} | "
                    f"{elapsed:.0f}s elapsed"
                )

    if n_clips == 0:
        return {k: float("nan") for k in totals}
    return {k: v / n_clips for k, v in totals.items()}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=C.MODEL_ID)
    parser.add_argument("--anno",       default=C.ANNO_FILE)
    parser.add_argument("--frames",     default=C.FRAMES_DIR)
    parser.add_argument("--epochs",     type=int,   default=C.MAX_EPOCHS)
    parser.add_argument("--lr",         type=float, default=C.LR)
    parser.add_argument("--clip_len",   type=int,   default=C.CLIP_LEN)
    parser.add_argument("--lambda_tnorm", type=float, default=C.LAMBDA_TNORM)
    parser.add_argument("--tnorm",      default=C.TNORM_TYPE,
                        choices=["lukasiewicz", "godel"])
    parser.add_argument("--ckpt_dir",   default=C.CKPT_DIR)
    parser.add_argument("--log_dir",    default=C.LOG_DIR)
    parser.add_argument("--resume",     default=None, help="path to checkpoint to resume from")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16

    print(f"Device: {device} | dtype: {dtype}")
    print(f"Model:  {args.model}")

    # ── Directories ──────────────────────────────────────────────────────────
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_dir) / "metrics.jsonl"

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("Loading dataset …")
    train_ds = ROADWaymoDataset(
        args.anno, args.frames,
        split="train", clip_len=args.clip_len, stride=C.CLIP_STRIDE,
    )
    val_ds = ROADWaymoDataset(
        args.anno, args.frames,
        split="val", clip_len=args.clip_len, stride=C.CLIP_STRIDE,
    )
    print(f"  train clips: {len(train_ds):,}  |  val clips: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, collate_fn=collate_fn)

    # ── Processor (image only — no text tokenizer needed) ─────────────────────
    print(f"Loading processor from {args.model} …")
    processor = AutoProcessor.from_pretrained(args.model)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"Loading model …")
    print("  Instantiating QwenROADModel …")
    try:
        model = QwenROADModel(
            model_id   = args.model,
            d_model    = C.VIT_DIM,
            freeze_vit = C.FREEZE_VIT,
            tube_heads = C.TUBE_N_HEADS,
        )
        print("  Model instantiated on CPU.")
        print(f"  Moving model to {device} …")
        model = model.to(device)
        print(f"  Model moved to {device}.")
    except Exception:
        print("  ERROR during model initialization / device transfer:")
        print(traceback.format_exc())
        raise

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_trainable:,} trainable / {n_total:,} total")

    # ── Loss function ─────────────────────────────────────────────────────────
    import json as _json
    with open(args.anno) as f:
        anno_data = _json.load(f)

    loss_fn = ROADLoss(
        duplex_childs  = anno_data["duplex_childs"],
        triplet_childs = anno_data["triplet_childs"],
        lambda_tnorm   = args.lambda_tnorm,
        tnorm          = args.tnorm,
    ).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=C.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ── Optional resume ───────────────────────────────────────────────────────
    start_epoch = 1
    best_val    = float("inf")
    global_step = [0]

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt.get("best_val", float("inf"))
        global_step = [ckpt.get("global_step", 0)]
        print(f"  Resumed from epoch {ckpt['epoch']} (best val: {best_val:.4f})")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        print(f"Starting epoch {epoch}/{args.epochs} …")

        train_metrics = run_epoch(
            model, train_loader, processor, loss_fn, optimizer,
            scheduler, device, dtype, is_train=True,
            warmup_steps=C.WARMUP_STEPS, global_step=global_step,
            epoch=epoch, n_epochs=args.epochs,
        )
        val_metrics = run_epoch(
            model, val_loader, processor, loss_fn, optimizer,
            scheduler, device, dtype, is_train=False,
            epoch=epoch, n_epochs=args.epochs,
        )

        scheduler.step()
        elapsed = time.time() - t0

        line = (
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train L={train_metrics['L_total']:.4f} "
            f"(cls={train_metrics['L_cls']:.4f} tnorm={train_metrics['L_tnorm']:.4f}) | "
            f"val L={val_metrics['L_total']:.4f} "
            f"(cls={val_metrics['L_cls']:.4f} tnorm={val_metrics['L_tnorm']:.4f}) | "
            f"{elapsed:.0f}s"
        )
        print(line)

        # ── Log ───────────────────────────────────────────────────────────
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "train": train_metrics,
                "val":   val_metrics,
                "lr":    optimizer.param_groups[0]["lr"],
                "elapsed_s": round(elapsed, 1),
            }) + "\n")

        # ── Checkpoint ────────────────────────────────────────────────────
        val_loss = val_metrics["L_total"]
        ckpt = {
            "epoch":       epoch,
            "model":       model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "scheduler":   scheduler.state_dict(),
            "best_val":    best_val,
            "global_step": global_step[0],
        }

        # Always save latest
        torch.save(ckpt, Path(args.ckpt_dir) / "latest.pt")

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            ckpt["best_val"] = best_val
            torch.save(ckpt, Path(args.ckpt_dir) / "best.pt")
            print(f"  ★ New best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()

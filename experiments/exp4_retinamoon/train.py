"""Exp4 — train RetinaMoonFusion on ROAD-Waymo.

Sprint 1 goal: verify loss decreases monotonically over a debug run (--steps 50).
Sprint 2 goal: full 30-epoch run, gate ≥ 17.76% agent f-mAP via eval.py.

Trainable parameter groups (all single LR group at 2e-4):
  - Q-Former (proj_spatial, proj_enc, queries, layers) + frame PE
  - TemporalSelfAttn (one TransformerEncoderLayer)
  - FlatHead (Linear 256 → 184)

Detector and encoder are frozen (set in their wrappers); we run forward through
them under bf16 autocast to halve activation memory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config as C
from dataloader import build_dataloader
from losses import build_criterion, greedy_group_tubes
from model import RetinaMoonFusion


def build_optimizer(model: RetinaMoonFusion) -> torch.optim.Optimizer:
    """Single AdamW group at LR=2e-4 on Q-Former + temporal + head. WD=1e-4."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        trainable, lr=C.LR, weight_decay=C.WEIGHT_DECAY, betas=(0.9, 0.999),
    )


def set_warmup_lr(opt: torch.optim.Optimizer, step: int, warmup: int) -> None:
    """Linear warmup from 0 → C.LR over `warmup` steps; identity after."""
    if step >= warmup:
        for g in opt.param_groups:
            g["lr"] = C.LR
        return
    scale = max(1, step) / max(1, warmup)
    for g in opt.param_groups:
        g["lr"] = C.LR * scale


def set_lr_drop(opt: torch.optim.Optimizer, epoch: int) -> None:
    """Step decay: ×LR_DROP_FACTOR at LR_DROP_EPOCH (kept idempotent across epochs)."""
    if epoch < C.LR_DROP_EPOCH:
        lr = C.LR
    else:
        lr = C.LR * C.LR_DROP_FACTOR
    for g in opt.param_groups:
        g["lr"] = lr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=C.MAX_EPOCHS)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--debug", action="store_true",
                   help="Run --steps steps, no eval, no checkpoint. For Sprint 1 gate.")
    p.add_argument("--steps", type=int, default=50,
                   help="In --debug mode, number of optimizer steps to take.")
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}  bf16=True  debug={args.debug}", flush=True)

    # ---- Data -----------------------------------------------------------
    train_loader = build_dataloader(split="train", batch_size=1,
                                    num_workers=args.num_workers, shuffle=True)
    print(f"[train] train clips: {len(train_loader.dataset):,}", flush=True)

    # ---- Model ----------------------------------------------------------
    print("[train] building RetinaMoonFusion (loads frozen RetinaNet + encoder)...",
          flush=True)
    t0 = time.time()
    model = RetinaMoonFusion().to(device)
    print(f"[train] built in {time.time()-t0:.1f}s", flush=True)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] params total={n_total/1e6:.2f}M  trainable={n_train/1e6:.2f}M  "
          f"frozen={(n_total-n_train)/1e6:.2f}M", flush=True)

    # ---- Loss + optimizer ----------------------------------------------
    criterion = build_criterion(C.ANNO_FILE).to(device)
    optimizer = build_optimizer(model)
    autocast = torch.amp.autocast("cuda", dtype=torch.bfloat16)

    # ---- Checkpoint dir -------------------------------------------------
    ckpt_dir = Path(C.CKPT_DIR); ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(C.LOG_DIR);   log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{time.strftime('%Y%m%d-%H%M%S')}.log"
    print(f"[train] log file: {log_file}", flush=True)

    # ---- Train loop -----------------------------------------------------
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        # Belt-and-braces: keep frozen branches in eval mode (BN, dropout off).
        model.detector.eval()
        model.encoder.eval()
        set_lr_drop(optimizer, epoch)
        optimizer.zero_grad(set_to_none=True)

        running = {"loss": 0.0, "cls": 0.0, "tnorm": 0.0, "matched": 0}
        n_batches = 0
        t_epoch = time.time()
        print(f"[train] === Starting epoch {epoch}/{args.epochs} ===", flush=True)

        for step, (clip, frame_targets) in enumerate(train_loader, start=1):
            clip = clip.to(device, non_blocking=True)
            # Build tubes on CPU (cheap; under 1 ms per clip).
            gt_tubes = greedy_group_tubes(frame_targets, iou_thresh=0.3)

            with autocast:
                outputs = model(clip)
            # Loss in fp32 for numerical safety with focal + small probs.
            outputs["logits"] = outputs["logits"].float()
            outputs["boxes"]  = outputs["boxes"].float()
            loss, log = criterion(outputs, [gt_tubes], epoch=epoch)

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

            if step % 10 == 0 or args.debug:
                avg = {k: running[k] / max(n_batches, 1) for k in running}
                elapsed = time.time() - t_epoch
                lr = optimizer.param_groups[0]["lr"]
                msg = (f"[train] ep{epoch}/{args.epochs} step {step}/{len(train_loader)} "
                       f"gs={global_step}  "
                       f"L={avg['loss']:.4f} cls={avg['cls']:.4f} tnorm={avg['tnorm']:.4f}  "
                       f"matched={avg['matched']:.1f}  lr={lr:.2e}  {elapsed:.0f}s")
                print(msg, flush=True)
                with open(log_file, "a") as f:
                    f.write(msg + "\n")

            if args.debug and global_step >= args.steps:
                print(f"[train] DEBUG: reached {args.steps} optimizer steps, stopping.",
                      flush=True)
                return

        # ---- End of epoch: checkpoint --------------------------------
        if args.debug:
            return
        ckpt_path = ckpt_dir / f"model_{epoch:03d}.pth"
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model": {k: v for k, v in model.state_dict().items()
                      if not k.startswith(("detector.net.", "encoder.encoder."))},
            "optimizer": optimizer.state_dict(),
        }, ckpt_path)
        print(f"[train] saved {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()

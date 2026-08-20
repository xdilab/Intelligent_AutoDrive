"""Exp9 train — strict-alternation round-robin with per-corpus losses.

Each cycle (the unit Moradi described):
    ROAD sample  → focal + λ·t-norm → backward → step → zero_grad
    BDD-X sample → LM loss          → backward → step → zero_grad
    CoVLA sample → LM loss          → backward → step → zero_grad   (if enabled)

Three separate optimizer steps per cycle — no gradient mixing across corpora
(exp8 accumulated the triple into one step; this is the strict-alternation
semantics from DESIGN.md §6). Smaller corpora wrap with per-epoch reshuffles
(exp8 RoundRobinSampler's block logic, inlined). One epoch = one pass over the
ROAD subset (largest corpus drives exp8's epochs; here ROAD is the experiment's
subject, so it drives).

Runs:
  R1  --corpora road                 --lam 0
  R2  --corpora road                 --lam {0.1,1.0,10.0}
  R3  --corpora road,bddx,covla      --lam 0
  R4  --corpora road,bddx,covla      --lam <best from R2>

Usage:
  CUDA_VISIBLE_DEVICES=0 python -u train.py --corpora road --lam 0 --tag r1
  CUDA_VISIBLE_DEVICES=0 python -u train.py --max-cycles 2 --tag smoke  # smoke
"""

from __future__ import annotations

import argparse
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch

import config as C
from dataset import RoadDataset, LangDataset
from losses import RoadCriterion
from model import Exp9Model


def _epoch_perm(n: int, epoch: int, salt: int) -> np.ndarray:
    rng = np.random.default_rng(C.SEED + 1000 * epoch + salt)
    return rng.permutation(n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="road,bddx,covla" if C.COVLA_ENABLED
                    else "road,bddx")
    ap.add_argument("--lam", type=float, default=C.TNORM_LAMBDA)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--tag", required=True, help="checkpoint/log name, e.g. r1")
    ap.add_argument("--max-cycles", type=int, default=0, help="smoke cap; 0 = full")
    ap.add_argument("--head-warmup", type=int, default=C.HEAD_WARMUP_CYCLES,
                    help="cycles of ROAD-only, heads-only training before the "
                         "language legs join (Moradi 2026-08-18). 0 = off.")
    args = ap.parse_args()

    corpora = args.corpora.split(",")
    assert corpora[0] == "road", "road must be in every run (it is the subject)"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Exp9Model(device)
    criterion = RoadCriterion().to(device)
    criterion.tnorm.lam = args.lam

    data = {"road": RoadDataset()}
    if "bddx" in corpora:
        data["bddx"] = LangDataset(C.BDDX_TRAIN_JSON, "bddx")
    if "covla" in corpora:
        data["covla"] = LangDataset(C.COVLA_TRAIN_JSON, "covla")

    lang_lr = C.LR_LORA * C.LANG_LR_SCALE
    opt = torch.optim.AdamW([
        {"params": model.head.parameters(), "lr": C.LR_HEAD},
        {"params": [p for p in model.vlm.parameters() if p.requires_grad],
         "lr": C.LR_LORA},
    ], weight_decay=C.WEIGHT_DECAY)
    # Strict alternation lets us scale the language legs' LoRA LR by swapping
    # the group LR around each language step (config LANG_LR_SCALE, default 1.0).

    trainable = [p for g in opt.param_groups for p in g["params"]]
    print(f"[train] tag={args.tag} corpora={corpora} lam={args.lam} "
          f"head_warmup={args.head_warmup} epochs={args.epochs} "
          f"trainable={sum(p.numel() for p in trainable):,}", flush=True)

    n_cycles_per_epoch = len(data["road"])
    cycle = 0
    for epoch in range(1, args.epochs + 1):
        perms = {n: _epoch_perm(len(d), epoch, salt=s)
                 for s, (n, d) in enumerate(data.items())}
        run_loss = {n: 0.0 for n in corpora}
        run_tnorm = 0.0
        t0 = time.time()
        for i in range(n_cycles_per_epoch):
            in_warmup = cycle < args.head_warmup
            for name in corpora:
                if in_warmup and name != "road":
                    continue                     # heads-first warmup: ROAD leg only
                d = data[name]
                item = d[int(perms[name][i % len(d)])]
                if name == "road":
                    img, boxes, tgts = item
                    if boxes.shape[0] == 0:
                        continue
                    # during warmup the LoRA group is frozen — heads-only updates
                    opt.param_groups[1]["lr"] = 0.0 if in_warmup else C.LR_LORA
                    logits = model.road_forward(img, boxes)
                    loss, parts = criterion(logits, tgts.to(device))
                    run_tnorm += float(parts["tnorm"])
                else:
                    opt.param_groups[1]["lr"] = lang_lr
                    loss = model.lang_forward(*item)
                if not torch.isfinite(loss):
                    print(f"[train] NON-FINITE {name} loss at cycle {cycle} — skip",
                          flush=True)
                    opt.zero_grad(set_to_none=True)
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, C.GRAD_CLIP)
                opt.step()
                opt.zero_grad(set_to_none=True)
                opt.param_groups[1]["lr"] = C.LR_LORA
                run_loss[name] += float(loss)
            cycle += 1
            if args.head_warmup and cycle == args.head_warmup:
                print(f"[train] head warmup complete at cycle {cycle} — "
                      f"language legs + LoRA now active", flush=True)
            if cycle % C.LOG_EVERY == 0:
                el = time.time() - t0
                parts_str = " ".join(
                    f"{n}={run_loss[n] / max(1, min(i + 1, C.LOG_EVERY)):.4f}"
                    for n in corpora)
                tn_str = (f" tnorm={run_tnorm / max(1, min(i + 1, C.LOG_EVERY)):.5f}"
                          if args.lam else "")
                print(f"[train] ep{epoch} cycle {i + 1}/{n_cycles_per_epoch} "
                      f"| {parts_str}{tn_str} | {el / (i + 1):.2f}s/cycle", flush=True)
                run_loss = {n: 0.0 for n in corpora}
                run_tnorm = 0.0
            if args.max_cycles and cycle >= args.max_cycles:
                print(f"[train] smoke cap reached ({args.max_cycles} cycles)",
                      flush=True)
                model.save(C.CKPT_DIR / f"{args.tag}_smoke")
                return
        ck = C.CKPT_DIR / f"{args.tag}_ep{epoch:03d}"
        model.save(ck)
        print(f"[train] epoch {epoch}/{args.epochs} done "
              f"{time.time() - t0:.0f}s → {ck}", flush=True)
    print("[train] finished.", flush=True)


if __name__ == "__main__":
    main()

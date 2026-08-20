"""Exp6 stage 4/4 — train the detection-steered fusion head.

Trains ONLY DetectionSteeredFusion (~0.9M params) on cached (frame, box)
samples from the TRAIN split — Moradi 2026-06-11: "VLM and retina fusion
should be trained on training, not test". No early stopping on the test
split (that would leak model selection into the fixed test set); training
runs the configured epochs, every epoch checkpoints, eval.py scores any
checkpoint on val.

Loss: focal-on-all with the flat per-class alphas (exp2f/exp4 recipe, reused
from exp4/losses.py).

Usage:
  python -u train.py [--split train] [--epochs N] [--limit N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config as C
from model import DetectionSteeredFusion            # exp6's model — import BEFORE exp4 shadows it
from dataset import build_samples, FusionSampleDataset

# exp4's losses.py (focal + flat alphas). Inserting exp4 first on sys.path is
# safe here: `config` and `model` are already bound to exp6's in sys.modules.
sys.path.insert(0, str(C.EXP4_DIR))
from losses import sigmoid_focal_with_logits, compute_flat_alphas  # noqa: E402


def _flat_alphas() -> torch.Tensor:
    """compute_flat_alphas scans the multi-GB anno JSON — cache the result."""
    cache = C.CACHE_DIR / "flat_alphas.pt"
    if cache.exists():
        return torch.load(cache, weights_only=True)
    print("[train] computing flat alphas from anno JSON (one-time) ...", flush=True)
    alphas = compute_flat_alphas(C.ANNO_FILE)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(alphas, cache)
    return alphas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--epochs", type=int, default=C.MAX_EPOCHS)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="First N frames (debug). 0 = all.")
    ap.add_argument("--zero-lang", action="store_true",
                    help="Diagnostic: zero the language tracks (struct + rationale) so the "
                         "head can only recalibrate detector logits. Separates 'objective "
                         "hurts AP' from 'language hurts AP'. Checkpoints as fusion_zl_ep*.")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    C.CKPT_DIR.mkdir(parents=True, exist_ok=True)

    samples = build_samples(args.split, limit=args.limit)
    ds = FusionSampleDataset(samples)
    dl = DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=True,
                    num_workers=2, drop_last=False)
    n_pos = int((samples["target"][:, 0] > 0).sum())
    print(f"[train] {len(ds):,} samples ({n_pos:,} IoU-matched positives, "
          f"{len(ds) - n_pos:,} background)", flush=True)

    model = DetectionSteeredFusion().to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] trainable params: {n_train/1e6:.2f}M", flush=True)

    alphas = _flat_alphas().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss, n_steps = 0.0, 0
        for step, (det, struct, rat, tgt) in enumerate(dl, 1):
            det, struct, rat, tgt = (x.to(device, non_blocking=True)
                                     for x in (det, struct, rat, tgt))
            if args.zero_lang:
                struct = torch.zeros_like(struct)
                rat = torch.zeros_like(rat)
            logits = model(det, struct, rat)
            loss = sigmoid_focal_with_logits(logits, tgt, gamma=C.FOCAL_GAMMA,
                                             alpha=alphas)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            opt.step()
            run_loss += loss.item()
            n_steps += 1
            if step % 200 == 0:
                print(f"[train] ep{epoch} step {step}/{len(dl)} "
                      f"loss={run_loss/n_steps:.5f} "
                      f"({(time.time()-t0)/step:.2f}s/step)", flush=True)

        prefix = "fusion_zl" if args.zero_lang else "fusion"
        ckpt = C.CKPT_DIR / f"{prefix}_ep{epoch:03d}.pth"
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "loss": run_loss / max(n_steps, 1)}, ckpt)
        print(f"[train] epoch {epoch}/{args.epochs} done  "
              f"loss={run_loss/max(n_steps,1):.5f}  "
              f"{time.time()-t0:.0f}s  → {ckpt.name}", flush=True)

    print("[train] finished. Score checkpoints with: "
          "python -u eval.py --ckpt checkpoints/fusion_ep{N}.pth", flush=True)


if __name__ == "__main__":
    main()

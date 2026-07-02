"""Exp4 smoke test — one-clip forward pass through the late-fusion stack.

Validates:
  1. All four components load without errors (detector mock, SigLIP, Q-Former, head).
  2. Forward shapes match the spec in model.py docstrings.
  3. Only Q-Former + head parameters receive gradients (detector + encoder frozen).
  4. No OOM at native res (800x1333, T=8, K=100) on a single A6000.

Run:  CUDA_VISIBLE_DEVICES=1 python -u smoke_test.py
"""

from __future__ import annotations

import sys
import time

import torch

import config as C
from model import RetinaMoonFusion


def fmt(t: torch.Tensor) -> str:
    return f"{tuple(t.shape)} {t.dtype}"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device}")
    print(f"[smoke] encoder={C.ENCODER_NAME}  hidden={C.ENCODER_HIDDEN}  "
          f"token_dim={C.ENCODER_TOKEN_DIM}  (native res via 2D RoPE + pixel-shuffle {C.ENCODER_MERGE}x{C.ENCODER_MERGE})")
    print(f"[smoke] detector=3D-RetinaNet (frozen, epoch 25 ckpt)  K={C.RETINANET_TOPK_TUBES}  D_RETINA={C.RETINANET_SPATIAL_DIM}")
    print(f"[smoke] fusion: D_MODEL={C.D_MODEL}  N_QUERY={C.N_QUERY_TOKENS}  layers={C.NUM_FUSION_LAYERS}")
    print(f"[smoke] head: NUM_CLASSES={C.NUM_CLASSES}")
    sys.stdout.flush()

    # ---- Build model ----
    t0 = time.time()
    print("[smoke] building model (downloads SigLIP on first run; loads RetinaNet ckpt)...", flush=True)
    model = RetinaMoonFusion().to(device)
    print(f"[smoke] built in {time.time()-t0:.1f}s", flush=True)

    # ---- Parameter accounting ----
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[smoke] params total={n_total/1e6:.2f}M  trainable={n_train/1e6:.2f}M  frozen={(n_total-n_train)/1e6:.2f}M")

    # ---- Detector & encoder must be fully frozen ----
    n_det = sum(p.numel() for p in model.detector.parameters() if p.requires_grad)
    n_enc = sum(p.numel() for p in model.encoder.parameters()  if p.requires_grad)
    assert n_det == 0, f"detector has {n_det} trainable params (expected 0)"
    assert n_enc == 0, f"encoder has {n_enc} trainable params (expected 0)"
    print("[smoke] ✓ detector + encoder fully frozen")

    # ---- Trainable params should be Q-Former + temporal + head only ----
    n_fusion   = sum(p.numel() for p in model.fusion.parameters()   if p.requires_grad)
    n_temporal = sum(p.numel() for p in model.temporal.parameters() if p.requires_grad)
    n_head     = sum(p.numel() for p in model.head.parameters()     if p.requires_grad)
    assert n_train == n_fusion + n_temporal + n_head, (
        f"trainable {n_train} != fusion {n_fusion} + temporal {n_temporal} + head {n_head}"
    )
    print(f"[smoke] ✓ trainable accounted for: "
          f"fusion={n_fusion/1e6:.2f}M  temporal={n_temporal/1e6:.3f}M  head={n_head/1e6:.3f}M")

    # ---- Forward pass at native baseline resolution ----
    B, T, H, W = 1, C.CLIP_LEN, C.VAL_SHORT_SIDE, C.VAL_MAX_SIZE
    clip = torch.rand(B, T, 3, H, W, device=device)
    print(f"[smoke] clip input: {fmt(clip)}")

    t0 = time.time()
    out = model(clip)
    fwd_ms = (time.time() - t0) * 1000
    print(f"[smoke] forward: {fwd_ms:.0f} ms")
    print(f"[smoke]   logits: {fmt(out['logits'])}")
    print(f"[smoke]   boxes:  {fmt(out['boxes'])}")
    print(f"[smoke]   scores: {fmt(out['scores'])}")

    # ---- Shape assertions ----
    K = C.RETINANET_TOPK_TUBES
    assert out["logits"].shape == (B, K, T, C.NUM_CLASSES), (
        f"logits shape {tuple(out['logits'].shape)} != {(B, K, T, C.NUM_CLASSES)}"
    )
    assert out["boxes"].shape  == (B, K, T, 4)
    assert out["scores"].shape == (B, K)
    print("[smoke] ✓ all output shapes match spec")

    # ---- Grad-flow check: only fusion + head should accumulate grads ----
    loss = out["logits"].sum()
    loss.backward()
    bad = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is None:
            bad.append((name, "no grad"))
        if (not p.requires_grad) and p.grad is not None and p.grad.abs().sum() > 0:
            bad.append((name, "grad on frozen"))
    if bad:
        print("[smoke] ✗ gradient flow problems:")
        for name, msg in bad[:10]:
            print(f"          {name}: {msg}")
        sys.exit(1)
    print("[smoke] ✓ gradients flow through fusion + head only")

    # ---- GPU memory snapshot ----
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[smoke] peak GPU mem: {peak:.2f} GB")

    print("[smoke] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

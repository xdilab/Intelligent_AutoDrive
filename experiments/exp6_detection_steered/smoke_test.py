"""Exp6 smoke test — fusion head shapes, floor guarantee, grad flow.

Validates (no caches needed; random tensors are shape probes, not results):
  1. Model builds; trainable surface is the fusion head only.
  2. FLOOR GUARANTEE: at init, output logits == detector logits exactly
     (zero-init delta layer) — the fused model starts AT the detector control.
  3. Backward pass reaches every trainable parameter.
  4. _struct_vector layout: dims, null flags, and field offsets.
If cache/detections_val.pkl + rationale_val.pkl exist, also assembles real
samples end-to-end and runs one optimizer step on them.

Run:  python -u smoke_test.py
"""

from __future__ import annotations

import sys

import numpy as np
import torch

import config as C
from model import DetectionSteeredFusion
from dataset import _struct_vector


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device}")
    print(f"[smoke] dims: det={C.NUM_CLASSES} struct={C.STRUCT_DIM} "
          f"rat={C.RATIONALE_DIM}+1 d_model={C.D_MODEL}", flush=True)

    # ---- 1. Build + param accounting ----
    model = DetectionSteeredFusion().to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    assert n_train == n_total, "fusion head must be fully trainable"
    print(f"[smoke] ✓ model built, trainable={n_train/1e6:.2f}M (all params)")

    # ---- 2. Floor guarantee: identity at init ----
    B = 8
    det = torch.randn(B, C.NUM_CLASSES, device=device)
    struct = torch.rand(B, C.STRUCT_DIM, device=device)
    rat = torch.randn(B, C.RATIONALE_DIM + 1, device=device)
    model.eval()
    with torch.no_grad():
        out = model(det, struct, rat)
    assert out.shape == (B, C.NUM_CLASSES), f"bad output shape {tuple(out.shape)}"
    max_dev = (out - det).abs().max().item()
    assert max_dev == 0.0, f"init output deviates from detector logits by {max_dev}"
    print("[smoke] ✓ floor guarantee: init output == detector logits exactly")

    # ---- 3. Grad flow ----
    model.train()
    loss = model(det, struct, rat).square().mean()
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"params with no grad: {missing}"
    print("[smoke] ✓ gradients reach every trainable parameter")

    # ---- 4. Struct vector layout ----
    agent_idx = {f"A{i}": i for i in range(C.N_AGENTS)}
    action_idx = {f"B{i}": i for i in range(C.N_ACTIONS)}
    loc_idx = {f"L{i}": i for i in range(C.N_LOCS)}
    v = _struct_vector({"agent": "A3", "actions": ["B0", "B5"], "locations": ["L2"],
                        "risk": "high", "rationale": "x"}, agent_idx, action_idx, loc_idx)
    assert v.shape == (C.STRUCT_DIM,)
    assert v[3] == 1.0 and v[C.N_AGENTS] == 0.0                     # agent one-hot, no null
    assert v[11 + 0] == 1.0 and v[11 + 5] == 1.0                    # actions after agent block
    assert v[33 + 2] == 1.0                                         # locations block
    assert v[49 + 2] == 1.0 and v[49 + 3] == 0.0                    # risk=high, no null
    assert v[-1] == 1.0                                             # parsed flag
    vn = _struct_vector(None, agent_idx, action_idx, loc_idx)
    assert vn[C.N_AGENTS] == 1.0 and vn[49 + 3] == 1.0 and vn[-1] == 0.0
    assert vn.sum() == 2.0                                          # only the two nulls
    print("[smoke] ✓ struct vector layout (one-hots, nulls, parsed flag)")

    # ---- 5. Real-cache micro-run (optional) ----
    det_pkl = C.CACHE_DIR / "detections_val.pkl"
    rat_pkl = C.CACHE_DIR / "rationale_val.pkl"
    if det_pkl.exists() and rat_pkl.exists():
        from dataset import build_samples
        s = build_samples("val", limit=20)
        if s["det"].shape[0] > 0:
            opt = torch.optim.AdamW(model.parameters(), lr=C.LR)
            logits = model(s["det"][:64].to(device), s["struct"][:64].to(device),
                           s["rat"][:64].to(device))
            l = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, s["target"][:64].to(device))
            opt.zero_grad(); l.backward(); opt.step()
            print(f"[smoke] ✓ real-cache micro-step on {min(64, s['det'].shape[0])} samples "
                  f"(loss={l.item():.4f})")
    else:
        print("[smoke] (cache pkls not present — skipped real-cache micro-run; "
              "run dump_detections.py + embed_rationale.py first)")

    print("[smoke] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

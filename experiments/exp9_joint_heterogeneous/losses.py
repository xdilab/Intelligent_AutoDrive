"""Exp9 ROAD-leg loss: flat 184-dim focal (exp2f/exp4 recipe) + corrected t-norm.

The t-norm penalty set is built from constraints_verified.json — index tuples
derived from the duplex_labels/triplet_labels strings. The JSON's own
duplex_childs/triplet_childs arrays are mis-indexed fossils of the original
ROAD's label ordering and must never be used (see DESIGN.md §9.2 and
wiki findings/road-waymo-childs-mis-indexed).

sigmoid_focal_with_logits is copied verbatim from exp4_retinamoon/losses.py
(importing it would collide: both modules are named `losses`). The 184-dim
alpha vector is exp4's compute_flat_alphas output, loaded from exp6's cache —
same anno JSON, deterministic.
"""

from __future__ import annotations

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C

AGENT_OFF = 1
ACTION_OFF = 1 + C.N_AGENTS                       # 11
LOC_OFF = 1 + C.N_AGENTS + C.N_ACTIONS            # 33

_ALPHAS_CACHE = C.EXP6_DIR / "cache" / "flat_alphas.pt"


def sigmoid_focal_with_logits(logits, targets, gamma=2.0, alpha=None):
    """Verbatim from exp4_retinamoon/losses.py (exp2f focal-on-all recipe)."""
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = targets * prob + (1.0 - targets) * (1.0 - prob)
    loss = ce * (1.0 - pt).pow(gamma)
    if alpha is not None:
        alpha = alpha.to(logits.device)
        alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha)
        loss = alpha_t * loss
    return loss.mean()


def flat_alphas() -> torch.Tensor:
    assert _ALPHAS_CACHE.exists(), (
        f"{_ALPHAS_CACHE} missing — run exp6/train.py once (it caches "
        "compute_flat_alphas over the anno JSON) or copy the file"
    )
    a = torch.load(_ALPHAS_CACHE, weights_only=True)
    assert a.shape == (C.NUM_CLASSES,), f"alphas shape {tuple(a.shape)} != (184,)"
    return a


class CorrectedTNorm(nn.Module):
    """Gödel/Łukasiewicz violation penalty over INVALID compositions, where the
    valid sets come from constraints_verified.json (49 duplex / 86 triplet)."""

    def __init__(self, tnorm: str = C.TNORM, lam: float = C.TNORM_LAMBDA):
        super().__init__()
        spec = json.loads(C.CONSTRAINTS_JSON.read_text())
        valid_d = set(map(tuple, spec["duplex_childs_derived"]))
        valid_t = set(map(tuple, spec["triplet_childs_derived"]))
        assert len(valid_d) == C.N_DUPLEXES, f"expected 49 valid duplexes, got {len(valid_d)}"
        assert len(valid_t) == C.N_TRIPLETS, f"expected 86 valid triplets, got {len(valid_t)}"

        inv_d = [(i, j) for i in range(C.N_AGENTS) for j in range(C.N_ACTIONS)
                 if (i, j) not in valid_d]
        inv_t = [(i, j, k) for i in range(C.N_AGENTS) for j in range(C.N_ACTIONS)
                 for k in range(C.N_LOCS) if (i, j, k) not in valid_t]
        self.register_buffer("inv_d", torch.tensor(inv_d, dtype=torch.long))
        self.register_buffer("inv_t", torch.tensor(inv_t, dtype=torch.long))
        self.lam = lam
        self.fn = (lambda a, b: torch.min(a, b)) if tnorm == "godel" \
            else (lambda a, b: torch.clamp(a + b - 1.0, min=0.0))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """logits: [N, 184] raw head outputs. Returns scalar lam * violation."""
        if self.lam == 0.0 or logits.shape[0] == 0:
            return logits.new_zeros(())
        p = torch.sigmoid(logits.float())
        pa = p[:, AGENT_OFF + self.inv_d[:, 0]]
        pc = p[:, ACTION_OFF + self.inv_d[:, 1]]
        loss = self.fn(pa, pc).mean()
        pa = p[:, AGENT_OFF + self.inv_t[:, 0]]
        pc = p[:, ACTION_OFF + self.inv_t[:, 1]]
        pl = p[:, LOC_OFF + self.inv_t[:, 2]]
        loss = loss + self.fn(self.fn(pa, pc), pl).mean()
        return self.lam * loss


class RoadCriterion(nn.Module):
    """focal-on-all(184) + lam * corrected t-norm. Targets include explicit
    all-zeros rows for unmatched detector boxes (exp2f negative-supervision
    lesson) — the focal term covers every box, matched or not."""

    def __init__(self):
        super().__init__()
        self.register_buffer("alphas", flat_alphas())
        self.tnorm = CorrectedTNorm()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        focal = sigmoid_focal_with_logits(
            logits.float(), targets, gamma=C.FOCAL_GAMMA, alpha=self.alphas
        )
        tn = self.tnorm(logits)
        return focal + tn, {"focal": focal.detach(), "tnorm": tn.detach()}

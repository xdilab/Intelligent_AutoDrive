"""
Loss functions for Experiment 1b (dense detection redesign).

Total loss = L_agentness + λ_box * L_box + L_focal + L_tnorm

L_agentness : focal loss on ALL tokens (fg target=1, bg target=0)
              Handles extreme class imbalance (e.g. 4/484 tokens are fg)
L_box       : SmoothL1 on FCOS (l,t,r,b) targets — FOREGROUND tokens only
L_focal     : per-head focal BCE with per-class α — FOREGROUND tokens only
L_tnorm     : Gödel t-norm constraint — FOREGROUND tokens only;
              uses predicted agentness (not hardcoded 1.0)

Per-class α for focal loss:
    α_c = 1 - (n_positive_c / n_total_instances)   clipped to [0.01, 0.99]
Computed from training split by compute_class_alphas().
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tnorm_loss import TNormConstraintLoss


# ── Per-class α computation (unchanged from original Exp1b) ───────────────────

def compute_class_alphas(anno_file: str) -> Dict[str, torch.Tensor]:
    """
    Scan the training split of the annotation JSON and compute per-class
    inverse-frequency α weights for focal loss.

    Returns dict mapping head name → FloatTensor [n_classes], values in [0.01, 0.99].
    """
    with open(anno_file) as f:
        data = json.load(f)

    agent_labels   = data["agent_labels"]
    action_labels  = data["action_labels"]
    loc_labels     = data["loc_labels"]
    triplet_labels = data["triplet_labels"]
    loc_set        = set(loc_labels)

    triplet_lookup: Dict[tuple, int] = {}
    for idx, s in enumerate(triplet_labels):
        for agt in agent_labels:
            if not s.startswith(agt + "-"):
                continue
            rest = s[len(agt) + 1:]
            for act in action_labels:
                if not rest.startswith(act + "-"):
                    continue
                loc = rest[len(act) + 1:]
                if loc in loc_set:
                    triplet_lookup[(agt, act, loc)] = idx
                break
            break

    counts = {
        "agent":   torch.zeros(10),
        "action":  torch.zeros(22),
        "loc":     torch.zeros(16),
        "duplex":  torch.zeros(49),
        "triplet": torch.zeros(86),
    }
    n_instances = 0

    for vname, vdata in data["db"].items():
        split_ids = vdata.get("split_ids", [])
        if isinstance(split_ids, str):
            split_ids = [split_ids]
        is_train = "train" in split_ids or ("all" in split_ids and "val" not in split_ids)
        if not is_train:
            continue

        for fid, fdata in vdata.get("frames", {}).items():
            if not fdata.get("annotated", 0):
                continue
            for anno in fdata.get("annos", {}).values():
                if not isinstance(anno, dict) or anno.get("box") is None:
                    continue

                n_instances += 1
                agent_ids  = [i for i in anno.get("agent_ids",  []) if 0 <= i < 10]
                action_ids = [i for i in anno.get("action_ids", []) if 0 <= i < 22]
                loc_ids    = [i for i in anno.get("loc_ids",    []) if 0 <= i < 16]
                duplex_ids = [i for i in anno.get("duplex_ids", []) if 0 <= i < 49]

                for i in agent_ids:  counts["agent"][i]  += 1
                for i in action_ids: counts["action"][i] += 1
                for i in loc_ids:    counts["loc"][i]    += 1
                for i in duplex_ids: counts["duplex"][i] += 1

                for a_i in agent_ids:
                    for ac_i in action_ids:
                        for l_i in loc_ids:
                            key = (agent_labels[a_i], action_labels[ac_i], loc_labels[l_i])
                            t_i = triplet_lookup.get(key)
                            if t_i is not None:
                                counts["triplet"][t_i] += 1

    alphas: Dict[str, torch.Tensor] = {}
    for head, cnt in counts.items():
        freq = cnt / max(n_instances, 1)
        alphas[head] = (1.0 - freq).clamp(0.01, 0.99)

    print(f"  compute_class_alphas: {n_instances:,} train instances scanned")
    for head, a in alphas.items():
        print(f"    {head:8s}  α min={a.min():.3f}  max={a.max():.3f}  mean={a.mean():.3f}")

    return alphas


# ── Focal loss (unchanged) ─────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Binary focal loss for multi-label classification.

    Args:
        gamma: focusing parameter (default 2.0)
        alpha: per-class weight tensor [C]; if None, no class weighting
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(pred, target, reduction="none")   # [N, C]
        pt  = target * pred + (1.0 - target) * (1.0 - pred)
        focal_weight = (1.0 - pt).pow(self.gamma)
        loss = focal_weight * bce

        if self.alpha is not None:
            alpha  = self.alpha.to(pred.device)
            alpha_t = target * alpha + (1.0 - target) * (1.0 - alpha)
            loss   = alpha_t * loss

        return loss.mean()


# ── Per-head focal loss (unchanged) ───────────────────────────────────────────

class ROADFocalLoss(nn.Module):
    """Sum of five per-head focal losses."""

    HEAD_SIZES = {
        "agent":   10,
        "action":  22,
        "loc":     16,
        "duplex":  49,
        "triplet": 86,
    }

    def __init__(self, gamma: float = 2.0, alphas: Optional[Dict[str, torch.Tensor]] = None):
        super().__init__()
        for head in self.HEAD_SIZES:
            alpha = alphas[head] if (alphas and head in alphas) else None
            self.add_module(f"focal_{head}", FocalLoss(gamma, alpha))

    def forward(self, preds: dict, targets: dict) -> torch.Tensor:
        return sum(
            getattr(self, f"focal_{head}")(preds[head], targets[head])
            for head in self.HEAD_SIZES
        )


# ── Full loss ──────────────────────────────────────────────────────────────────

class ROADLoss(nn.Module):
    """
    Experiment 1b loss: agentness focal + box SmoothL1 + classification focal + Gödel t-norm.

    forward(preds, assignment) where:
        preds      : output of QwenROADModel.forward() — dense per-token predictions
        assignment : merged dict from assign.merge_assignments() — per-token GT targets

    Args:
        duplex_childs:    list of [agent_idx, action_idx] valid pairs
        triplet_childs:   list of [agent_idx, action_idx, loc_idx] valid triples
        lambda_tnorm:     weight for t-norm loss
        tnorm:            'godel' or 'lukasiewicz'
        gamma:            focal loss γ (for classification heads)
        alphas:           per-class α weights from compute_class_alphas()
        lambda_box:       weight for box regression loss
        agentness_gamma:  focal γ for agentness head
    """

    def __init__(
        self,
        duplex_childs:    list,
        triplet_childs:   list,
        lambda_tnorm:     float = 1.0,
        tnorm:            str   = "godel",
        gamma:            float = 2.0,
        alphas:           Optional[Dict[str, torch.Tensor]] = None,
        lambda_box:       float = 1.0,
        agentness_gamma:  float = 2.0,
    ):
        super().__init__()
        self.lambda_box = lambda_box

        # Agentness head: scalar focal loss (no per-class α — binary fg/bg)
        self.focal_agentness = FocalLoss(gamma=agentness_gamma, alpha=None)

        # Classification heads
        self.focal_cls = ROADFocalLoss(gamma, alphas)

        # T-norm constraint
        self.tnorm_loss = TNormConstraintLoss(
            duplex_childs,
            triplet_childs,
            n_agents  = 10,
            n_actions = 22,
            n_locs    = 16,
            tnorm     = tnorm,
            lam       = lambda_tnorm,
        )

    def forward(
        self,
        preds:      dict,
        assignment: dict,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            preds:      model output dict (agentness, box, agent, action, loc, duplex, triplet)
            assignment: merged assignment dict (is_fg, ltrb_target, agent_target, ...)

        Returns:
            (total_loss, log_dict)
        """
        is_fg = assignment["is_fg"]    # [T*H'*W'] bool
        n_fg  = is_fg.sum().item()

        # ── Agentness focal loss (all tokens) ─────────────────────────────────
        agentness_target = is_fg.float().unsqueeze(1)   # [N, 1]
        L_agentness = self.focal_agentness(preds["agentness"], agentness_target)

        # ── Box regression (foreground only) ──────────────────────────────────
        if n_fg > 0:
            L_box = F.smooth_l1_loss(
                preds["box"][is_fg],
                assignment["ltrb_target"][is_fg],
                beta      = 0.1,
                reduction = "mean",
            )
        else:
            L_box = preds["box"].new_zeros(1).squeeze()

        # ── Classification focal loss (foreground only) ────────────────────────
        if n_fg > 0:
            fg_preds = {
                head: preds[head][is_fg]
                for head in ("agent", "action", "loc", "duplex", "triplet")
            }
            fg_targets = {
                head: assignment[f"{head}_target"][is_fg]
                for head in ("agent", "action", "loc", "duplex", "triplet")
            }
            L_focal = self.focal_cls(fg_preds, fg_targets)
        else:
            L_focal = preds["agent"].new_zeros(1).squeeze()

        # ── T-norm constraint (foreground only; predicted agentness) ──────────
        if n_fg > 0:
            flat = torch.cat([
                preds["agentness"][is_fg],   # [n_fg, 1] — real learned score
                preds["agent"][is_fg],        # [n_fg, 10]
                preds["action"][is_fg],       # [n_fg, 22]
                preds["loc"][is_fg],          # [n_fg, 16]
            ], dim=1)                         # [n_fg, 49]
            L_tnorm = self.tnorm_loss(flat)
        else:
            L_tnorm = preds["agent"].new_zeros(1).squeeze()

        total = L_agentness + self.lambda_box * L_box + L_focal + L_tnorm

        return total, {
            "L_agentness": L_agentness.item(),
            "L_box":       L_box.item(),
            "L_focal":     L_focal.item(),
            "L_tnorm":     L_tnorm.item(),
            "L_total":     total.item(),
            "n_fg":        int(n_fg),
        }

"""
Loss functions for Experiment 1.

Total loss = L_cls + λ · L_tnorm

L_cls: sum of binary cross-entropy losses for agent / action / loc / duplex / triplet
L_tnorm: Łukasiewicz t-norm constraint violation loss from tnorm_loss.py

The TNormConstraintLoss in tnorm_loss.py expects a flat prediction vector with
the following layout (matching the offsets defined in that file):
    offset 0:      agentness (1 class)   — we set this to 1.0 (all detections are positive)
    offset 1-10:   agent     (10 classes)
    offset 11-32:  action    (22 classes)
    offset 33-48:  loc       (16 classes)

Only these first 49 dimensions are used by the t-norm loss.
"""

import sys
from pathlib import Path

# Allow importing tnorm_loss.py from the ROAD_Reason repo root
_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tnorm_loss import TNormConstraintLoss

import torch
import torch.nn as nn


class ROADClassificationLoss(nn.Module):
    """
    Binary cross-entropy classification loss, summed across all five label types.

    Args:
        label_smoothing: small ε for label smoothing (reduces overconfidence)
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.eps = label_smoothing

    def _bce(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.eps > 0:
            target = target * (1 - self.eps) + self.eps * 0.5
        return nn.functional.binary_cross_entropy(pred, target)

    def forward(self, preds: dict, targets: dict) -> torch.Tensor:
        L = (
            self._bce(preds["agent"],   targets["agent"])
            + self._bce(preds["action"], targets["action"])
            + self._bce(preds["loc"],    targets["loc"])
            + self._bce(preds["duplex"], targets["duplex"])
            + self._bce(preds["triplet"], targets["triplet"])
        )
        return L


class ROADLoss(nn.Module):
    """
    Full Experiment 1 loss: classification BCE + Łukasiewicz t-norm.

    Args:
        duplex_childs:  list of [agent_idx, action_idx] valid pairs
        triplet_childs: list of [agent_idx, action_idx, loc_idx] valid triples
        lambda_tnorm:   weight for constraint loss
        tnorm:          'lukasiewicz' or 'godel'
        label_smoothing: BCE label smoothing
    """

    def __init__(
        self,
        duplex_childs:   list,
        triplet_childs:  list,
        lambda_tnorm:    float = 0.1,
        tnorm:           str   = "lukasiewicz",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.cls_loss  = ROADClassificationLoss(label_smoothing)
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
        preds:   dict,
        targets: dict,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            preds:   dict from ClassificationHeads [N, n_class] each
            targets: dict with same keys, stacked GT labels [N, n_class]

        Returns:
            (total_loss, log_dict)
        """
        # Classification loss
        L_cls = self.cls_loss(preds, targets)

        # T-norm constraint loss
        # Build flat prediction vector [N, 1 + 10 + 22 + 16 = 49]
        N = preds["agent"].shape[0]
        agentness = torch.ones(N, 1, device=preds["agent"].device, dtype=preds["agent"].dtype)
        flat = torch.cat([agentness, preds["agent"], preds["action"], preds["loc"]], dim=1)
        L_tnorm = self.tnorm_loss(flat)

        total = L_cls + L_tnorm

        return total, {
            "L_cls":   L_cls.item(),
            "L_tnorm": L_tnorm.item(),
            "L_total": total.item(),
        }

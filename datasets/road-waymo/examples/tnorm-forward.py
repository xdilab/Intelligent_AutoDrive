"""
T-norm forward pass — extracted from tnorm_loss.py for reference.
Full file: /data/repos/ROAD_Reason/tnorm_loss.py

Flat prediction vector layout (1+10+22+16 = 49 dims used by t-norm loss):
  offset 0:      agentness (1)   — hardcoded to 1.0 in ROADLoss (losses.py:112)
  offset 1-10:   agent     (10)
  offset 11-32:  action    (22)
  offset 33-48:  loc       (16)
  offset 49-97:  duplex    (49)  — not used by t-norm, only by BCE
  offset 98-183: triplet   (86)  — not used by t-norm, only by BCE
"""

import torch
import torch.nn as nn

AGENT_OFFSET  = 1
ACTION_OFFSET = 11
LOC_OFFSET    = 33


def _godel(p_a, p_b):
    # T(a,b) = min(a,b)  — best on ROAD-Waymo per paper Table 7
    return torch.min(p_a, p_b)


def _lukasiewicz(p_a, p_b):
    # T(a,b) = max(0, a+b-1)  — currently configured in exp1 config.py
    return torch.clamp(p_a + p_b - 1.0, min=0.0)


class TNormConstraintLoss(nn.Module):
    def forward(self, preds):
        """
        preds: [N, 49] sigmoid-activated flat predictions for positive anchors.

        Returns scalar = lambda * mean violation over all invalid (agent,action)
        and (agent,action,location) combinations.
        """
        loss = preds.new_zeros(1)

        # ── Duplex violations (invalid agent+action pairs) ──────────────────────
        # inv_d: [K_d, 2] — all (agent_idx, action_idx) NOT in duplex_childs
        # 220 possible pairs − 49 valid = 171 invalid pairs
        if self.inv_d is not None:
            ai  = self.inv_d[:, 0]
            aci = self.inv_d[:, 1]
            p_agent  = preds[:, AGENT_OFFSET  + ai]   # [N, 171]
            p_action = preds[:, ACTION_OFFSET + aci]  # [N, 171]
            loss = loss + self.violation_fn(p_agent, p_action).mean()

        # ── Triplet violations (invalid agent+action+location triples) ──────────
        # inv_t: [K_t, 3] — all (a,ac,l) NOT in triplet_childs
        # 3520 possible − 86 valid = 3434 invalid triples
        if self.inv_t is not None:
            ai  = self.inv_t[:, 0]
            aci = self.inv_t[:, 1]
            li  = self.inv_t[:, 2]
            p_agent  = preds[:, AGENT_OFFSET  + ai]
            p_action = preds[:, ACTION_OFFSET + aci]
            p_loc    = preds[:, LOC_OFFSET    + li]
            # Composite: T(T(agent, action), location) — t-norm applied twice
            loss = loss + self.violation_fn(
                self.violation_fn(p_agent, p_action), p_loc
            ).mean()

        return self.lam * loss


# ── How ROADLoss calls this (from losses.py:109-114) ──────────────────────────
#
# N = preds["agent"].shape[0]
# agentness = torch.ones(N, 1, device=..., dtype=...)   # all positives → 1.0
# flat = torch.cat([agentness, preds["agent"], preds["action"], preds["loc"]], dim=1)
# L_tnorm = self.tnorm_loss(flat)                        # scalar
#
# Config (exp1_road_r/config.py):
#   TNORM_TYPE   = "lukasiewicz"   # per email; paper Table 7 says godel is best
#   LAMBDA_TNORM = 0.1

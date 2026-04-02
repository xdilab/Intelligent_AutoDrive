"""
T-norm constraint violation loss for ROAD-Waymo.

Penalises co-prediction of agent/action/location combinations that are
invalid according to duplex_childs and triplet_childs from the annotation JSON.

Two t-norms supported:
  - 'godel'       : T(a,b) = min(a,b)          [best on ROAD-Waymo per paper Table 7]
  - 'lukasiewicz' : T(a,b) = max(0, a+b-1)

Label layout in flat prediction vector [num_classes_list = 1,10,22,16,49,86]:
  offset 0        : agent_ness (1 class)
  offset 1-10     : agent      (10 classes)
  offset 11-32    : action     (22 classes)
  offset 33-48    : loc        (16 classes)
  offset 49-97    : duplex     (49 classes)
  offset 98-183   : triplet    (86 classes)
"""

import torch
import torch.nn as nn


AGENT_OFFSET  = 1
ACTION_OFFSET = 11
LOC_OFFSET    = 33


def _godel(p_a, p_b):
    return torch.min(p_a, p_b)


def _lukasiewicz(p_a, p_b):
    return torch.clamp(p_a + p_b - 1.0, min=0.0)


class TNormConstraintLoss(nn.Module):
    """
    Args:
        duplex_childs  : list of [agent_idx, action_idx] valid pairs (from JSON)
        triplet_childs : list of [agent_idx, action_idx, loc_idx] valid triples (from JSON)
        n_agents       : number of agent classes (10)
        n_actions      : number of action classes (22)
        n_locs         : number of location classes (16)
        tnorm          : 'godel' or 'lukasiewicz'
        lam            : loss weighting factor (lambda)
    """

    def __init__(self, duplex_childs, triplet_childs,
                 n_agents=10, n_actions=22, n_locs=16,
                 tnorm='godel', lam=1.0):
        super().__init__()
        self.lam = lam
        self.violation_fn = _godel if tnorm == 'godel' else _lukasiewicz

        # Build invalid pair tensors (all pairs NOT in childs)
        valid_d = set(map(tuple, duplex_childs))
        invalid_d = [(i, j) for i in range(n_agents) for j in range(n_actions)
                     if (i, j) not in valid_d]

        valid_t = set(map(tuple, triplet_childs))
        invalid_t = [(i, j, k) for i in range(n_agents) for j in range(n_actions)
                     for k in range(n_locs) if (i, j, k) not in valid_t]

        if invalid_d:
            self.register_buffer('inv_d', torch.tensor(invalid_d, dtype=torch.long))
        else:
            self.inv_d = None

        if invalid_t:
            self.register_buffer('inv_t', torch.tensor(invalid_t, dtype=torch.long))
        else:
            self.inv_t = None

    def forward(self, preds):
        """
        Args:
            preds: [N, num_classes] sigmoid-activated predictions for positive anchors
        Returns:
            scalar constraint violation loss
        """
        loss = preds.new_zeros(1)

        if self.inv_d is not None:
            ai = self.inv_d[:, 0]
            aci = self.inv_d[:, 1]
            p_agent  = preds[:, AGENT_OFFSET  + ai]   # [N, K]
            p_action = preds[:, ACTION_OFFSET + aci]   # [N, K]
            loss = loss + self.violation_fn(p_agent, p_action).mean()

        if self.inv_t is not None:
            ai  = self.inv_t[:, 0]
            aci = self.inv_t[:, 1]
            li  = self.inv_t[:, 2]
            p_agent  = preds[:, AGENT_OFFSET  + ai]
            p_action = preds[:, ACTION_OFFSET + aci]
            p_loc    = preds[:, LOC_OFFSET    + li]
            loss = loss + self.violation_fn(
                self.violation_fn(p_agent, p_action), p_loc
            ).mean()

        return self.lam * loss

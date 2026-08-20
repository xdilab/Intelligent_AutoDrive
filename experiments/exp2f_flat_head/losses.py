from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
from matcher import HungarianMatcher, box_cxcywh_to_xyxy, box_iou, box_xyxy_to_cxcywh, generalized_box_iou

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tnorm_loss import TNormConstraintLoss


def compute_flat_alphas(anno_file: str) -> torch.Tensor:
    """
    Build a flat 184-dim alpha vector for focal loss, concatenating per-head
    inverse-frequency weights in the same order as the flat label vector:
        [agentness(1), agent(10), action(22), loc(16), duplex(49), triplet(86)]

    Delegates per-head computation to exp1b's implementation.
    Agentness alpha is set to 0.25 (standard focal loss default for binary objectness).
    """
    from experiments.exp1b_road_r.losses import compute_class_alphas as _compute
    head_alphas = _compute(anno_file)

    parts = [torch.tensor([0.25])]  # agentness
    for head in ("agent", "action", "loc", "duplex", "triplet"):
        parts.append(head_alphas[head])

    return torch.cat(parts, dim=0)  # [184]


def sigmoid_focal_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Focal Loss (Lin et al., RetinaNet, ICCV 2017) for multi-label binary classification.

    Applied to ALL queries (matched AND unmatched). Unmatched queries have
    target=all-zeros, so focal loss actively pushes their predictions toward 0.
    This is the key fix vs exp2e where unmatched queries got zero gradient on
    classification heads.

    FL(p, y) = -α_t · (1 - p_t)^γ · log(p_t)

    Args:
        logits:  [N, C] raw logits (before sigmoid)
        targets: [N, C] binary targets in {0, 1}
        gamma:   focusing parameter (2.0 is standard)
        alpha:   [C] per-class positive weight, or None for uniform weighting
    """
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = targets * prob + (1.0 - targets) * (1.0 - prob)
    loss = ce * (1.0 - pt).pow(gamma)

    if alpha is not None:
        alpha = alpha.to(logits.device)
        alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha)
        loss = alpha_t * loss

    return loss.mean()


def greedy_group_tubes(frame_targets: List[dict | None], iou_thresh: float = 0.3) -> List[dict]:
    """
    Links per-frame annotations into multi-frame tubes using greedy IoU matching.

    Identical to exp2e — the tube format is unchanged. Labels are still stored
    as a dict of per-head multi-hot vectors in each tube.
    """
    tubes: List[dict] = []

    for t, frame in enumerate(frame_targets):
        if frame is None:
            continue

        boxes = frame["boxes"]
        n = boxes.shape[0]
        assigned = set()

        if tubes:
            active_boxes = []
            active_idx = []
            for idx, tube in enumerate(tubes):
                prev_frames = torch.where(tube["box_mask"])[0]
                if len(prev_frames) == 0:
                    continue
                last_t = int(prev_frames[-1])
                active_boxes.append(tube["boxes"][last_t])
                active_idx.append(idx)

            if active_boxes:
                active_boxes = torch.stack(active_boxes)
                ious = box_iou(active_boxes, boxes)

                for ai, tube_idx in enumerate(active_idx):
                    best_iou, best_j = ious[ai].max(dim=0)
                    if best_iou >= iou_thresh and int(best_j) not in assigned:
                        assigned.add(int(best_j))
                        tubes[tube_idx]["boxes"][t] = boxes[best_j]
                        tubes[tube_idx]["box_mask"][t] = True

        for j in range(n):
            if j in assigned:
                continue
            labels = {head: frame[head][j].clone() for head in C.HEAD_SIZES}
            tube = {
                "boxes": torch.zeros(len(frame_targets), 4, dtype=torch.float32, device=boxes.device),
                "box_mask": torch.zeros(len(frame_targets), dtype=torch.bool, device=boxes.device),
                "labels": labels,
            }
            tube["boxes"][t] = boxes[j]
            tube["box_mask"][t] = True
            tubes.append(tube)

    return tubes


def _pack_flat_target(gt_tube: dict, device: torch.device) -> torch.Tensor:
    """
    Pack a GT tube's per-head labels into a flat 184-dim target vector.

    Layout: [agentness(1), agent(10), action(22), loc(16), duplex(49), triplet(86)]
    Agentness = 1.0 for any matched tube (it contains an agent by definition).
    """
    target = torch.zeros(C.NUM_CLASSES, device=device)
    target[0] = 1.0  # agentness

    labels = gt_tube["labels"]
    off = C.CLS_OFFSETS
    target[off["agent"]:off["agent"] + C.N_AGENTS] = labels["agent"].to(device)
    target[off["action"]:off["action"] + C.N_ACTIONS] = labels["action"].to(device)
    target[off["loc"]:off["loc"] + C.N_LOCS] = labels["loc"].to(device)
    target[off["duplex"]:off["duplex"] + C.N_DUPLEXES] = labels["duplex"].to(device)
    target[off["triplet"]:off["triplet"] + C.N_TRIPLETS] = labels["triplet"].to(device)

    return target


class SetCriterion(nn.Module):
    """
    Computes all losses for one clip after Hungarian matching.

    Key difference from exp2e: classification loss is computed on ALL 300 queries
    using a single flat 184-dim target vector. Matched queries get their GT labels;
    unmatched queries get all-zeros. This provides explicit negative supervision
    on all 184 dims for unmatched queries — fixing the score-localization
    decorrelation found in exp2e.

    Loss components:
        L_total = 2.0 * L_cls + 5.0 * L_bbox + 2.0 * L_giou + L_tnorm

    No separate L_agentness — agentness is slot 0 in the flat vector,
    supervised as part of L_cls on all queries.
    """

    def __init__(
        self,
        matcher: HungarianMatcher,
        duplex_childs: list,
        triplet_childs: list,
        flat_alphas: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.matcher = matcher
        self.register_buffer("flat_alphas", flat_alphas if flat_alphas is not None else torch.empty(0))
        self.tnorm = TNormConstraintLoss(
            duplex_childs=duplex_childs,
            triplet_childs=triplet_childs,
            n_agents=C.N_AGENTS,
            n_actions=C.N_ACTIONS,
            n_locs=C.N_LOCS,
            tnorm="godel",
            lam=C.LAMBDA_TNORM,
        )

    def _classification_loss(
        self,
        pred_logits: torch.Tensor,
        matched_pred: torch.Tensor,
        gt_tubes: List[dict],
        matched_gt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Flat focal loss on ALL queries × 184 dims.

        Matched queries → packed GT labels (agentness=1, multi-hot per head).
        Unmatched queries → all-zeros (184 dims of negative supervision).

        This is the baseline's approach: every anchor (here, query) gets a target
        for every class. The focal loss down-weights easy negatives (unmatched
        queries quickly learn to predict ~0 everywhere).
        """
        N = pred_logits.shape[0]  # 300
        device = pred_logits.device
        targets = torch.zeros(N, C.NUM_CLASSES, device=device)

        for pi, gi in zip(matched_pred, matched_gt):
            targets[int(pi)] = _pack_flat_target(gt_tubes[int(gi)], device)

        alpha = self.flat_alphas if self.flat_alphas.numel() > 0 else None
        return sigmoid_focal_with_logits(
            pred_logits, targets, gamma=C.FOCAL_GAMMA, alpha=alpha,
        )

    def _box_losses(
        self,
        pred_boxes: torch.Tensor,
        matched_pred: torch.Tensor,
        gt_tubes: List[dict],
        matched_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """L1 and GIoU losses on matched query boxes. Identical to exp2e."""
        if len(matched_pred) == 0:
            zero = pred_boxes.sum() * 0.0
            return zero, zero

        l1_losses = []
        giou_losses = []
        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes[matched_pred])
        pred_cxcywh = pred_boxes[matched_pred]

        for i, gt_idx in enumerate(matched_gt):
            tube = gt_tubes[int(gt_idx)]
            mask = tube["box_mask"].to(device=pred_boxes.device)
            if not mask.any():
                continue
            gt_xyxy = tube["boxes"].to(device=pred_boxes.device)[mask]
            gt_cxcywh = box_xyxy_to_cxcywh(gt_xyxy)

            l1_losses.append(F.l1_loss(pred_cxcywh[i][mask], gt_cxcywh))

            giou = generalized_box_iou(pred_xyxy[i][mask], gt_xyxy)
            giou_losses.append((1.0 - giou.diag()).mean())

        if not l1_losses:
            zero = pred_boxes.sum() * 0.0
            return zero, zero
        return torch.stack(l1_losses).mean(), torch.stack(giou_losses).mean()

    def _tnorm_loss(
        self,
        pred_logits: torch.Tensor,
        matched_pred: torch.Tensor,
        gt_tubes: List[dict],
        matched_gt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gödel t-norm constraint violation penalty on matched queries.

        Slices the flat 184-dim vector to extract [agentness, agent, action, loc]
        for the TNormConstraintLoss (which expects a 49-dim input).
        """
        if len(matched_pred) == 0:
            return pred_logits.sum() * 0.0

        flat_probs = pred_logits[matched_pred].sigmoid()  # [N_matched, 184]

        off = C.CLS_OFFSETS
        tnorm_input = torch.cat([
            flat_probs[:, 0:1],                                          # agentness (1)
            flat_probs[:, off["agent"]:off["agent"] + C.N_AGENTS],       # agent (10)
            flat_probs[:, off["action"]:off["action"] + C.N_ACTIONS],    # action (22)
            flat_probs[:, off["loc"]:off["loc"] + C.N_LOCS],            # loc (16)
        ], dim=1)  # [N_matched, 49]

        return self.tnorm(tnorm_input)

    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        gt_tubes: List[dict],
    ) -> tuple[torch.Tensor, dict]:
        """Compute all loss components for one set of predictions."""
        matched_pred, matched_gt = self.matcher(
            outputs["pred_boxes"],
            outputs["pred_logits"],
            gt_tubes,
        )

        l_cls = self._classification_loss(outputs["pred_logits"], matched_pred, gt_tubes, matched_gt)
        l_bbox, l_giou = self._box_losses(outputs["pred_boxes"], matched_pred, gt_tubes, matched_gt)
        l_tnorm = self._tnorm_loss(outputs["pred_logits"], matched_pred, gt_tubes, matched_gt)

        total = (
            C.LAMBDA_CLS  * l_cls
            + C.LAMBDA_BBOX * l_bbox
            + C.LAMBDA_GIOU * l_giou
            + l_tnorm
        )

        log = {
            "L_total":    float(total.detach().item()),
            "L_cls":      float(l_cls.detach().item()),
            "L_bbox":     float(l_bbox.detach().item()),
            "L_giou":     float(l_giou.detach().item()),
            "L_tnorm":    float(l_tnorm.detach().item()),
            "n_gt_tubes": float(len(gt_tubes)),
            "n_matched":  float(len(matched_pred)),
        }
        return total, log

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        frame_targets: List[dict | None],
    ) -> tuple[torch.Tensor, dict]:
        """Full loss with auxiliary layer losses."""
        gt_tubes = greedy_group_tubes(frame_targets, iou_thresh=C.TUBE_LINK_IOU)

        main_loss, main_log = self._compute_loss(outputs, gt_tubes)

        aux_loss = torch.tensor(0.0, device=main_loss.device)
        aux_outputs = outputs.get("aux_outputs", [])
        if aux_outputs:
            for aux_out in aux_outputs:
                aux_l, _ = self._compute_loss(aux_out, gt_tubes)
                aux_loss = aux_loss + aux_l
            aux_loss = aux_loss / len(aux_outputs)

        total = main_loss + aux_loss
        main_log["L_aux"] = float(aux_loss.detach().item())
        main_log["L_total"] = float(total.detach().item())
        return total, main_log


def load_constraint_children(anno_file: str) -> dict:
    """Read valid duplex/triplet child combinations from annotation JSON."""
    with open(anno_file) as f:
        data = json.load(f)
    return {
        "duplex_childs": data["duplex_childs"],
        "triplet_childs": data["triplet_childs"],
    }

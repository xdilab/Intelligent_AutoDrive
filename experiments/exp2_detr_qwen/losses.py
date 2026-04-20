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


def compute_class_alphas(anno_file: str) -> Dict[str, torch.Tensor]:
    from experiments.exp1b_road_r.losses import compute_class_alphas as _compute

    return _compute(anno_file)


def sigmoid_focal_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[torch.Tensor] = None,
) -> torch.Tensor:
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
    Best-effort tube grouping when the dataset does not expose annotation keys.
    This keeps the experiment moving while making the limitation explicit.
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


class SetCriterion(nn.Module):
    def __init__(
        self,
        matcher: HungarianMatcher,
        duplex_childs: list,
        triplet_childs: list,
        class_alphas: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.matcher = matcher
        self.class_alphas = class_alphas or {}
        self.tnorm = TNormConstraintLoss(
            duplex_childs=duplex_childs,
            triplet_childs=triplet_childs,
            n_agents=C.N_AGENTS,
            n_actions=C.N_ACTIONS,
            n_locs=C.N_LOCS,
            tnorm="godel",
            lam=C.LAMBDA_TNORM,
        )

    def _classification_loss(self, pred_logits: Dict[str, torch.Tensor], matched_pred: torch.Tensor, gt_tubes: List[dict], matched_gt: torch.Tensor) -> torch.Tensor:
        losses = []
        for head in C.HEAD_SIZES:
            if len(matched_pred) == 0:
                continue
            gt = torch.stack([gt_tubes[int(j)]["labels"][head] for j in matched_gt], dim=0).to(pred_logits[head].device)
            logits = pred_logits[head][matched_pred]
            losses.append(
                sigmoid_focal_with_logits(
                    logits,
                    gt,
                    gamma=C.FOCAL_GAMMA,
                    alpha=self.class_alphas.get(head),
                )
            )
        return torch.stack(losses).mean() if losses else pred_logits["agent"].sum() * 0.0

    def _box_losses(self, pred_boxes: torch.Tensor, matched_pred: torch.Tensor, gt_tubes: List[dict], matched_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if len(matched_pred) == 0:
            zero = pred_boxes.sum() * 0.0
            return zero, zero

        l1_losses = []
        giou_losses = []
        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes[matched_pred])
        pred_cxcywh = pred_boxes[matched_pred]
        for i, gt_idx in enumerate(matched_gt):
            tube = gt_tubes[int(gt_idx)]
            mask = tube["box_mask"]
            if not mask.any():
                continue
            gt_xyxy = tube["boxes"][mask]
            gt_cxcywh = box_xyxy_to_cxcywh(gt_xyxy)
            l1_losses.append(F.l1_loss(pred_cxcywh[i][mask], gt_cxcywh))
            giou = generalized_box_iou(pred_xyxy[i][mask], gt_xyxy)
            giou_losses.append((1.0 - giou.diag()).mean())

        if not l1_losses:
            zero = pred_boxes.sum() * 0.0
            return zero, zero
        return torch.stack(l1_losses).mean(), torch.stack(giou_losses).mean()

    def _noobj_loss(self, pred_logits: Dict[str, torch.Tensor], matched_pred: torch.Tensor) -> torch.Tensor:
        n_queries = pred_logits["agent"].shape[0]
        unmatched = torch.ones(n_queries, dtype=torch.bool, device=pred_logits["agent"].device)
        unmatched[matched_pred] = False
        if not unmatched.any():
            return pred_logits["agent"].sum() * 0.0

        logits = pred_logits["agent"][unmatched]
        noobj_logit = logits.max(dim=1, keepdim=True).values
        targets = torch.zeros_like(noobj_logit)
        return sigmoid_focal_with_logits(noobj_logit, targets, gamma=C.NOOBJ_GAMMA) * C.EOS_COEF

    def _tnorm_loss(self, pred_logits: Dict[str, torch.Tensor], matched_pred: torch.Tensor, gt_tubes: List[dict], matched_gt: torch.Tensor) -> torch.Tensor:
        if len(matched_pred) == 0:
            return pred_logits["agent"].sum() * 0.0

        probs = {
            "agent": pred_logits["agent"][matched_pred].sigmoid(),
            "action": pred_logits["action"][matched_pred].sigmoid(),
            "loc": pred_logits["loc"][matched_pred].sigmoid(),
            "duplex": pred_logits["duplex"][matched_pred].sigmoid(),
            "triplet": pred_logits["triplet"][matched_pred].sigmoid(),
        }
        flat = torch.cat([probs["agent"], probs["action"], probs["loc"], probs["duplex"], probs["triplet"]], dim=1)
        return self.tnorm(flat)

    def forward(self, outputs: Dict[str, torch.Tensor], frame_targets: List[dict | None]) -> tuple[torch.Tensor, dict]:
        gt_tubes = greedy_group_tubes(frame_targets, iou_thresh=C.TUBE_LINK_IOU)
        matched_pred, matched_gt = self.matcher(
            outputs["pred_boxes"],
            outputs["pred_logits"],
            gt_tubes,
        )

        l_cls = self._classification_loss(outputs["pred_logits"], matched_pred, gt_tubes, matched_gt)
        l_bbox, l_giou = self._box_losses(outputs["pred_boxes"], matched_pred, gt_tubes, matched_gt)
        l_tnorm = self._tnorm_loss(outputs["pred_logits"], matched_pred, gt_tubes, matched_gt)
        l_noobj = self._noobj_loss(outputs["pred_logits"], matched_pred)

        total = (
            C.LAMBDA_CLS * l_cls
            + C.LAMBDA_BBOX * l_bbox
            + C.LAMBDA_GIOU * l_giou
            + l_tnorm
            + l_noobj
        )

        return total, {
            "L_total": float(total.detach().item()),
            "L_cls": float(l_cls.detach().item()),
            "L_bbox": float(l_bbox.detach().item()),
            "L_giou": float(l_giou.detach().item()),
            "L_tnorm": float(l_tnorm.detach().item()),
            "L_noobj": float(l_noobj.detach().item()),
            "n_gt_tubes": float(len(gt_tubes)),
            "n_matched": float(len(matched_pred)),
        }


def load_constraint_children(anno_file: str) -> dict:
    with open(anno_file) as f:
        data = json.load(f)
    return {
        "duplex_childs": data["duplex_childs"],
        "triplet_childs": data["triplet_childs"],
    }


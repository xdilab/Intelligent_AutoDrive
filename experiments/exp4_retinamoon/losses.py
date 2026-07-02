"""Exp4 losses — focal-on-all over K tube predictions, plus deferred T-norm.

Differences from exp2f's losses.py:
  - exp2f uses DETR's Hungarian matcher on (cost_class + cost_bbox + cost_giou)
    because its boxes are LEARNED queries that need to land on GT positions.
  - exp4's boxes are FIXED (decoded anchors from the frozen RetinaNet). The match
    step is purely about ASSIGNING each GT tube a label-target; we do greedy
    IoU matching at the standard ROAD-Waymo IoU=0.5 threshold.
  - No L1 or GIoU loss in v1 (per plan §B — no box refinement).
  - T-norm loss is constructed but only invoked when epoch >= TNORM_START_EPOCH.

Public API:
  build_criterion(anno_file) -> FlatCriterion
  FlatCriterion(forward(outputs, gt_tubes, epoch) -> (total_loss, loss_dict))
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


# ---- Reuse helpers from the existing exp1b implementation. ---------------
_ROAD_REASON = "/data/repos/ROAD_Reason"
if _ROAD_REASON not in sys.path:
    sys.path.insert(0, _ROAD_REASON)
from tnorm_loss import TNormConstraintLoss                      # noqa: E402
from experiments.exp1b_road_r.losses import compute_class_alphas  # noqa: E402


# ---------------------------------------------------------------------------
# Focal loss (exp2f formulation, unchanged).
# ---------------------------------------------------------------------------
def sigmoid_focal_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Multi-label focal loss applied to every prediction × every class.

        FL(p, y) = -α_t · (1 - p_t)^γ · log(p_t)

    Matched predictions get GT label vector; unmatched get all-zeros target.
    Focal pushes unmatched predictions toward 0 explicitly (exp2f's fix for
    score-localization decorrelation, see findings/exp2f-flat-head).
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


def compute_flat_alphas(anno_file: str) -> torch.Tensor:
    """
    Build the 184-dim flat alpha vector: [agentness=0.25, agent(10), action(22),
    loc(16), duplex(49), triplet(86)] in flat layout matching CLS_OFFSETS.
    """
    head_alphas = compute_class_alphas(anno_file)
    parts = [torch.tensor([0.25])]  # agentness
    for head in ("agent", "action", "loc", "duplex", "triplet"):
        parts.append(head_alphas[head])
    return torch.cat(parts, dim=0)  # [184]


# ---------------------------------------------------------------------------
# Group per-frame GT boxes into tubes (greedy IoU linking, exp2f's approach).
# ---------------------------------------------------------------------------
_HEAD_NAMES = ("agent", "action", "loc", "duplex", "triplet")


def _box_iou_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between two box sets in xyxy. Returns [|a|, |b|]."""
    a = a.unsqueeze(1)                                                  # [|a|, 1, 4]
    b = b.unsqueeze(0)                                                  # [1, |b|, 4]
    x1 = torch.max(a[..., 0], b[..., 0]); y1 = torch.max(a[..., 1], b[..., 1])
    x2 = torch.min(a[..., 2], b[..., 2]); y2 = torch.min(a[..., 3], b[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    return inter / (area_a + area_b - inter).clamp(min=1e-7)


def greedy_group_tubes(frame_targets: List[Optional[dict]],
                       iou_thresh: float = 0.3) -> List[dict]:
    """
    Links per-frame GT annotations into multi-frame tubes (exp2f's recipe).

    Each tube is a dict:
        boxes:    [T, 4]   xyxy in pixel coords (zeros where invalid)
        box_mask: [T] bool (True on frames where the tube has a box)
        labels:   dict mapping head_name → per-head label tensor
                  (carried from the frame where the tube was first instantiated)
    """
    tubes: List[dict] = []
    T = len(frame_targets)

    for t, frame in enumerate(frame_targets):
        if frame is None:
            continue
        boxes = frame["boxes"]
        n = boxes.shape[0]
        assigned: set[int] = set()

        if tubes:
            active_boxes = []
            active_idx = []
            for idx, tube in enumerate(tubes):
                prev_frames = torch.where(tube["box_mask"])[0]
                if len(prev_frames) == 0:
                    continue
                active_boxes.append(tube["boxes"][int(prev_frames[-1])])
                active_idx.append(idx)
            if active_boxes:
                active_boxes_t = torch.stack(active_boxes)
                ious = _box_iou_matrix(active_boxes_t, boxes)
                for ai, tube_idx in enumerate(active_idx):
                    best_iou, best_j = ious[ai].max(dim=0)
                    j = int(best_j)
                    if float(best_iou) >= iou_thresh and j not in assigned:
                        assigned.add(j)
                        tubes[tube_idx]["boxes"][t] = boxes[j]
                        tubes[tube_idx]["box_mask"][t] = True

        for j in range(n):
            if j in assigned:
                continue
            labels = {h: frame[h][j].clone() for h in _HEAD_NAMES}
            tube = {
                "boxes":   torch.zeros(T, 4, dtype=torch.float32, device=boxes.device),
                "box_mask": torch.zeros(T, dtype=torch.bool,      device=boxes.device),
                "labels":  labels,
            }
            tube["boxes"][t]    = boxes[j]
            tube["box_mask"][t] = True
            tubes.append(tube)

    return tubes


# ---------------------------------------------------------------------------
# Tube matching: greedy IoU at ROAD-Waymo's eval threshold (0.5).
# ---------------------------------------------------------------------------
def _box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU between two single boxes a, b in xyxy. Returns scalar tensor."""
    x1 = torch.max(a[0], b[0]); y1 = torch.max(a[1], b[1])
    x2 = torch.min(a[2], b[2]); y2 = torch.min(a[3], b[3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (a[2] - a[0]).clamp(min=0) * (a[3] - a[1]).clamp(min=0)
    area_b = (b[2] - b[0]).clamp(min=0) * (b[3] - b[1]).clamp(min=0)
    union = area_a + area_b - inter
    return inter / union.clamp(min=1e-7)


def _tube_iou(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor,
              gt_mask: torch.Tensor) -> float:
    """
    Mean per-frame IoU over the frames where the GT tube is valid.

    pred_boxes:  [T, 4]   pixel xyxy
    gt_boxes:    [T, 4]   pixel xyxy (zeros where gt_mask is False)
    gt_mask:     [T] bool
    """
    if not gt_mask.any():
        return 0.0
    ious = []
    for t in range(gt_mask.shape[0]):
        if not gt_mask[t]:
            continue
        ious.append(_box_iou_xyxy(pred_boxes[t], gt_boxes[t]).item())
    return float(sum(ious) / max(len(ious), 1))


def match_tubes_greedy(pred_boxes: torch.Tensor, gt_tubes: List[dict],
                       iou_thresh: float = 0.5) -> Tuple[List[int], List[int]]:
    """
    Greedy one-to-one matching between predicted tubes and GT tubes by mean-IoU.

    For each GT tube (iterated in original order), pick the highest-IoU
    unmatched predicted tube with mean-IoU ≥ iou_thresh. If none qualifies,
    the GT tube is unmatched (carries no training signal for cls).

    Returns (matched_pred_idx, matched_gt_idx) — same length, parallel lists.
    """
    K = pred_boxes.shape[0]
    if not gt_tubes:
        return [], []

    device = pred_boxes.device
    gt_boxes_list = [t["boxes"].to(device)    for t in gt_tubes]
    gt_mask_list  = [t["box_mask"].to(device) for t in gt_tubes]

    matched_pred: List[int] = []
    matched_gt:   List[int] = []
    used = set()
    for gi, (gb, gm) in enumerate(zip(gt_boxes_list, gt_mask_list)):
        best_iou = 0.0
        best_pi  = -1
        for pi in range(K):
            if pi in used:
                continue
            iou = _tube_iou(pred_boxes[pi], gb, gm)
            if iou > best_iou:
                best_iou = iou
                best_pi  = pi
        if best_pi >= 0 and best_iou >= iou_thresh:
            matched_pred.append(best_pi)
            matched_gt.append(gi)
            used.add(best_pi)
    return matched_pred, matched_gt


# ---------------------------------------------------------------------------
# Pack a GT tube's per-head labels into the flat 184-dim target vector.
# ---------------------------------------------------------------------------
def pack_flat_target(gt_tube: dict, device: torch.device) -> torch.Tensor:
    """
    Flat layout = [agentness(1), agent(10), action(22), loc(16), duplex(49), triplet(86)]
    Matched tubes have agentness = 1.0 by construction.
    """
    target = torch.zeros(C.NUM_CLASSES, device=device)
    target[0] = 1.0
    labels = gt_tube["labels"]
    off = C.CLS_OFFSETS
    target[off["agent"]    : off["agent"]    + C.N_AGENTS  ] = labels["agent"   ].to(device)
    target[off["action"]   : off["action"]   + C.N_ACTIONS ] = labels["action"  ].to(device)
    target[off["loc"]      : off["loc"]      + C.N_LOCS    ] = labels["loc"     ].to(device)
    target[off["duplex"]   : off["duplex"]   + C.N_DUPLEXES] = labels["duplex"  ].to(device)
    target[off["triplet"]  : off["triplet"]  + C.N_TRIPLETS] = labels["triplet" ].to(device)
    return target


# ---------------------------------------------------------------------------
# Top-level criterion.
# ---------------------------------------------------------------------------
class FlatCriterion(nn.Module):
    """
    Computes focal-on-all loss + (curriculum) Godel T-norm.

    Inputs to forward:
      outputs:  dict from RetinaMoonFusion {"logits": [B,K,T,184],
                                            "boxes":  [B,K,T,4]}
      gt_tubes_per_clip: list (length B) of list-of-dicts; each tube dict has
                        {"boxes": [T,4], "box_mask": [T], "labels": dict-of-tensors}
      epoch:    current epoch index (T-norm enabled at >= TNORM_START_EPOCH)

    Output:
      total_loss: scalar tensor
      loss_dict:  dict for logging {"cls", "tnorm", "n_matched"}
    """

    def __init__(self, flat_alphas: torch.Tensor,
                 duplex_childs: list, triplet_childs: list):
        super().__init__()
        self.register_buffer("flat_alphas", flat_alphas)
        self.tnorm = TNormConstraintLoss(
            duplex_childs=duplex_childs,
            triplet_childs=triplet_childs,
            n_agents=C.N_AGENTS,
            n_actions=C.N_ACTIONS,
            n_locs=C.N_LOCS,
            tnorm=C.TNORM_TYPE,
            lam=C.LAMBDA_TNORM,
        )

    def _cls_loss_for_clip(self, logits: torch.Tensor,         # [K, T, 184]
                           pred_boxes: torch.Tensor,           # [K, T, 4]
                           gt_tubes: List[dict]) -> Tuple[torch.Tensor, List[int]]:
        """Per-clip focal-on-all. Match → pack target → focal. Mean per frame too.

        Loss is computed on the (K, T, 184) tensor by broadcasting the GT label
        across the GT's valid frames only — frames where the GT mask is False
        contribute the all-zeros target.
        """
        K, T, _ = logits.shape
        device = logits.device
        targets = torch.zeros(K, T, C.NUM_CLASSES, device=device)

        matched_pred, matched_gt = match_tubes_greedy(
            pred_boxes, gt_tubes, iou_thresh=0.5,
        )
        for pi, gi in zip(matched_pred, matched_gt):
            tube = gt_tubes[gi]
            flat = pack_flat_target(tube, device)                # [184]
            mask = tube["box_mask"].to(device)                   # [T]
            # On frames where the GT tube is valid, set target = flat label;
            # on invalid frames, leave as zeros (still trains the negatives).
            targets[pi][mask] = flat

        flat_logits = logits.reshape(-1, C.NUM_CLASSES)
        flat_target = targets.reshape(-1, C.NUM_CLASSES)
        cls = sigmoid_focal_with_logits(
            flat_logits, flat_target,
            gamma=C.FOCAL_GAMMA, alpha=self.flat_alphas,
        )
        return cls, matched_pred

    def _tnorm_for_clip(self, logits: torch.Tensor,           # [K, T, 184]
                        matched_pred: List[int]) -> torch.Tensor:
        """T-norm on matched predictions only (sigmoid → slice 49 dims)."""
        if not matched_pred:
            return logits.sum() * 0.0
        # Pull matched per-frame logits and apply only where the GT tube is real;
        # for simplicity, evaluate T-norm at every frame of matched tubes.
        flat_probs = logits[matched_pred].sigmoid().reshape(-1, C.NUM_CLASSES)
        off = C.CLS_OFFSETS
        tnorm_input = torch.cat([
            flat_probs[:, 0:1],
            flat_probs[:, off["agent"]  : off["agent"]  + C.N_AGENTS ],
            flat_probs[:, off["action"] : off["action"] + C.N_ACTIONS],
            flat_probs[:, off["loc"]    : off["loc"]    + C.N_LOCS   ],
        ], dim=1)  # [N_matched_frames, 49]
        return self.tnorm(tnorm_input)

    def forward(self, outputs: Dict[str, torch.Tensor],
                gt_tubes_per_clip: List[List[dict]],
                epoch: int = 0) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits = outputs["logits"]          # [B, K, T, 184]
        boxes  = outputs["boxes"]           # [B, K, T, 4]
        B = logits.shape[0]
        assert len(gt_tubes_per_clip) == B

        cls_total   = logits.new_zeros(())
        tnorm_total = logits.new_zeros(())
        n_matched   = 0
        for b in range(B):
            cls_b, matched = self._cls_loss_for_clip(
                logits[b], boxes[b], gt_tubes_per_clip[b],
            )
            cls_total = cls_total + cls_b
            n_matched += len(matched)
            if epoch >= C.TNORM_START_EPOCH:
                tnorm_total = tnorm_total + self._tnorm_for_clip(logits[b], matched)

        cls_total   = cls_total   / max(B, 1)
        tnorm_total = tnorm_total / max(B, 1)
        total = cls_total + tnorm_total
        return total, {
            "loss":    float(total.detach()),
            "cls":     float(cls_total.detach()),
            "tnorm":   float(tnorm_total.detach()),
            "matched": n_matched,
        }


def build_criterion(anno_file: str) -> FlatCriterion:
    """Factory: read constraint lists + class frequencies from anno_file, build FlatCriterion."""
    with open(anno_file) as f:
        data = json.load(f)

    # ROAD-R constraint lists live in the JSON at the top level.
    duplex_childs  = data.get("duplex_childs",  [])
    triplet_childs = data.get("triplet_childs", [])
    if not duplex_childs or not triplet_childs:
        raise RuntimeError(
            f"{anno_file} missing duplex_childs / triplet_childs — "
            "needed by T-norm constraint loss."
        )

    flat_alphas = compute_flat_alphas(anno_file)
    return FlatCriterion(
        flat_alphas=flat_alphas,
        duplex_childs=duplex_childs,
        triplet_childs=triplet_childs,
    )

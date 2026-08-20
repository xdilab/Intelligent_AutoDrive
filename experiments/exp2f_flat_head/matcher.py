from __future__ import annotations

from typing import Dict, List, Tuple

import torch

import config as C


# ---------------------------------------------------------------------------
# Box format conversion utilities (identical to exp2e)
# ---------------------------------------------------------------------------

def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [cx, cy, w, h] → [x1, y1, x2, y2], clamped to [0,1]."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = (cx - 0.5 * w).clamp(0.0, 1.0)
    y1 = (cy - 0.5 * h).clamp(0.0, 1.0)
    x2 = (cx + 0.5 * w).clamp(0.0, 1.0)
    y2 = (cy + 0.5 * h).clamp(0.0, 1.0)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [x1, y1, x2, y2] → [cx, cy, w, h]."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = (x2 - x1).clamp(min=0.0)
    h = (y2 - y1).clamp(min=0.0)
    return torch.stack([cx, cy, w, h], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between two sets of [x1,y1,x2,y2] boxes. Returns [N, M]."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """GIoU between two sets of [x1,y1,x2,y2] boxes. Returns [N, M]."""
    iou = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[..., 0] * wh[..., 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt2 = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb2 = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh2 = (rb2 - lt2).clamp(min=0)
    inter = wh2[..., 0] * wh2[..., 1]
    union = area1[:, None] + area2 - inter

    return iou - (area - union) / area.clamp(min=1e-6)


# ---------------------------------------------------------------------------
# Hungarian Matcher — adapted for flat 184-dim logits
# ---------------------------------------------------------------------------

class HungarianMatcher:
    """
    Bipartite matching between predicted tubes and GT tubes.

    Adapted from exp2e: pred_logits is now a flat [N_queries, 184] tensor
    instead of a dict of per-head tensors. Class cost slices agent and action
    from the flat vector using CLS_OFFSETS.
    """

    def __init__(self, cost_class: float, cost_bbox: float, cost_giou: float):
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    def __call__(self, pred_boxes, pred_logits, gt_tubes):
        return self.forward(pred_boxes, pred_logits, gt_tubes)

    def _class_cost(self, pred_logits: torch.Tensor, gt_tubes: List[dict]) -> torch.Tensor:
        """
        Class cost from flat 184-dim sigmoid predictions.

        Slices agent and action probabilities from the flat vector and computes
        negative mean probability of active GT classes — same semantics as exp2e
        but reading from flat positions instead of dict keys.
        """
        probs = pred_logits.sigmoid()  # [N_queries, 184]
        n_queries = probs.shape[0]
        n_gt = len(gt_tubes)
        device = probs.device
        cost = torch.zeros(n_queries, n_gt, device=device)

        off = C.CLS_OFFSETS
        agent_probs = probs[:, off["agent"]:off["agent"] + C.N_AGENTS]      # [N, 10]
        action_probs = probs[:, off["action"]:off["action"] + C.N_ACTIONS]  # [N, 22]

        for j, tube in enumerate(gt_tubes):
            labels = tube["labels"]

            agent_labels = labels["agent"].to(device)
            agent_pos = agent_labels > 0
            if agent_pos.any():
                cost[:, j] -= agent_probs[:, agent_pos].mean(dim=1)

            action_labels = labels["action"].to(device)
            action_pos = action_labels > 0
            if action_pos.any():
                cost[:, j] -= action_probs[:, action_pos].mean(dim=1)

        return cost

    def _tube_box_cost(self, pred_boxes: torch.Tensor, gt_tubes: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Box costs averaged over frames where GT is present. Identical to exp2e."""
        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        n_queries = pred_boxes.shape[0]
        n_gt = len(gt_tubes)
        cost_bbox = torch.zeros(n_queries, n_gt, device=pred_boxes.device)
        cost_giou = torch.zeros(n_queries, n_gt, device=pred_boxes.device)

        for j, tube in enumerate(gt_tubes):
            mask = tube["box_mask"].to(device=pred_boxes.device)
            if not mask.any():
                continue

            gt_boxes = tube["boxes"].to(device=pred_boxes.device)
            gt_cxcywh = box_xyxy_to_cxcywh(gt_boxes[mask])
            gt_xyxy = gt_boxes[mask]

            pred_sel_c = pred_boxes[:, mask, :]
            pred_sel_x = pred_xyxy[:, mask, :]

            l1 = (pred_sel_c - gt_cxcywh.unsqueeze(0)).abs().mean(dim=(1, 2))

            per_frame_giou = []
            for frame_idx in range(gt_xyxy.shape[0]):
                g = generalized_box_iou(
                    pred_sel_x[:, frame_idx, :],
                    gt_xyxy[frame_idx : frame_idx + 1]
                ).squeeze(1)
                per_frame_giou.append(g)
            giou = torch.stack(per_frame_giou, dim=1).mean(dim=1)

            cost_bbox[:, j] = l1
            cost_giou[:, j] = 1.0 - giou

        return cost_bbox, cost_giou

    def forward(
        self,
        pred_boxes: torch.Tensor,           # [N_queries, T, 4]
        pred_logits: torch.Tensor,           # [N_queries, 184] — flat tensor
        gt_tubes: List[dict],
    ):
        """Returns (matched_pred, matched_gt) index tensors."""
        if len(gt_tubes) == 0:
            empty = torch.empty(0, dtype=torch.int64, device=pred_boxes.device)
            return empty, empty

        cost_class = self._class_cost(pred_logits, gt_tubes)
        cost_bbox, cost_giou = self._tube_box_cost(pred_boxes, gt_tubes)

        total_cost = (
            self.cost_class * cost_class
            + self.cost_bbox * cost_bbox
            + self.cost_giou * cost_giou
        )

        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:
            raise ImportError("scipy is required for Hungarian matching") from exc

        rows, cols = linear_sum_assignment(total_cost.detach().cpu().numpy())

        return (
            torch.as_tensor(rows, dtype=torch.int64, device=pred_boxes.device),
            torch.as_tensor(cols, dtype=torch.int64, device=pred_boxes.device),
        )

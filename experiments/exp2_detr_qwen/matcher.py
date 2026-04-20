from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = (cx - 0.5 * w).clamp(0.0, 1.0)
    y1 = (cy - 0.5 * h).clamp(0.0, 1.0)
    x2 = (cx + 0.5 * w).clamp(0.0, 1.0)
    y2 = (cy + 0.5 * h).clamp(0.0, 1.0)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = (x2 - x1).clamp(min=0.0)
    h = (y2 - y1).clamp(min=0.0)
    return torch.stack([cx, cy, w, h], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    iou = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[..., 0] * wh[..., 1]

    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    lt2 = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb2 = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh2 = (rb2 - lt2).clamp(min=0)
    inter = wh2[..., 0] * wh2[..., 1]
    union = area1[:, None] + area2 - inter

    return iou - (area - union) / area.clamp(min=1e-6)


class HungarianMatcher:
    def __init__(self, cost_class: float, cost_bbox: float, cost_giou: float):
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    def _class_cost(self, pred_logits: Dict[str, torch.Tensor], gt_tubes: List[dict]) -> torch.Tensor:
        probs = {name: tensor.sigmoid() for name, tensor in pred_logits.items()}
        n_queries = next(iter(probs.values())).shape[0]
        n_gt = len(gt_tubes)
        cost = torch.zeros(n_queries, n_gt, device=next(iter(probs.values())).device)
        for j, tube in enumerate(gt_tubes):
            for head, labels in tube["labels"].items():
                pos = labels > 0
                if pos.any():
                    cost[:, j] -= probs[head][:, pos].mean(dim=1)
        return cost

    def _tube_box_cost(self, pred_boxes: torch.Tensor, gt_tubes: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        n_queries = pred_boxes.shape[0]
        n_gt = len(gt_tubes)
        cost_bbox = torch.zeros(n_queries, n_gt, device=pred_boxes.device)
        cost_giou = torch.zeros(n_queries, n_gt, device=pred_boxes.device)

        for j, tube in enumerate(gt_tubes):
            mask = tube["box_mask"]
            if not mask.any():
                continue
            gt_cxcywh = box_xyxy_to_cxcywh(tube["boxes"][mask])
            gt_xyxy = tube["boxes"][mask]
            pred_sel_c = pred_boxes[:, mask, :]
            pred_sel_x = pred_xyxy[:, mask, :]

            l1 = (pred_sel_c - gt_cxcywh.unsqueeze(0)).abs().mean(dim=(1, 2))

            per_frame_giou = []
            for frame_idx in range(gt_xyxy.shape[0]):
                g = generalized_box_iou(pred_sel_x[:, frame_idx, :], gt_xyxy[frame_idx : frame_idx + 1]).squeeze(1)
                per_frame_giou.append(g)
            giou = torch.stack(per_frame_giou, dim=1).mean(dim=1)

            cost_bbox[:, j] = l1
            cost_giou[:, j] = 1.0 - giou

        return cost_bbox, cost_giou

    def forward(self, pred_boxes: torch.Tensor, pred_logits: Dict[str, torch.Tensor], gt_tubes: List[dict]):
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


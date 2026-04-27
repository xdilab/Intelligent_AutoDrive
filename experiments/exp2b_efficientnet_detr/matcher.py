from __future__ import annotations

from typing import Dict, List, Tuple

import torch


# ---------------------------------------------------------------------------
# Box format conversion utilities
# ---------------------------------------------------------------------------

def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert [cx, cy, w, h] → [x1, y1, x2, y2], all in [0,1] normalised coords.

    Why two formats:
        DETR uses [cx,cy,w,h] for box regression (L1 loss on centre/size is more
        stable than L1 on corners, which can produce conflicting gradients if the
        box collapses). GIoU and IoU require [x1,y1,x2,y2] (corner format) to
        compute intersection area.  We convert back and forth as needed.

    Clamped to [0,1] because normalised coordinates outside this range would
    correspond to predictions off the image — meaningless for detection.
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = (cx - 0.5 * w).clamp(0.0, 1.0)
    y1 = (cy - 0.5 * h).clamp(0.0, 1.0)
    x2 = (cx + 0.5 * w).clamp(0.0, 1.0)
    y2 = (cy + 0.5 * h).clamp(0.0, 1.0)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [x1, y1, x2, y2] → [cx, cy, w, h]. Width/height clamped ≥ 0."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = (x2 - x1).clamp(min=0.0)
    h = (y2 - y1).clamp(min=0.0)
    return torch.stack([cx, cy, w, h], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Pairwise IoU between two sets of boxes in [x1,y1,x2,y2] format.

    Returns [N, M] where result[i,j] = IoU(boxes1[i], boxes2[j]).

    IoU = intersection / union
        = intersection / (area1 + area2 - intersection)

    Used in:
        - greedy_group_tubes: linking per-frame detections into tubes by IoU overlap
        - tube_iou in eval_baseline_compat: approximate video-level metric
        - HungarianMatcher._tube_box_cost: GIoU computation (calls box_iou internally)
    """
    # Area of each box: max(0,...) guards against malformed boxes (x2 < x1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    # Intersection: element-wise max of top-left corners, min of bottom-right corners
    # boxes1[:, None, :2] is [N, 1, 2]; boxes2[:, :2] is [M, 2] → broadcast to [N, M, 2]
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])   # intersection top-left
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])   # intersection bottom-right
    wh = (rb - lt).clamp(min=0)                           # [N, M, 2], clamped (no negative overlap)
    inter = wh[..., 0] * wh[..., 1]                      # [N, M]

    # Union = area1 + area2 - intersection (inclusion-exclusion)
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)  # clamp avoids div-by-zero for degenerate boxes


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Generalised IoU (GIoU) — extends IoU to handle non-overlapping boxes.

    GIoU = IoU - (enclosing_area - union) / enclosing_area
         = IoU - penalty

    Why GIoU instead of plain IoU:
        When two boxes don't overlap, IoU = 0 regardless of how far apart they are.
        This means the gradient of IoU w.r.t. box coordinates is zero — the loss
        provides no signal about which direction to move the predicted box.
        GIoU subtracts a penalty based on the enclosing rectangle, which is always
        non-zero when boxes are separated. This gives a gradient that pushes the
        predicted box toward the GT even with zero overlap.

    GIoU ranges from -1 (maximum misalignment) to +1 (perfect overlap).
    Loss = 1 - GIoU ranges from 0 to 2.
    """
    iou = box_iou(boxes1, boxes2)

    # Enclosing rectangle: min of top-left corners, max of bottom-right corners
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[..., 0] * wh[..., 1]  # area of enclosing rectangle [N, M]

    # Recompute union (needed for the GIoU penalty term)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt2 = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb2 = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh2 = (rb2 - lt2).clamp(min=0)
    inter = wh2[..., 0] * wh2[..., 1]
    union = area1[:, None] + area2 - inter

    return iou - (area - union) / area.clamp(min=1e-6)


# ---------------------------------------------------------------------------
# Hungarian Matcher
# ---------------------------------------------------------------------------

class HungarianMatcher:
    """
    Finds the optimal 1-to-1 assignment between predicted tubes and M GT tubes.

    Why bipartite matching (Hungarian algorithm):
        With anchor-based detectors (like 3D-RetinaNet), many anchors are assigned
        to each GT box based on IoU thresholds — this creates many positives per GT,
        requires NMS to remove duplicates, and the loss is summed over all positives.
        DETR's set prediction instead treats detection as a set assignment problem:
        find the single best query for each GT, and enforce a 1:1 correspondence.
        Duplicates are structurally impossible because each GT can only be matched once.

    Cost matrix C[i, j] = cost of assigning predicted tube i to GT tube j:
        C = cost_class * C_class + cost_bbox * C_bbox + cost_giou * C_giou

    The Hungarian algorithm finds rows/cols that minimise the total cost —
    a classic O(N³) assignment problem solved by scipy.optimize.linear_sum_assignment.

    Important: this is run on detached tensors (no gradient flows through matching).
    The matching is treated as a fixed assignment; gradients flow only through the
    loss computation on the matched pairs.
    """

    def __init__(self, cost_class: float, cost_bbox: float, cost_giou: float):
        self.cost_class = cost_class  # weight for semantic class cost
        self.cost_bbox = cost_bbox    # weight for L1 box cost
        self.cost_giou = cost_giou   # weight for GIoU box cost

    def __call__(self, pred_boxes, pred_logits, gt_tubes):
        return self.forward(pred_boxes, pred_logits, gt_tubes)

    def _class_cost(self, pred_logits: Dict[str, torch.Tensor], gt_tubes: List[dict]) -> torch.Tensor:
        """
        Cost based on predicted probability of each GT's true classes.

        For each GT tube j and each predicted tube i:
            C_class[i, j] = -mean(p_i[correct classes of j])

        Negative because lower cost = better match, and higher probability of
        the correct class = better match.

        Only agent and action heads contribute here (loc/duplex/triplet are
        derived combinations — matching on agent+action captures the semantics).
        All heads in pred_logits are iterated, but agentness is excluded because
        it is not a semantic class — it shouldn't influence which GT a query matches.
        """
        probs = {name: tensor.sigmoid() for name, tensor in pred_logits.items()}
        n_queries = next(iter(probs.values())).shape[0]
        n_gt = len(gt_tubes)
        device = next(iter(probs.values())).device
        cost = torch.zeros(n_queries, n_gt, device=device)

        for j, tube in enumerate(gt_tubes):
            for head, labels in tube["labels"].items():
                # labels is a multi-hot vector [C]; pos selects the active classes
                labels = labels.to(device=device)
                pos = labels > 0
                if pos.any():
                    # Mean probability assigned to the GT's active classes
                    # Shape: probs[head][:, pos] is [N_queries, n_active_classes]
                    cost[:, j] -= probs[head][:, pos].mean(dim=1)

        return cost  # [N_queries, N_gt]

    def _tube_box_cost(self, pred_boxes: torch.Tensor, gt_tubes: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Box costs averaged over frames where the GT tube is present (box_mask=True).

        GT tubes don't span all T frames — an agent may not appear in every frame.
        box_mask is a bool tensor [T] marking which frames have GT boxes.
        We average L1 and (1-GIoU) over the present frames only.

        pred_boxes: [N_queries, T, 4] in [cx,cy,w,h]
        """
        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)  # convert for GIoU
        n_queries = pred_boxes.shape[0]
        n_gt = len(gt_tubes)
        cost_bbox = torch.zeros(n_queries, n_gt, device=pred_boxes.device)
        cost_giou = torch.zeros(n_queries, n_gt, device=pred_boxes.device)

        for j, tube in enumerate(gt_tubes):
            mask = tube["box_mask"].to(device=pred_boxes.device)  # [T] bool
            if not mask.any():
                continue

            gt_boxes = tube["boxes"].to(device=pred_boxes.device)  # [T, 4] in [x1,y1,x2,y2]
            gt_cxcywh = box_xyxy_to_cxcywh(gt_boxes[mask])         # [n_present, 4]
            gt_xyxy = gt_boxes[mask]                                # [n_present, 4]

            # pred_boxes for the present frames: [N_queries, n_present, 4]
            pred_sel_c = pred_boxes[:, mask, :]
            pred_sel_x = pred_xyxy[:, mask, :]

            # L1 averaged over present frames and box coordinates
            l1 = (pred_sel_c - gt_cxcywh.unsqueeze(0)).abs().mean(dim=(1, 2))  # [N_queries]

            # GIoU per frame then averaged
            per_frame_giou = []
            for frame_idx in range(gt_xyxy.shape[0]):
                # generalized_box_iou returns [N_queries, 1]; squeeze to [N_queries]
                g = generalized_box_iou(
                    pred_sel_x[:, frame_idx, :],
                    gt_xyxy[frame_idx : frame_idx + 1]
                ).squeeze(1)
                per_frame_giou.append(g)
            giou = torch.stack(per_frame_giou, dim=1).mean(dim=1)  # [N_queries]

            cost_bbox[:, j] = l1
            cost_giou[:, j] = 1.0 - giou  # 1 - GIoU: 0 = perfect, 2 = worst

        return cost_bbox, cost_giou

    def forward(
        self,
        pred_boxes: torch.Tensor,           # [N_queries, T, 4]
        pred_logits: Dict[str, torch.Tensor],
        gt_tubes: List[dict],
    ):
        """
        Returns:
            matched_pred: [N_matched] — indices into the 100 queries
            matched_gt:   [N_matched] — indices into gt_tubes

        If gt_tubes is empty (no annotated agents in this clip), returns empty tensors.
        N_matched = number of GT tubes (≤ NUM_QUERIES, since each GT gets one query).
        """
        if len(gt_tubes) == 0:
            empty = torch.empty(0, dtype=torch.int64, device=pred_boxes.device)
            return empty, empty

        cost_class = self._class_cost(pred_logits, gt_tubes)
        cost_bbox, cost_giou = self._tube_box_cost(pred_boxes, gt_tubes)

        # Weighted sum — same weights as loss lambdas so that the matching
        # reflects the same priorities as the loss
        total_cost = (
            self.cost_class * cost_class
            + self.cost_bbox * cost_bbox
            + self.cost_giou * cost_giou
        )

        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:
            raise ImportError("scipy is required for Hungarian matching") from exc

        # linear_sum_assignment returns (row_indices, col_indices) that minimise
        # the total cost. Detach from autograd — matching is not differentiable.
        rows, cols = linear_sum_assignment(total_cost.detach().cpu().numpy())

        return (
            torch.as_tensor(rows, dtype=torch.int64, device=pred_boxes.device),
            torch.as_tensor(cols, dtype=torch.int64, device=pred_boxes.device),
        )

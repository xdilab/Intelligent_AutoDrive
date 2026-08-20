"""One-to-many (O2M) assigner for tubes, adapted from MS-DETR's Stage2Assigner.

Reference: /data/repos/Frozen-DETR/MS-DETR/models/matcher_o2m.py

For each GT tube, selects top-k predicted queries based on a combined
IoU + classification cost. Uses greedy IoU-threshold matching followed
by top-k selection per GT.

Key difference from the original: we operate on tubes (T-frame boxes)
instead of single-frame detections. Cost is computed as mean IoU across
visible frames + mean classification score.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from matcher import box_cxcywh_to_xyxy, box_iou


class Stage2AssignerTubes(nn.Module):
    """One-to-many assignment: each GT tube matches top-k predictions.

    Cost = coef_box * mean_IoU(pred, gt) + coef_cls * mean_cls_score

    For each GT: IoU-threshold filter, then select top-k by combined cost.
    Returns (matched_pred, matched_gt) tensors — many preds can map to same GT.
    """

    def __init__(self, k: int = 6, threshold: float = 0.4,
                 coef_box: float = 0.7, coef_cls: float = 0.3):
        super().__init__()
        self.k = k
        self.threshold = threshold
        self.coef_box = coef_box
        self.coef_cls = coef_cls

    @torch.no_grad()
    def forward(
        self,
        pred_boxes: torch.Tensor,
        pred_logits: torch.Tensor,
        gt_tubes: List[dict],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pred_boxes:  [N_q, T, 4] in cxcywh format, normalized
            pred_logits: [N_q, C] raw logits (184-dim flat vector)
            gt_tubes:    list of GT tube dicts with 'boxes' [T,4] xyxy and 'box_mask' [T]

        Returns:
            matched_pred: [M] indices into pred_boxes
            matched_gt:   [M] indices into gt_tubes
        """
        if len(gt_tubes) == 0:
            return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)

        device = pred_boxes.device
        N_q = pred_boxes.shape[0]
        T = pred_boxes.shape[1]
        N_gt = len(gt_tubes)

        # Compute mean IoU between each pred and each GT across visible frames
        # cost_iou: [N_gt, N_q]
        cost_iou = torch.zeros(N_gt, N_q, device=device)

        for gi, tube in enumerate(gt_tubes):
            mask = tube["box_mask"].to(device)
            if not mask.any():
                continue
            visible_frames = torch.where(mask)[0]
            gt_boxes_vis = tube["boxes"].to(device)[visible_frames]  # [V, 4] xyxy

            frame_ious = []
            for fi, t in enumerate(visible_frames):
                pred_t = box_cxcywh_to_xyxy(pred_boxes[:, int(t), :])  # [N_q, 4]
                iou_matrix = box_iou(pred_t, gt_boxes_vis[fi:fi+1])  # [N_q, 1]
                frame_ious.append(iou_matrix.squeeze(1))

            cost_iou[gi] = torch.stack(frame_ious).mean(dim=0)  # [N_q]

        # Compute classification cost: mean agentness score (slot 0)
        pred_probs = pred_logits.sigmoid()
        cost_cls = pred_probs[:, 0]  # [N_q] — agentness as objectness proxy

        # Combined cost: [N_gt, N_q]
        cost = self.coef_box * cost_iou + self.coef_cls * cost_cls.unsqueeze(0)

        # For each GT, find predictions above IoU threshold, then select top-k
        all_pred_inds = []
        all_gt_inds = []

        for gi in range(N_gt):
            above_thresh = cost_iou[gi] >= self.threshold
            if not above_thresh.any():
                # Fallback: take the single best match even if below threshold
                best_idx = cost[gi].argmax()
                all_pred_inds.append(best_idx.unsqueeze(0))
                all_gt_inds.append(torch.tensor([gi], device=device))
                continue

            valid_indices = torch.where(above_thresh)[0]
            valid_costs = cost[gi, valid_indices]

            # Select top-k
            topk = min(self.k, len(valid_indices))
            _, topk_local = valid_costs.topk(topk)
            selected = valid_indices[topk_local]

            all_pred_inds.append(selected)
            all_gt_inds.append(torch.full((topk,), gi, device=device, dtype=torch.long))

        matched_pred = torch.cat(all_pred_inds)
        matched_gt = torch.cat(all_gt_inds)

        return matched_pred, matched_gt

"""
FCOS-style GT token assignment for Exp1b dense detection.

For each spatial token at grid position (i, j) in a [H', W'] feature map,
its normalized center is:
    cx = (j + 0.5) / W'
    cy = (i + 0.5) / H'

Assignment rules:
  1. A token is POSITIVE (foreground) if its center falls strictly inside a GT box.
  2. If a token is inside multiple GT boxes, assign to the smallest (by area).
  3. Tokens not inside any GT box are NEGATIVE (background).

FCOS ltrb targets for a positive token assigned to GT box [x1, y1, x2, y2]:
    l = cx - x1     (left distance, normalized ≥ 0)
    t = cy - y1     (top distance, normalized ≥ 0)
    r = x2 - cx     (right distance, normalized ≥ 0)
    b = y2 - cy     (bottom distance, normalized ≥ 0)

Classification labels for positive tokens are copied from the assigned GT box.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


_LABEL_SIZES = {"agent": 10, "action": 22, "loc": 16, "duplex": 49, "triplet": 86}


def assign_tokens_to_gt(
    H_prime: int,
    W_prime: int,
    gt_boxes:  torch.Tensor,               # [n_gt, 4]  normalized (x1,y1,x2,y2)
    gt_labels: Dict[str, torch.Tensor],    # head → [n_gt, n_class] multi-hot
    device:    torch.device,
) -> dict:
    """
    Assign [H'*W'] spatial tokens to GT boxes (FCOS-style).

    Returns dict with keys:
        is_fg          [H'*W'] bool
        gt_idx         [H'*W'] int64   (-1 = background)
        ltrb_target    [H'*W', 4] float32  (zeros for background tokens)
        agent_target   [H'*W', 10]
        action_target  [H'*W', 22]
        loc_target     [H'*W', 16]
        duplex_target  [H'*W', 49]
        triplet_target [H'*W', 86]
    """
    n_tokens = H_prime * W_prime

    # Token center coordinates in normalized [0,1] space
    rows = torch.arange(H_prime, dtype=torch.float32, device=device)
    cols = torch.arange(W_prime, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
    cy = ((grid_y + 0.5) / H_prime).reshape(-1)   # [H'*W']
    cx = ((grid_x + 0.5) / W_prime).reshape(-1)   # [H'*W']

    gt_idx_out = torch.full((n_tokens,), -1, dtype=torch.long,    device=device)
    ltrb_out   = torch.zeros(n_tokens, 4,   dtype=torch.float32,  device=device)
    label_outs = {
        k: torch.zeros(n_tokens, sz, dtype=torch.float32, device=device)
        for k, sz in _LABEL_SIZES.items()
    }

    if gt_boxes is not None and gt_boxes.shape[0] > 0:
        gt_boxes = gt_boxes.to(device)
        x1 = gt_boxes[:, 0]   # [n_gt]
        y1 = gt_boxes[:, 1]
        x2 = gt_boxes[:, 2]
        y2 = gt_boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)   # [n_gt]

        # Broadcast: [H'*W', n_gt] — is each token inside each GT box?
        inside = (
            (cx.unsqueeze(1) > x1.unsqueeze(0)) &
            (cx.unsqueeze(1) < x2.unsqueeze(0)) &
            (cy.unsqueeze(1) > y1.unsqueeze(0)) &
            (cy.unsqueeze(1) < y2.unsqueeze(0))
        )   # [H'*W', n_gt] bool

        # For each token, pick the smallest GT box it falls inside
        inf_areas = areas.unsqueeze(0).expand(n_tokens, -1).clone()
        inf_areas[~inside] = float("inf")
        best_area, best_gt = inf_areas.min(dim=1)   # [H'*W']

        is_fg = best_area < float("inf")
        gt_idx_out[is_fg] = best_gt[is_fg]

        # FCOS ltrb targets
        fg_cx = cx[is_fg]
        fg_cy = cy[is_fg]
        fg_gt = gt_idx_out[is_fg]

        ltrb_out[is_fg, 0] = fg_cx - x1[fg_gt]
        ltrb_out[is_fg, 1] = fg_cy - y1[fg_gt]
        ltrb_out[is_fg, 2] = x2[fg_gt] - fg_cx
        ltrb_out[is_fg, 3] = y2[fg_gt] - fg_cy

        # Classification labels from assigned GT box
        for k, lbl in gt_labels.items():
            if k in label_outs and lbl is not None and lbl.shape[0] > 0:
                label_outs[k][is_fg] = lbl.to(device)[fg_gt]
    else:
        is_fg = torch.zeros(n_tokens, dtype=torch.bool, device=device)

    result = {
        "is_fg":       is_fg,
        "gt_idx":      gt_idx_out,
        "ltrb_target": ltrb_out,
    }
    for k, v in label_outs.items():
        result[f"{k}_target"] = v
    return result


def empty_assignment(n_tokens: int, device: torch.device) -> dict:
    """All-background assignment for a frame with no GT annotations."""
    result = {
        "is_fg":       torch.zeros(n_tokens, dtype=torch.bool,   device=device),
        "gt_idx":      torch.full((n_tokens,), -1, dtype=torch.long, device=device),
        "ltrb_target": torch.zeros(n_tokens, 4,  dtype=torch.float32, device=device),
    }
    for k, sz in _LABEL_SIZES.items():
        result[f"{k}_target"] = torch.zeros(n_tokens, sz, dtype=torch.float32, device=device)
    return result


def merge_assignments(per_frame: list) -> dict:
    """
    Concatenate a list of per-frame assignment dicts into one combined dict
    spanning all T*H'*W' tokens.

    Args:
        per_frame: list of T dicts, each from assign_tokens_to_gt or empty_assignment

    Returns:
        single dict with the same keys, all tensors concatenated along dim 0
    """
    keys = list(per_frame[0].keys())
    merged = {}
    for k in keys:
        tensors = [d[k] for d in per_frame]
        merged[k] = torch.cat(tensors, dim=0)
    return merged

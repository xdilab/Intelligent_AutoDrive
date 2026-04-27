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
    """
    Delegates to exp1b's implementation — reuses the per-class inverse-frequency
    alpha weights computed from the training set class distribution.

    Why inverse-frequency alpha:
        ROAD-Waymo is heavily imbalanced — Cars appear in nearly every clip, while
        Cyclists are rare. Without balancing, the model maximises accuracy by
        always predicting "Car" and ignoring rare classes.
        Alpha[c] ∝ 1 / freq[c] upweights rare classes in the focal loss.
    """
    from experiments.exp1b_road_r.losses import compute_class_alphas as _compute
    return _compute(anno_file)


def sigmoid_focal_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Focal Loss (Lin et al., RetinaNet, ICCV 2017) for multi-label binary classification.

    Standard cross-entropy treats easy and hard examples equally. In object detection,
    the vast majority of predictions are easy negatives (background), and their large
    number drowns out the signal from the few hard positives.

    Focal loss multiplies the CE loss by a modulating factor (1 - p_t)^γ:
        - When the model is confident and correct (p_t → 1): factor → 0, loss is small
        - When the model is wrong or uncertain (p_t → 0.5): factor → 1, loss is full CE
        - γ=2 means easy examples contribute 100× less than hard examples

    Combined with α (per-class weight for class imbalance):
        FL(p, y) = -α_t · (1 - p_t)^γ · log(p_t)

    where p_t = p if y=1 else (1-p), and α_t = α if y=1 else (1-α).

    Args:
        logits:  [N, C] raw logits (before sigmoid)
        targets: [N, C] binary targets in {0, 1}
        gamma:   focusing parameter (2.0 is standard)
        alpha:   [C] per-class positive weight, or None for uniform weighting
    """
    prob = logits.sigmoid()
    # binary_cross_entropy_with_logits is numerically stable (uses log-sum-exp trick)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    # p_t: probability of the "correct" class for each element
    pt = targets * prob + (1.0 - targets) * (1.0 - prob)
    loss = ce * (1.0 - pt).pow(gamma)  # modulate by (1 - p_t)^γ

    if alpha is not None:
        alpha = alpha.to(logits.device)
        # α_t: apply α to positives, (1-α) to negatives
        alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha)
        loss = alpha_t * loss

    return loss.mean()


def greedy_group_tubes(frame_targets: List[dict | None], iou_thresh: float = 0.3) -> List[dict]:
    """
    Links per-frame annotations into multi-frame tubes using greedy IoU matching.

    Why we need this:
        The dataset provides per-frame annotation dicts (boxes + labels per frame).
        The DETR decoder predicts per-clip tubes (one label set, T boxes). To compute
        the Hungarian matching loss, we need the GT in tube format too.
        Ideally the dataset would expose annotation keys (persistent agent IDs across
        frames), but ROADWaymoDataset doesn't expose them. This is a best-effort
        reconstruction.

    Algorithm (greedy, frame by frame):
        For each new frame t:
          1. For each existing tube, find the best-overlapping detection in frame t.
          2. If IoU ≥ thresh and the detection hasn't been claimed by another tube,
             extend that tube to frame t.
          3. Any unclaimed detections start new tubes.

    Limitations:
        - Greedy: suboptimal for crossing agents (IDs may swap at intersections).
        - Labels are taken from the FIRST frame each tube appears in and held fixed
          for the whole tube. If an agent's label changes (rare in ROAD-Waymo), this
          will be wrong.
        - IoU=0.3 is a lenient threshold — aggressive matching to minimise broken tubes.

    Returns: list of tube dicts, each with:
        'boxes':    [T, 4] in [x1,y1,x2,y2], zeros where agent not present
        'box_mask': [T] bool, True where agent is present
        'labels':   {head: [C] multi-hot} — tube-level label from first appearance
    """
    tubes: List[dict] = []

    for t, frame in enumerate(frame_targets):
        if frame is None:
            continue  # no annotations in this frame

        boxes = frame["boxes"]  # [N_detections, 4]
        n = boxes.shape[0]
        assigned = set()  # detection indices already matched to an existing tube

        if tubes:
            # Collect the last known box for each active tube
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
                # Compute IoU between active tube boxes and current frame detections
                active_boxes = torch.stack(active_boxes)  # [N_active, 4]
                ious = box_iou(active_boxes, boxes)       # [N_active, N_detections]

                for ai, tube_idx in enumerate(active_idx):
                    best_iou, best_j = ious[ai].max(dim=0)
                    if best_iou >= iou_thresh and int(best_j) not in assigned:
                        # Extend this tube to frame t
                        assigned.add(int(best_j))
                        tubes[tube_idx]["boxes"][t] = boxes[best_j]
                        tubes[tube_idx]["box_mask"][t] = True

        # Remaining detections become new tubes
        for j in range(n):
            if j in assigned:
                continue
            # Labels come from this first frame — held constant for the tube's lifetime
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
    """
    Computes all losses for one clip after Hungarian matching.

    Loss components and weights (from config.py):
        L_total = 2.0 * L_cls + 5.0 * L_bbox + 2.0 * L_giou + L_agentness + L_tnorm

    Weight rationale:
        - L_bbox gets the highest weight (5.0) because box quality is the main
          bottleneck for baseline-compatible f-mAP. L1 is in [0,1] normalised coords
          so its raw values are small; the high weight compensates.
        - L_cls and L_giou both at 2.0 — equally important for detection quality.
        - L_agentness at 1.0 — the matched/unmatched signal is already strong
          (focal loss gamma=2 makes correct high-confidence predictions contribute
          nearly zero loss).
        - L_tnorm at 1.0 — a soft regulariser; shouldn't dominate the main losses.
    """

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
            tnorm="godel",          # Gödel t-norm: violation = max(0, p_a + p_b - 1)
            lam=C.LAMBDA_TNORM,
        )

    def _classification_loss(
        self,
        pred_logits: Dict[str, torch.Tensor],
        matched_pred: torch.Tensor,
        gt_tubes: List[dict],
        matched_gt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Focal loss on all 5 classification heads for the N_matched queries.

        Applied ONLY to matched queries (those assigned to a GT tube). Unmatched
        queries don't get a semantic label — their label pressure comes solely from
        the agentness loss pushing them toward zero agentness.

        Average across all 5 heads (not sum) so that the loss magnitude doesn't
        scale with the number of heads — making the λ_cls=2.0 weight interpretable.
        """
        losses = []
        for head in C.HEAD_SIZES:
            if len(matched_pred) == 0:
                continue
            # Gather GT labels for matched GT tubes: [N_matched, C]
            gt = torch.stack(
                [gt_tubes[int(j)]["labels"][head] for j in matched_gt], dim=0
            ).to(pred_logits[head].device)
            logits = pred_logits[head][matched_pred]  # [N_matched, C]
            losses.append(
                sigmoid_focal_with_logits(
                    logits,
                    gt,
                    gamma=C.FOCAL_GAMMA,
                    alpha=self.class_alphas.get(head),  # per-class inverse-freq weight
                )
            )
        # If nothing was matched (empty clip), return a zero loss that still has a grad
        return torch.stack(losses).mean() if losses else pred_logits["agent"].sum() * 0.0

    def _box_losses(
        self,
        pred_boxes: torch.Tensor,
        matched_pred: torch.Tensor,
        gt_tubes: List[dict],
        matched_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        L1 and GIoU losses on box coordinates, averaged over frames where GT exists.

        Both losses are computed on matched queries only — supervising an unmatched
        query's box position is meaningless (there's no GT to compare against).

        Why L1 + GIoU together:
            L1 (on [cx,cy,w,h]) gives gradient for size and centre position.
            GIoU gives gradient for overlap — especially important when boxes
            don't yet overlap (L1 alone has conflicting gradients in that case:
            moving cx left could reduce L1 while moving the box further from GT).
            Together they converge faster and to better box quality.

        Frames with no GT box (box_mask=False) are excluded from the loss —
        there's no ground truth to compare against for those frames.
        """
        if len(matched_pred) == 0:
            zero = pred_boxes.sum() * 0.0
            return zero, zero

        l1_losses = []
        giou_losses = []
        # pred_boxes[matched_pred]: [N_matched, T, 4] in [cx,cy,w,h]
        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes[matched_pred])   # [N_matched, T, 4] in [x1,y1,x2,y2]
        pred_cxcywh = pred_boxes[matched_pred]

        for i, gt_idx in enumerate(matched_gt):
            tube = gt_tubes[int(gt_idx)]
            mask = tube["box_mask"].to(device=pred_boxes.device)  # [T] bool
            if not mask.any():
                continue
            gt_xyxy = tube["boxes"].to(device=pred_boxes.device)[mask]  # [n_present, 4]
            gt_cxcywh = box_xyxy_to_cxcywh(gt_xyxy)                    # [n_present, 4]

            # L1 on present frames only
            l1_losses.append(F.l1_loss(pred_cxcywh[i][mask], gt_cxcywh))

            # GIoU on present frames: generalized_box_iou returns [n_present, n_present]
            # diagonal gives each frame's GIoU between the matched pred and GT boxes
            giou = generalized_box_iou(pred_xyxy[i][mask], gt_xyxy)
            giou_losses.append((1.0 - giou.diag()).mean())  # 1 - GIoU as a loss

        if not l1_losses:
            zero = pred_boxes.sum() * 0.0
            return zero, zero
        return torch.stack(l1_losses).mean(), torch.stack(giou_losses).mean()

    def _agentness_loss(
        self,
        pred_logits: Dict[str, torch.Tensor],
        matched_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Focal loss on agentness for ALL queries (300 in exp2b).

        Matched queries → target 1.0; unmatched → target 0.0.
        Focal loss down-weights the ~290 easy negatives per clip.
        """
        logits = pred_logits["agentness"]           # [N_queries, 1]
        n_queries = logits.shape[0]
        targets = torch.zeros(n_queries, 1, device=logits.device)
        targets[matched_pred] = 1.0                 # matched queries are positive
        return sigmoid_focal_with_logits(logits, targets, gamma=C.FOCAL_GAMMA)

    def _tnorm_loss(
        self,
        pred_logits: Dict[str, torch.Tensor],
        matched_pred: torch.Tensor,
        gt_tubes: List[dict],
        matched_gt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gödel t-norm constraint violation penalty on matched queries.

        ROAD++ encodes logical constraints between label levels. For example:
            LarVeh (large vehicle) cannot be Xing (crossing the road on foot)
            → the duplex (LarVeh, Xing) is invalid.

        TNormConstraintLoss reads the valid duplex and triplet combinations from
        the annotation JSON and penalises predicted probability distributions that
        assign high probability to forbidden combinations.

        Gödel t-norm:
            For a conjunction (A AND B), the Gödel t-norm is: min(p_A, p_B).
            A violation of "NOT (A AND B)" = max(0, p_A + p_B - 1).
            This is the Łukasiewicz t-conorm applied to the negation.
            Gödel is sharper than Łukasiewicz — it only starts penalising when
            both probabilities exceed 0.5 (the model is simultaneously confident
            about both sides of a contradiction).

        Flat vector layout expected by TNormConstraintLoss:
            [agentness(1), agent(10), action(22), loc(16)]   → 49 values
            Offsets: 0=agentness, 1=agent start, 11=action start, 33=loc start
            duplex and triplet constraints are expressed in terms of agent/action/loc.

        Applied to matched queries only — unmatched queries have no semantic
        label assignment so their constraint violations are arbitrary.
        """
        if len(matched_pred) == 0:
            return pred_logits["agent"].sum() * 0.0

        agentness_probs = pred_logits["agentness"][matched_pred].sigmoid()  # [N_matched, 1]
        agent_probs     = pred_logits["agent"][matched_pred].sigmoid()      # [N_matched, 10]
        action_probs    = pred_logits["action"][matched_pred].sigmoid()     # [N_matched, 22]
        loc_probs       = pred_logits["loc"][matched_pred].sigmoid()        # [N_matched, 16]

        # Concatenate in the exact order expected by TNormConstraintLoss
        flat = torch.cat([agentness_probs, agent_probs, action_probs, loc_probs], dim=1)  # [N_matched, 49]
        return self.tnorm(flat)

    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        gt_tubes: List[dict],
    ) -> tuple[torch.Tensor, dict]:
        """Compute all loss components for one set of predictions against GT tubes.

        Used for both the main (final layer) loss and auxiliary (earlier layer) losses.
        """
        matched_pred, matched_gt = self.matcher(
            outputs["pred_boxes"],
            outputs["pred_logits"],
            gt_tubes,
        )

        l_cls = self._classification_loss(outputs["pred_logits"], matched_pred, gt_tubes, matched_gt)
        l_bbox, l_giou = self._box_losses(outputs["pred_boxes"], matched_pred, gt_tubes, matched_gt)
        l_tnorm = self._tnorm_loss(outputs["pred_logits"], matched_pred, gt_tubes, matched_gt)
        l_agentness = self._agentness_loss(outputs["pred_logits"], matched_pred)

        total = (
            C.LAMBDA_CLS   * l_cls
            + C.LAMBDA_BBOX * l_bbox
            + C.LAMBDA_GIOU * l_giou
            + l_tnorm
            + l_agentness
        )

        log = {
            "L_total":    float(total.detach().item()),
            "L_cls":      float(l_cls.detach().item()),
            "L_bbox":     float(l_bbox.detach().item()),
            "L_giou":     float(l_giou.detach().item()),
            "L_tnorm":    float(l_tnorm.detach().item()),
            "L_agentness": float(l_agentness.detach().item()),
            "n_gt_tubes": float(len(gt_tubes)),
            "n_matched":  float(len(matched_pred)),
        }
        return total, log

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        frame_targets: List[dict | None],
    ) -> tuple[torch.Tensor, dict]:
        """
        Full loss computation for one clip with auxiliary layer losses.

        Steps:
            1. Group per-frame annotations into GT tubes (greedy IoU linking)
            2. Compute main loss (final decoder layer)
            3. Compute auxiliary losses (earlier decoder layers), averaged
            4. Total = main + aux

        Returns (total_loss, log_dict) where log_dict has per-component floats
        for the training progress print.
        """
        gt_tubes = greedy_group_tubes(frame_targets, iou_thresh=C.TUBE_LINK_IOU)

        # Main loss (final decoder layer)
        main_loss, main_log = self._compute_loss(outputs, gt_tubes)

        # Auxiliary losses (earlier decoder layers)
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
    """
    Reads the valid duplex and triplet child combinations from the annotation JSON.

    The ROAD++ annotation file contains:
        duplex_childs:  list of valid (agent, action) index pairs
        triplet_childs: list of valid (agent, action, loc) index triples

    TNormConstraintLoss uses these to build the set of forbidden combinations
    (all pairs/triples NOT in these lists).
    """
    with open(anno_file) as f:
        data = json.load(f)
    return {
        "duplex_childs": data["duplex_childs"],
        "triplet_childs": data["triplet_childs"],
    }

"""Exp2g losses: flat 184-dim + softmax agent + encoder loss + O2M loss.

Builds on exp2f's SetCriterion with three additions:

1. **Softmax agent head** (USE_SOFTMAX_AGENT): Agent is single-label (10 classes),
   so we use softmax CE instead of sigmoid focal loss on agent dims [1:11].
   Matched queries get their GT agent class; unmatched get no agent loss
   (suppressed via agentness=0 in focal loss).

2. **Encoder loss** (two_stage): Full C-dim focal loss on encoder proposals
   (binary targets à la reference bin_targets) + L1/GIoU on matched encoder
   boxes. Uses same Hungarian matching and loss functions as decoder.

3. **O2M loss** (USE_MS_DETR): One-to-many matching via Stage2AssignerTubes,
   applied to the O2M decoder branch. Same loss terms as O2O but with k=6
   matches per GT.
"""

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

    FL(p, y) = -α_t · (1 - p_t)^γ · log(p_t)
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
    """Links per-frame annotations into multi-frame tubes using greedy IoU matching."""
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


def _get_agent_class_idx(gt_tube: dict) -> int:
    """Get the single agent class index from a GT tube's one-hot agent label."""
    agent_vec = gt_tube["labels"]["agent"]
    idx = int(agent_vec.argmax())
    return idx


class SetCriterion(nn.Module):
    """
    Computes all losses for one clip after Hungarian matching.

    Loss components:
        L_total = L_o2o + L_aux + L_enc + L_o2m

    Where L_o2o = 2.0 * L_cls + 5.0 * L_bbox + 2.0 * L_giou + L_tnorm

    With USE_SOFTMAX_AGENT: agent dims [1:11] use softmax CE on matched queries,
    remaining 174 dims use sigmoid focal loss on all queries.
    """

    def __init__(
        self,
        matcher: HungarianMatcher,
        duplex_childs: list,
        triplet_childs: list,
        flat_alphas: Optional[torch.Tensor] = None,
        o2m_matcher=None,
    ):
        super().__init__()
        self.matcher = matcher
        self.o2m_matcher = o2m_matcher
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
        Flat classification loss on ALL queries × 184 dims.

        When USE_SOFTMAX_AGENT:
          - Agent dims [1:11]: softmax CE on matched queries only
          - Other 174 dims: sigmoid focal loss on all queries
        Otherwise:
          - Sigmoid focal loss on all 184 dims (exp2f behavior)
        """
        N = pred_logits.shape[0]
        device = pred_logits.device
        targets = torch.zeros(N, C.NUM_CLASSES, device=device)

        for pi, gi in zip(matched_pred, matched_gt):
            targets[int(pi)] = _pack_flat_target(gt_tubes[int(gi)], device)

        if not getattr(C, "USE_SOFTMAX_AGENT", False):
            # exp2f behavior: sigmoid focal on all 184 dims
            alpha = self.flat_alphas if self.flat_alphas.numel() > 0 else None
            return sigmoid_focal_with_logits(
                pred_logits, targets, gamma=C.FOCAL_GAMMA, alpha=alpha,
            )

        # --- Softmax agent + sigmoid rest ---
        off = C.CLS_OFFSETS
        agent_start = off["agent"]      # 1
        agent_end = agent_start + C.N_AGENTS  # 11

        # 1) Sigmoid focal loss on non-agent dims (all queries)
        #    Indices: [0] (agentness) + [11:184] (action, loc, duplex, triplet)
        non_agent_idx = list(range(0, agent_start)) + list(range(agent_end, C.NUM_CLASSES))
        non_agent_logits = pred_logits[:, non_agent_idx]
        non_agent_targets = targets[:, non_agent_idx]
        alpha = self.flat_alphas if self.flat_alphas.numel() > 0 else None
        if alpha is not None:
            non_agent_alpha = alpha[non_agent_idx]
        else:
            non_agent_alpha = None
        l_focal = sigmoid_focal_with_logits(
            non_agent_logits, non_agent_targets, gamma=C.FOCAL_GAMMA,
            alpha=non_agent_alpha,
        )

        # 2) Softmax CE on agent dims (matched queries only)
        if len(matched_pred) == 0:
            return l_focal

        agent_logits_matched = pred_logits[matched_pred, agent_start:agent_end]  # [M, 10]
        agent_targets = torch.tensor(
            [_get_agent_class_idx(gt_tubes[int(gi)]) for gi in matched_gt],
            device=device, dtype=torch.long,
        )  # [M]
        l_agent = F.cross_entropy(agent_logits_matched, agent_targets)

        # Weight agent loss to be comparable to focal loss magnitude
        # focal loss is mean over N*174 elements; agent CE is mean over M elements
        # Scale agent loss by (M / N) to account for matched-only computation
        agent_weight = len(matched_pred) / max(N, 1)
        return l_focal + agent_weight * l_agent

    def _box_losses(
        self,
        pred_boxes: torch.Tensor,
        matched_pred: torch.Tensor,
        gt_tubes: List[dict],
        matched_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """L1 and GIoU losses on matched query boxes."""
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
        """Gödel t-norm constraint violation penalty on matched queries."""
        if len(matched_pred) == 0:
            return pred_logits.sum() * 0.0

        # For softmax agent: use softmax probs for agent, sigmoid for rest
        if getattr(C, "USE_SOFTMAX_AGENT", False):
            off = C.CLS_OFFSETS
            agent_start = off["agent"]
            agent_end = agent_start + C.N_AGENTS
            matched_logits = pred_logits[matched_pred]
            agentness = matched_logits[:, 0:1].sigmoid()
            agent_probs = F.softmax(matched_logits[:, agent_start:agent_end], dim=-1)
            action_probs = matched_logits[:, off["action"]:off["action"] + C.N_ACTIONS].sigmoid()
            loc_probs = matched_logits[:, off["loc"]:off["loc"] + C.N_LOCS].sigmoid()
        else:
            flat_probs = pred_logits[matched_pred].sigmoid()
            off = C.CLS_OFFSETS
            agentness = flat_probs[:, 0:1]
            agent_probs = flat_probs[:, off["agent"]:off["agent"] + C.N_AGENTS]
            action_probs = flat_probs[:, off["action"]:off["action"] + C.N_ACTIONS]
            loc_probs = flat_probs[:, off["loc"]:off["loc"] + C.N_LOCS]

        tnorm_input = torch.cat([agentness, agent_probs, action_probs, loc_probs], dim=1)
        return self.tnorm(tnorm_input)

    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        gt_tubes: List[dict],
    ) -> tuple[torch.Tensor, dict]:
        """Compute O2O loss components for one set of predictions."""
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

    def _encoder_loss(
        self,
        enc_outputs: Dict[str, torch.Tensor],
        gt_tubes: List[dict],
    ) -> tuple[torch.Tensor, dict]:
        """Encoder proposal loss — matches reference Frozen-DETR/MS-DETR exactly.

        Uses the same class head as decoder (full C-dim, last clone), Hungarian
        matching, and standard focal + L1 + GIoU losses. GT labels are binarized
        (all set to class 0) following the reference's bin_targets pattern.
        """
        import copy
        enc_logits = enc_outputs["pred_logits"]  # [T, N_spatial, C]
        enc_boxes = enc_outputs["pred_boxes"]     # [T, N_spatial, 4] — sigmoid space

        device = enc_logits.device

        if len(gt_tubes) == 0:
            zero = enc_logits.sum() * 0.0
            return zero, {"L_enc_cls": 0.0, "L_enc_bbox": 0.0, "L_enc_giou": 0.0}

        # Mean-pool encoder outputs across T → single-frame pseudo-predictions
        mean_logits = enc_logits.mean(dim=0)  # [N_spatial, C]
        mean_boxes = enc_boxes.mean(dim=0)    # [N_spatial, 4]

        # Wrap as [N_spatial, 1, 4] for matcher (T=1 pseudo-tube)
        mean_boxes_tube = mean_boxes.unsqueeze(1)  # [N_spatial, 1, 4]

        # Create binary GT tubes: same structure, but all labels zeroed to class 0
        # Reference: bin_targets sets all labels to 0 (single-class detection)
        bin_tubes = []
        for tube in gt_tubes:
            bt = copy.deepcopy(tube)
            # Mean GT box → single-frame pseudo-tube in xyxy
            mask = tube["box_mask"].to(device)
            if mask.any():
                gt_xyxy = tube["boxes"].to(device)[mask]
                mean_box_xyxy = gt_xyxy.mean(dim=0, keepdim=True)  # [1, 4]
            else:
                mean_box_xyxy = torch.zeros(1, 4, device=device)
            bt["boxes"] = mean_box_xyxy
            bt["box_mask"] = torch.ones(1, dtype=torch.bool, device=device)
            # Zero all labels — binary "is there an object?" supervision
            bt["labels"] = {k: torch.zeros_like(v) for k, v in tube["labels"].items()}
            bin_tubes.append(bt)

        # Hungarian matching on mean-pooled predictions vs binary GT
        matched_pred, matched_gt = self.matcher(
            mean_boxes_tube, mean_logits, bin_tubes,
        )

        # Classification loss: focal on all proposals, targets are all-zero class vectors
        # except matched proposals which get class-0 = 1
        N = mean_logits.shape[0]
        targets = torch.zeros(N, C.NUM_CLASSES, device=device)
        for pi, gi in zip(matched_pred, matched_gt):
            # Binary target: only agentness (dim 0) = 1 for matched proposals
            targets[int(pi), 0] = 1.0

        alpha = self.flat_alphas if self.flat_alphas.numel() > 0 else None
        l_enc_cls = sigmoid_focal_with_logits(
            mean_logits, targets, gamma=C.FOCAL_GAMMA, alpha=alpha,
        )

        # Box losses on matched proposals
        if len(matched_pred) > 0:
            l_enc_bbox, l_enc_giou = self._box_losses(
                mean_boxes_tube, matched_pred, bin_tubes, matched_gt,
            )
        else:
            l_enc_bbox = enc_logits.sum() * 0.0
            l_enc_giou = enc_logits.sum() * 0.0

        total = (
            C.LAMBDA_ENC_CLS * l_enc_cls
            + C.LAMBDA_ENC_BBOX * l_enc_bbox
            + C.LAMBDA_ENC_GIOU * l_enc_giou
        )

        log = {
            "L_enc_cls":  float(l_enc_cls.detach().item()),
            "L_enc_bbox": float(l_enc_bbox.detach().item()),
            "L_enc_giou": float(l_enc_giou.detach().item()),
        }
        return total, log

    def _o2m_loss(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        gt_tubes: List[dict],
    ) -> tuple[torch.Tensor, dict]:
        """One-to-many loss using Stage2AssignerTubes matching.

        Same loss terms as O2O (cls + bbox + giou) but with k matches per GT.
        """
        if self.o2m_matcher is None or len(gt_tubes) == 0:
            zero = pred_logits.sum() * 0.0
            return zero, {"L_o2m_cls": 0.0, "L_o2m_bbox": 0.0, "L_o2m_giou": 0.0}

        matched_pred, matched_gt = self.o2m_matcher(pred_boxes, pred_logits, gt_tubes)

        l_cls = self._classification_loss(pred_logits, matched_pred, gt_tubes, matched_gt)
        l_bbox, l_giou = self._box_losses(pred_boxes, matched_pred, gt_tubes, matched_gt)

        total = (
            C.LAMBDA_O2M_CLS * l_cls
            + C.LAMBDA_O2M_BBOX * l_bbox
            + C.LAMBDA_O2M_GIOU * l_giou
        )

        log = {
            "L_o2m_cls":  float(l_cls.detach().item()),
            "L_o2m_bbox": float(l_bbox.detach().item()),
            "L_o2m_giou": float(l_giou.detach().item()),
        }
        return total, log

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        frame_targets: List[dict | None],
    ) -> tuple[torch.Tensor, dict]:
        """Full loss with auxiliary layer losses, encoder loss, and O2M loss."""
        gt_tubes = greedy_group_tubes(frame_targets, iou_thresh=C.TUBE_LINK_IOU)

        # --- O2O main loss ---
        main_loss, main_log = self._compute_loss(outputs, gt_tubes)

        # --- O2O auxiliary losses (per decoder layer) ---
        aux_loss = torch.tensor(0.0, device=main_loss.device)
        aux_o2m_loss = torch.tensor(0.0, device=main_loss.device)
        aux_outputs = outputs.get("aux_outputs", [])
        n_aux = 0
        if aux_outputs:
            for aux_out in aux_outputs:
                aux_l, _ = self._compute_loss(aux_out, gt_tubes)
                aux_loss = aux_loss + aux_l

                # O2M loss on auxiliary layer outputs
                if "pred_logits_o2m" in aux_out:
                    o2m_l, _ = self._o2m_loss(
                        aux_out["pred_logits_o2m"], aux_out["pred_boxes"], gt_tubes,
                    )
                    aux_o2m_loss = aux_o2m_loss + o2m_l
            n_aux = len(aux_outputs)
            aux_loss = aux_loss / n_aux
            if aux_o2m_loss > 0:
                aux_o2m_loss = aux_o2m_loss / n_aux

        # --- O2M main loss ---
        o2m_loss = torch.tensor(0.0, device=main_loss.device)
        o2m_log = {}
        if "o2m_pred_logits" in outputs:
            o2m_loss, o2m_log = self._o2m_loss(
                outputs["o2m_pred_logits"], outputs["pred_boxes"], gt_tubes,
            )

        # --- Encoder loss ---
        enc_loss = torch.tensor(0.0, device=main_loss.device)
        enc_log = {}
        if "enc_outputs" in outputs:
            enc_loss, enc_log = self._encoder_loss(outputs["enc_outputs"], gt_tubes)

        total = main_loss + aux_loss + o2m_loss + aux_o2m_loss + enc_loss
        main_log["L_aux"] = float(aux_loss.detach().item())
        main_log["L_o2m"] = float((o2m_loss + aux_o2m_loss).detach().item())
        main_log["L_enc"] = float(enc_loss.detach().item())
        main_log.update(o2m_log)
        main_log.update(enc_log)
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

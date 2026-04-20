"""
Experiment 1 model: Qwen2.5-VL ViT encoder + ROI-pool + tube attention + classification.

Architecture
------------
1. Qwen2.5-VL ViT (frozen in Exp 1; LoRA r=8 added in Phase 1b)
   - Processes T frames independently; outputs merged spatial tokens [T, H', W', D]
   - H' = H/28, W' = W/28 after 2× spatial merger; D = 1280 for 7B model
2. ROI-pool
   - For each GT box [x1, y1, x2, y2] (normalized), average ViT tokens in that region
   - Produces one D-dim feature per annotated agent
3. TubeLinkingModule
   - Single cross-frame multi-head self-attention across all agent features
   - Gives each agent feature context from the other frames
4. ClassificationHeads (5×)
   - agent:   Linear(D→10)  + sigmoid
   - action:  Linear(D→22)  + sigmoid
   - loc:     Linear(D→16)  + sigmoid
   - duplex:  Linear(D→49)  + sigmoid
   - triplet: Linear(D→86)  + sigmoid

LoRA note
---------
peft is not currently installed. To add LoRA to the ViT shallow layers:
    pip install peft
Then wrap self.vit.visual with get_peft_model() using LoraConfig targeting
the QV projection modules of the first ~8 ViT transformer blocks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


# ── ViT feature extractor ──────────────────────────────────────────────────────

class QwenViTExtractor(nn.Module):
    """
    Loads the Qwen2.5-VL vision encoder and extracts merged spatial tokens.

    Only model.visual is retained; the LLM, projector, and text tokenizer are
    dropped immediately after loading to avoid unnecessary VRAM usage.

    forward inputs:
        pixel_values:   [total_patches, C * patch_h * patch_w]  — from image_processor
        image_grid_thw: [T, 3]  each row = [1, H', W'] (one temporal slice per frame)

    forward output:
        features: [T, H', W', D]  — merged token spatial map, one entry per frame
    """

    def __init__(self, model_id: str, freeze: bool = True):
        super().__init__()

        from transformers import Qwen2_5_VLForConditionalGeneration
        full = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        self.visual = full.model.visual
        del full  # release LLM weights

        if freeze:
            for p in self.visual.parameters():
                p.requires_grad_(False)

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Returns list of T tensors each [H'_t, W'_t, D].

        image_grid_thw[i] = [t_i, h_i, w_i] are PRE-merger patch dims (confirmed
        from Qwen2_5_VLModel.get_image_features source which computes split sizes as
        `image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2`).
        Post-merger tokens per frame = t_i * h_i * w_i // spatial_merge_size².
        Spatial dims after merger: H'_i = h_i // m, W'_i = w_i // m where m=spatial_merge_size.
        """
        out = self.visual(pixel_values, image_grid_thw)
        merged = out.pooler_output  # [total_merged_tokens, D]

        m = self.visual.spatial_merge_size  # 2 for all Qwen2.5-VL variants

        frame_feats: List[torch.Tensor] = []
        offset = 0
        for row in image_grid_thw:
            t, h, w = int(row[0]), int(row[1]), int(row[2])
            h_m, w_m = h // m, w // m
            n = t * h_m * w_m
            chunk = merged[offset : offset + n]      # [t*h_m*w_m, D]
            chunk = chunk.view(t, h_m, w_m, -1)      # [t, H', W', D]
            # t == 1 for each individual frame; squeeze to [H', W', D]
            frame_feats.append(chunk.squeeze(0))
            offset += n

        return frame_feats  # list of T tensors [H'_t, W'_t, D]


# ── ROI pooling ────────────────────────────────────────────────────────────────

class ROIAveragePool(nn.Module):
    """
    Extracts per-agent features from a spatial token map using GT boxes.

    Given a normalized box [x1, y1, x2, y2] and a feature map of shape [H', W', D],
    we identify the token indices that overlap the box and average them.
    Boxes that map to zero tokens (e.g., very small objects) are represented by
    the nearest single token.
    """

    @staticmethod
    def _pool_one(feat_map: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        """feat_map: [H', W', D]; box: [4] normalized."""
        H, W, _ = feat_map.shape
        x1, y1, x2, y2 = box.tolist()

        col_lo = max(0, int(x1 * W))
        col_hi = min(W, max(col_lo + 1, round(x2 * W + 0.5)))
        row_lo = max(0, int(y1 * H))
        row_hi = min(H, max(row_lo + 1, round(y2 * H + 0.5)))

        region = feat_map[row_lo:row_hi, col_lo:col_hi, :]  # [rH, rW, D]
        return region.mean(dim=(0, 1))                       # [D]

    def forward(
        self,
        feat_map: torch.Tensor,
        boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feat_map: [H', W', D]
            boxes:    [n, 4] normalized boxes
        Returns:
            [n, D]
        """
        D = feat_map.shape[-1]
        if boxes.shape[0] == 0:
            return torch.zeros(0, D, device=feat_map.device, dtype=feat_map.dtype)

        return torch.stack([self._pool_one(feat_map, b) for b in boxes])


# ── Tube-linking module ────────────────────────────────────────────────────────

class TubeLinkingModule(nn.Module):
    """
    Cross-frame multi-head self-attention over all agent features in a clip.

    All agent features from all T frames are concatenated into a single sequence
    and attended jointly, giving each agent context from agents in neighboring
    frames. The input residual is added after the attention (pre-norm style).

    Args:
        d_model:  feature dimension (1280)
        n_heads:  attention heads (8)
        dropout:  attention dropout
    """

    def __init__(self, d_model: int = 1280, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm  = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, frame_feats: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            frame_feats: list of T tensors each [n_agents_t, D]
        Returns:
            [total_agents, D]  with cross-frame context
        """
        seq = torch.cat(frame_feats, dim=0)          # [total_agents, D]
        seq_f = seq.float()                           # attn in fp32
        x = self.norm(seq_f)
        attn_out, _ = self.attn(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        x = self.norm2(seq_f + attn_out.squeeze(0))
        return x.to(seq.dtype)


# ── Classification heads ───────────────────────────────────────────────────────

class ClassificationHeads(nn.Module):
    """
    Five sigmoid multi-label classification heads.

    Input:  [N, D]  per-agent tube features
    Output: dict of five [N, n_class] sigmoid tensors
    """

    def __init__(
        self,
        d_in:       int = 1280,
        n_agents:   int = 10,
        n_actions:  int = 22,
        n_locs:     int = 16,
        n_duplexes: int = 49,
        n_triplets: int = 86,
    ):
        super().__init__()
        self.agent   = nn.Linear(d_in, n_agents)
        self.action  = nn.Linear(d_in, n_actions)
        self.loc     = nn.Linear(d_in, n_locs)
        self.duplex  = nn.Linear(d_in, n_duplexes)
        self.triplet = nn.Linear(d_in, n_triplets)

    def forward(self, x: torch.Tensor) -> dict:
        x = x.float()
        return {
            "agent":   torch.sigmoid(self.agent(x)),
            "action":  torch.sigmoid(self.action(x)),
            "loc":     torch.sigmoid(self.loc(x)),
            "duplex":  torch.sigmoid(self.duplex(x)),
            "triplet": torch.sigmoid(self.triplet(x)),
        }


# ── Full model ─────────────────────────────────────────────────────────────────

class QwenROADModel(nn.Module):
    """
    Experiment 1 full model.

    forward(pixel_values, image_grid_thw, frame_boxes_list)
    ↓
    QwenViTExtractor  →  [T, H', W', D]
    ↓ (per frame)
    ROIAveragePool    →  list of T × [n_t, D]
    ↓
    TubeLinkingModule →  [total_agents, D]
    ↓
    ClassificationHeads → dict of [total_agents, n_class]

    Returns None when no annotated boxes are present in the clip.
    """

    def __init__(
        self,
        model_id:  str,
        d_model:   int  = 1280,
        freeze_vit: bool = True,
        n_agents:  int  = 10,
        n_actions: int  = 22,
        n_locs:    int  = 16,
        n_duplexes: int = 49,
        n_triplets: int = 86,
        tube_heads: int = 8,
    ):
        super().__init__()
        self.vit       = QwenViTExtractor(model_id, freeze=freeze_vit)
        self.roi_pool  = ROIAveragePool()
        self.tube_link = TubeLinkingModule(d_model, tube_heads)
        self.heads     = ClassificationHeads(
            d_model, n_agents, n_actions, n_locs, n_duplexes, n_triplets
        )

    def forward(
        self,
        pixel_values:     torch.Tensor,
        image_grid_thw:   torch.Tensor,
        frame_boxes_list: List[Optional[torch.Tensor]],
    ) -> Optional[dict]:
        """
        Args:
            pixel_values:       [total_patches, C*pH*pW]  from Qwen image_processor
            image_grid_thw:     [T, 3]  one row per frame: [1, H', W']
            frame_boxes_list:   list of T tensors [n_t, 4] or None (no GT boxes)

        Returns:
            preds: dict with keys 'agent','action','loc','duplex','triplet'
                   each [total_annotated_agents, n_class], or None if no boxes.
        """
        # 1. ViT feature extraction → list of T tensors [H'_t, W'_t, D]
        vit_feats = self.vit(pixel_values, image_grid_thw)

        # 2. ROI-pool per frame
        frame_feats: List[torch.Tensor] = []
        for t, (feat_map, boxes) in enumerate(zip(vit_feats, frame_boxes_list)):
            if boxes is None or boxes.shape[0] == 0:
                continue
            boxes_dev = boxes.to(feat_map.device)
            roi_f = self.roi_pool(feat_map, boxes_dev)  # [n_t, D]
            if roi_f.shape[0] > 0:
                frame_feats.append(roi_f)

        if not frame_feats:
            return None

        # 3. Tube-linking (cross-frame attention)
        tube_feats = self.tube_link(frame_feats)        # [total_agents, D]

        # 4. Classification
        return self.heads(tube_feats)

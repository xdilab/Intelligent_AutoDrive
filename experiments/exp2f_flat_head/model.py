"""Exp2f — R50-FPN + Frozen-DETR Encoder + CLIP ViT-L/14 + Flat 184-dim Head.

Same architecture as exp2e except the classification head:
  - exp2e: 6 separate heads (agentness:1, agent:10, action:22, loc:16, duplex:49, triplet:86)
  - exp2f: 1 flat nn.Linear(256, 184) matching the 3D-RetinaNet baseline

The flat vector layout: [agentness(1), agent(10), action(22), loc(16), duplex(49), triplet(86)]
Sigmoid activation on all 184 dims, focal loss on ALL queries (not just matched).

This fixes the score-localization decorrelation found in exp2e: unmatched queries now
receive explicit target=0 supervision on all 184 dims, not just agentness.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.transforms import functional as TF

from backbone import ResNet50Backbone
from fpn import FPN
from clip_encoder import CLIPVisualEncoder, PatchTokenProjection, get_clip_preprocess
from deformable_decoder import DeformableDETREncoder, DeformableDETRDecoder


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _resize_shorter_side(img, short_side: int, max_size: int):
    """Resize PIL image so shorter side = short_side, respecting max_size."""
    w, h = img.size
    min_side = float(min(w, h))
    max_side = float(max(w, h))
    scale = short_side / min_side
    if max_side * scale > max_size:
        scale = max_size / max_side
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    return TF.resize(img, [new_h, new_w], antialias=True)


class FrozenDETRModel(nn.Module):
    """Exp2f: R50-FPN + Frozen-DETR encoder + CLIP ViT-L/14 + flat 184-dim head."""

    def __init__(
        self,
        clip_model: str = "ViT-L/14@336px",
        clip_cache_dir: str | None = None,
        clip_cls_dim: int = 768,
        clip_patch_dim: int = 1024,
        clip_grid: int = 24,
        d_model: int = 256,
        num_classes: int = 184,
        clip_len: int = 8,
        num_queries: int = 300,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        nhead: int = 8,
        dim_ffn: int = 2048,
        dropout: float = 0.0,
        n_deform_points: int = 4,
        encoder_n_levels: int = 4,
        decoder_n_levels: int = 3,
        fpn_in_channels: list[int] | None = None,
        val_short_side: int = 800,
        val_max_size: int = 1333,
    ):
        super().__init__()
        if fpn_in_channels is None:
            fpn_in_channels = [512, 1024, 2048]

        self.clip_len = clip_len
        self.d_model = d_model
        self.clip_grid = clip_grid
        self.val_short_side = val_short_side
        self.val_max_size = val_max_size
        self.num_classes = num_classes

        # ---- Spatial branch: ResNet-50 + FPN ----
        self.backbone = ResNet50Backbone(pretrained=True)
        self.fpn = FPN(in_channels=fpn_in_channels, out_channels=d_model)

        # ---- Semantic branch: Frozen CLIP ViT-L/14 ----
        self.clip_encoder = CLIPVisualEncoder(
            model_name=clip_model, cache_dir=clip_cache_dir,
        )
        self.patch_proj = PatchTokenProjection(
            clip_dim=clip_patch_dim, d_model=d_model,
        )
        self._clip_preprocess = get_clip_preprocess(
            self.clip_encoder.input_resolution
        )

        # ---- Deformable encoder (4 scales: P3+P4+P5+CLIP) ----
        self.encoder = DeformableDETREncoder(
            d_model=d_model,
            n_heads=nhead,
            num_layers=num_encoder_layers,
            d_ffn=dim_ffn,
            dropout=dropout,
            n_levels=encoder_n_levels,
            n_points=n_deform_points,
        )

        # ---- Deformable decoder (3 scales: P3+P4+P5, with CLS injection) ----
        self.decoder = DeformableDETRDecoder(
            d_model=d_model,
            n_heads=nhead,
            num_layers=num_decoder_layers,
            dim_ffn=dim_ffn,
            dropout=dropout,
            num_queries=num_queries,
            clip_len=clip_len,
            n_levels=decoder_n_levels,
            n_points=n_deform_points,
            clip_cls_dim=clip_cls_dim,
        )

        # ---- Per-layer classification heads (paper: _get_clones) ----
        _cls_head = nn.Linear(d_model, num_classes)
        # Focal loss bias init (RetinaNet, Lin et al. 2017)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)  # -4.595
        nn.init.constant_(_cls_head.bias, bias_value)
        self.cls_heads = nn.ModuleList(
            [copy.deepcopy(_cls_head) for _ in range(num_decoder_layers)]
        )

        # Register ImageNet normalization as buffers
        self.register_buffer(
            "_img_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_img_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        )

    # ---- Parameter groups for optimizer ----

    def backbone_parameters(self) -> list[nn.Parameter]:
        """R50 trainable layers (layer2/3/4)."""
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def encoder_decoder_parameters(self) -> list[nn.Parameter]:
        """FPN + patch_proj + encoder + decoder, excluding deformable params."""
        deform_set = set(id(p) for p in self.deformable_parameters())
        params = []
        for module in [self.fpn, self.patch_proj, self.encoder, self.decoder]:
            for p in module.parameters():
                if p.requires_grad and id(p) not in deform_set:
                    params.append(p)
        return params

    def head_parameters(self) -> list[nn.Parameter]:
        """Per-layer classification heads."""
        return list(self.cls_heads.parameters())

    def deformable_parameters(self) -> list[nn.Parameter]:
        """reference_points and sampling_offsets — get 0.1x LR per paper."""
        params = []
        for name, p in self.named_parameters():
            if p.requires_grad and ("reference_points" in name or "sampling_offsets" in name):
                params.append(p)
        return params

    def _normalize_for_backbone(self, frames: torch.Tensor) -> torch.Tensor:
        return (frames - self._img_mean.to(frames.dtype)) / self._img_std.to(frames.dtype)

    def _preprocess_for_clip(self, pil_frames: list, device: torch.device) -> torch.Tensor:
        clip_tensors = [self._clip_preprocess(img) for img in pil_frames]
        return torch.stack(clip_tensors).to(device)

    def forward(
        self,
        pil_frames: list,
    ) -> Dict[str, torch.Tensor]:
        device = next(self.backbone.parameters()).device
        dtype = next(self.backbone.parameters()).dtype

        # ---- Step 1: R50 per-frame features ----
        if not self.training:
            pil_frames = [
                _resize_shorter_side(img, self.val_short_side, self.val_max_size)
                for img in pil_frames
            ]

        frame_tensors = [TF.to_tensor(img) for img in pil_frames]
        frames_batch = torch.stack(frame_tensors).to(device=device, dtype=dtype)
        frames_batch = self._normalize_for_backbone(frames_batch)
        cnn_features = self.backbone(frames_batch)

        # ---- Step 2: FPN ----
        fpn_features = self.fpn(cnn_features)

        # ---- Step 3: CLIP features ----
        clip_frames = self._preprocess_for_clip(pil_frames, device)
        cls_token, patch_tokens = self.clip_encoder(clip_frames)
        vlm_spatial = self.patch_proj(patch_tokens)

        # ---- Step 4: Encoder — 4 scales (P3, P4, P5, CLIP patches) ----
        T = len(pil_frames)
        encoder_input = fpn_features + [vlm_spatial]

        spatial_shapes_list = []
        for feat in encoder_input:
            _, _, H_i, W_i = feat.shape
            spatial_shapes_list.append([H_i, W_i])
        spatial_shapes_enc = torch.tensor(spatial_shapes_list, device=device, dtype=torch.long)

        memory, _, level_start_index = self.encoder(encoder_input, spatial_shapes_enc)

        # ---- Step 4b: Strip VLM tokens from memory ----
        clip_patch_count = self.clip_grid * self.clip_grid
        decoder_memory = memory[:, :-clip_patch_count, :]
        decoder_shapes = spatial_shapes_enc[:3]

        decoder_features = []
        offset = 0
        for H, W in decoder_shapes:
            H, W = int(H), int(W)
            feat = decoder_memory[:, offset:offset + H * W, :]
            feat = feat.transpose(1, 2).view(T, self.d_model, H, W)
            decoder_features.append(feat)
            offset += H * W

        # ---- Step 5: Decoder with per-layer CLS injection ----
        query_feats, pred_boxes, aux_outputs = self.decoder(
            decoder_features, decoder_shapes, clip_len=T,
            image_query=cls_token,
        )

        # ---- Step 6: Per-layer classification heads ----
        pred_logits = self.cls_heads[-1](query_feats.float())  # [N_queries, 184]

        aux_list = []
        for layer_idx, (aux_feats, aux_boxes) in enumerate(aux_outputs):
            aux_logits = self.cls_heads[layer_idx](aux_feats.float())
            aux_list.append({
                "pred_boxes": aux_boxes,
                "pred_logits": aux_logits,
                "query_feats": aux_feats,
                "T": T,
            })

        return {
            "pred_boxes": pred_boxes,
            "pred_logits": pred_logits,  # [N_queries, 184] tensor, not dict
            "query_feats": query_feats,
            "T": T,
            "aux_outputs": aux_list,
        }

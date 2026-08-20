"""Exp2e — R50-FPN + Frozen-DETR Encoder + CLIP ViT-L/14.

Faithful replication of Frozen-DETR (Fu et al., NeurIPS 2024) spatial config:
  - ResNet-50 backbone (FrozenBatchNorm, layer2/3/4 trainable)
  - Variable aspect ratio input: train multi-scale [480..800], val 800×max1333
  - FPN channels [512, 1024, 2048] → 256
  - CLIP ViT-L/14@336px (frozen)
  - Deformable encoder 6 layers, 4 scales (P3+P4+P5+CLIP)
  - Deformable decoder 6 layers, 3 scales (P3+P4+P5, CLS injection)
  - 5 ROAD classification heads + agentness + Gödel t-norm loss

Only difference from paper: 5 ROAD multi-label heads instead of 91 COCO classes.

Data flow for one 8-frame clip:
    1. Augmentation outputs variable-size PIL frames (e.g. 800×1200)
    2. R50 extracts per-frame features: C3 [512, H/8, W/8], C4 [1024, H/16, W/16], C5 [2048, H/32, W/32]
    3. FPN merges to P3/P4/P5 at 256 channels
    4. CLIP ViT-L/14 extracts CLS [768] + patch tokens [1024, 24, 24]
    5. Encoder: 6 layers deformable self-attention over 4 scales
    6. Strip CLIP tokens → 3-scale decoder input
    7. Decoder: 300 queries with per-layer CLS injection + temporal attention
    8. Per-frame box prediction + 5 classification heads
"""

from __future__ import annotations

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


class ClassificationHeads(nn.Module):
    """Five independent linear classifiers for ROAD++ compositional labels."""

    def __init__(self, d_model: int, head_sizes: Dict[str, int]):
        super().__init__()
        self.heads = nn.ModuleDict(
            {name: nn.Linear(d_model, size) for name, size in head_sizes.items()}
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: head(x.float()) for name, head in self.heads.items()}


def _resize_shorter_side(img, short_side: int, max_size: int):
    """Resize PIL image so shorter side = short_side, respecting max_size.

    Matches DETR/DINO val resize: Frozen-DETR/MS-DETR/datasets/coco.py:156
    """
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
    """Exp2e: R50-FPN + Frozen-DETR encoder + CLIP ViT-L/14.

    Key difference from exp2c/exp2d: variable aspect ratio input (no fixed square resize).
    During training, augmentation pipeline outputs variable-size frames.
    During val, frames are resized to short_side=800, max=1333.
    """

    def __init__(
        self,
        clip_model: str = "ViT-L/14@336px",
        clip_cache_dir: str | None = None,
        clip_cls_dim: int = 768,
        clip_patch_dim: int = 1024,
        clip_grid: int = 24,
        d_model: int = 256,
        head_sizes: Dict[str, int] | None = None,
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
        if head_sizes is None:
            head_sizes = {"agent": 10, "action": 22, "loc": 16, "duplex": 49, "triplet": 86}
        if fpn_in_channels is None:
            fpn_in_channels = [512, 1024, 2048]

        self.clip_len = clip_len
        self.d_model = d_model
        self.clip_grid = clip_grid
        self.val_short_side = val_short_side
        self.val_max_size = val_max_size

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

        # ---- Detection heads ----
        self.cls_heads = ClassificationHeads(d_model, head_sizes)
        self.agentness_head = nn.Linear(d_model, 1)

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
        """FPN + patch_proj + encoder + decoder."""
        params = list(self.fpn.parameters())
        params += list(self.patch_proj.parameters())
        params += list(self.encoder.parameters())
        params += list(self.decoder.parameters())
        return params

    def head_parameters(self) -> list[nn.Parameter]:
        """Classification heads + agentness."""
        return list(self.cls_heads.parameters()) + list(self.agentness_head.parameters())

    def _normalize_for_backbone(self, frames: torch.Tensor) -> torch.Tensor:
        """ImageNet-normalize a batch of [0,1] RGB frames."""
        return (frames - self._img_mean.to(frames.dtype)) / self._img_std.to(frames.dtype)

    def _preprocess_for_clip(self, pil_frames: list, device: torch.device) -> torch.Tensor:
        """Apply CLIP preprocessing to PIL frames → [T, 3, 336, 336]."""
        clip_tensors = [self._clip_preprocess(img) for img in pil_frames]
        return torch.stack(clip_tensors).to(device)

    def forward(
        self,
        pil_frames: list,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for one clip.

        Args:
            pil_frames: List of T PIL images (variable size — augmentation or val resize
                        already applied by caller during training; val resize applied here).

        Returns:
            dict with pred_boxes, pred_logits, query_feats, T, aux_outputs.
        """
        device = next(self.backbone.parameters()).device
        dtype = next(self.backbone.parameters()).dtype

        # ---- Step 1: R50 per-frame features ----
        # During training, augmentation already resized frames to variable AR.
        # During eval, resize here to val resolution.
        if not self.training:
            pil_frames = [
                _resize_shorter_side(img, self.val_short_side, self.val_max_size)
                for img in pil_frames
            ]

        frame_tensors = [TF.to_tensor(img) for img in pil_frames]  # each [3, H, W]
        frames_batch = torch.stack(frame_tensors).to(device=device, dtype=dtype)
        frames_batch = self._normalize_for_backbone(frames_batch)
        cnn_features = self.backbone(frames_batch)  # dict: C3, C4, C5

        # ---- Step 2: FPN ----
        fpn_features = self.fpn(cnn_features)  # [P3, P4, P5], each [T, 256, H_i, W_i]

        # ---- Step 3: CLIP features ----
        clip_frames = self._preprocess_for_clip(pil_frames, device)  # [T, 3, 336, 336]
        cls_token, patch_tokens = self.clip_encoder(clip_frames)
        vlm_spatial = self.patch_proj(patch_tokens)  # [T, 256, 24, 24]

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
        clip_patch_count = self.clip_grid * self.clip_grid  # 576
        decoder_memory = memory[:, :-clip_patch_count, :]
        decoder_shapes = spatial_shapes_enc[:3]  # P3, P4, P5 only

        # Un-flatten memory back to list of [T, 256, H, W] for decoder interface
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

        # ---- Step 6: Classification heads ----
        pred_logits = self.cls_heads(query_feats)
        pred_logits["agentness"] = self.agentness_head(query_feats.float())

        aux_list = []
        for aux_feats, aux_boxes in aux_outputs:
            aux_logits = self.cls_heads(aux_feats)
            aux_logits["agentness"] = self.agentness_head(aux_feats.float())
            aux_list.append({
                "pred_boxes": aux_boxes,
                "pred_logits": aux_logits,
                "query_feats": aux_feats,
                "T": T,
            })

        return {
            "pred_boxes": pred_boxes,
            "pred_logits": pred_logits,
            "query_feats": query_feats,
            "T": T,
            "aux_outputs": aux_list,
        }

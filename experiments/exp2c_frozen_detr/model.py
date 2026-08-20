"""Exp2c — EfficientNet-FPN + Frozen-DETR Encoder + CLIP ViT-L/14.

Full model: EfficientNet-B0 (spatial) + frozen CLIP ViT-L/14 (semantic)
→ FPN → deformable encoder (4 scales) → strip CLIP tokens → decoder
(3 scales, per-layer CLS injection) → per-frame boxes + classification heads.

Architecture follows Frozen-DETR (Fu et al., NeurIPS 2024):
  - CLIP integration: DINO-coco/models/dino/dino.py:195-257
  - Encoder + VLM stripping: deformable_transformer.py:278-331
  - Per-layer CLS injection: deformable_transformer.py:604-782

Data flow for one 8-frame clip:
    1. EfficientNet extracts per-frame multi-scale features (C3/C4/C5)
    2. FPN merges to P3/P4/P5 at 256 channels
    3. CLIP ViT-L/14 extracts CLS token [768] + patch tokens [1024, 24, 24]
    4. Patch tokens projected to 256-dim → 4th encoder scale
    5. Encoder: 6 layers of deformable self-attention over 4 scales
    6. Strip CLIP tokens from encoder memory → 3-scale decoder input
    7. Decoder: 300 queries with per-layer CLS injection + temporal attention
    8. Per-frame box prediction: [300, 8, 4]
    9. Classification: 5 heads + agentness

Three optimizer param groups (CLIP is fully frozen):
    - backbone_parameters():          EfficientNet trainable blocks (LR=2e-5)
    - encoder_decoder_parameters():   FPN + patch_proj + encoder + decoder (LR=1e-4)
    - head_parameters():              cls heads + agentness (LR=1e-4)
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torchvision.transforms import functional as TF

from backbone import EfficientNetBackbone
from fpn import FPN
from clip_encoder import CLIPVisualEncoder, PatchTokenProjection, get_clip_preprocess
from deformable_decoder import DeformableDETREncoder, DeformableDETRDecoder


# ImageNet normalization constants for EfficientNet
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


class FrozenDETRModel(nn.Module):
    """Full Exp2c model: EfficientNet-FPN + Frozen-DETR encoder + CLIP ViT-L/14.

    Args:
        clip_model:         CLIP model name or path (default: "ViT-L/14@336px").
        clip_cache_dir:     Cache directory for CLIP checkpoint.
        clip_cls_dim:       CLIP CLS token dimension (768).
        clip_patch_dim:     CLIP patch token dimension (1024).
        clip_grid:          CLIP patch grid size (24).
        d_model:            Shared feature dimension (256).
        head_sizes:         Dict mapping head name → number of classes.
        clip_len:           Frames per clip (8).
        num_queries:        Object queries (300).
        num_encoder_layers: Deformable encoder layers (6).
        num_decoder_layers: Deformable decoder layers (6).
        nhead:              Attention heads (8).
        dim_ffn:            FFN hidden dim (1024).
        dropout:            Dropout rate (0.1).
        n_deform_points:    Sampling points per head per level (4).
        encoder_n_levels:   Encoder feature levels (4: P3+P4+P5+CLIP).
        decoder_n_levels:   Decoder feature levels (3: P3+P4+P5).
        backbone_name:      EfficientNet variant ("efficientnet_b0").
        backbone_freeze_blocks: Freeze first N blocks (2).
        fpn_in_channels:    EfficientNet stage output channels ([40, 112, 320]).
    """

    def __init__(
        self,
        clip_model: str = "ViT-L/14@336px",
        clip_cache_dir: str | None = None,
        clip_cls_dim: int = 768,
        clip_patch_dim: int = 1024,
        clip_grid: int = 24,
        d_model: int = 256,
        head_sizes: Dict[str, int] = None,
        clip_len: int = 8,
        num_queries: int = 300,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        nhead: int = 8,
        dim_ffn: int = 1024,
        dropout: float = 0.1,
        n_deform_points: int = 4,
        encoder_n_levels: int = 4,
        decoder_n_levels: int = 3,
        backbone_name: str = "efficientnet_b0",
        backbone_freeze_blocks: int = 2,
        fpn_in_channels: list[int] = None,
    ):
        super().__init__()
        if head_sizes is None:
            head_sizes = {"agent": 10, "action": 22, "loc": 16, "duplex": 49, "triplet": 86}
        if fpn_in_channels is None:
            fpn_in_channels = [40, 112, 320]

        self.clip_len = clip_len
        self.d_model = d_model
        self.clip_grid = clip_grid

        # ---- Spatial branch: EfficientNet + FPN ----
        self.backbone = EfficientNetBackbone(
            backbone_name, freeze_blocks=backbone_freeze_blocks, pretrained=True,
        )
        self.fpn = FPN(in_channels=fpn_in_channels, out_channels=d_model)

        # ---- Semantic branch: Frozen CLIP ViT-L/14 ----
        self.clip_encoder = CLIPVisualEncoder(
            model_name=clip_model, cache_dir=clip_cache_dir,
        )
        self.patch_proj = PatchTokenProjection(
            clip_dim=clip_patch_dim, d_model=d_model,
        )

        # CLIP preprocessing (stored for forward pass)
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
        """EfficientNet trainable blocks."""
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def encoder_decoder_parameters(self) -> list[nn.Parameter]:
        """FPN + patch_proj + encoder + decoder (all trainable non-backbone, non-head)."""
        params = list(self.fpn.parameters())
        params += list(self.patch_proj.parameters())
        params += list(self.encoder.parameters())
        params += list(self.decoder.parameters())
        return params

    def head_parameters(self) -> list[nn.Parameter]:
        """Classification heads + agentness."""
        return list(self.cls_heads.parameters()) + list(self.agentness_head.parameters())

    def _normalize_for_efficientnet(self, frames: torch.Tensor) -> torch.Tensor:
        """ImageNet-normalize a batch of [0,1] RGB frames."""
        return (frames - self._img_mean.to(frames.dtype)) / self._img_std.to(frames.dtype)

    def _preprocess_for_clip(self, pil_frames: list, device: torch.device) -> torch.Tensor:
        """Apply CLIP preprocessing to PIL frames.

        Args:
            pil_frames: List of T PIL images.
            device: Target device.

        Returns:
            [T, 3, 336, 336] CLIP-normalized tensor.
        """
        clip_tensors = [self._clip_preprocess(img) for img in pil_frames]
        return torch.stack(clip_tensors).to(device)

    def forward(
        self,
        pil_frames: list,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for one clip.

        Args:
            pil_frames: List of T PIL images.

        Returns:
            dict with:
                pred_boxes:  [N_queries, T, 4] — sigmoid [cx,cy,w,h] in [0,1]
                pred_logits: {head: [N_queries, C]} — raw logits
                query_feats: [N_queries, d_model]
                T: int
                aux_outputs: list of per-layer dicts
        """
        device = next(self.backbone.parameters()).device
        dtype = next(self.backbone.parameters()).dtype

        # ---- Step 1: EfficientNet per-frame features ----
        frame_tensors = []
        for img in pil_frames:
            t = TF.to_tensor(img)  # [3, H, W] in [0, 1]
            t = TF.resize(t, [448, 448], antialias=True)
            frame_tensors.append(t)

        frames_batch = torch.stack(frame_tensors).to(device=device, dtype=dtype)
        frames_batch = self._normalize_for_efficientnet(frames_batch)
        cnn_features = self.backbone(frames_batch)  # dict: C3, C4, C5

        # ---- Step 2: FPN ----
        fpn_features = self.fpn(cnn_features)  # [P3, P4, P5], each [T, 256, H_i, W_i]

        # ---- Step 3: CLIP features ----
        clip_frames = self._preprocess_for_clip(pil_frames, device)  # [T, 3, 336, 336]
        cls_token, patch_tokens = self.clip_encoder(clip_frames)
        # cls_token: [T, 768], patch_tokens: [T, 1024, 24, 24]
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
        # memory: [T, sum(H_i*W_i), 256] where sum = 3136+784+196+576 = 4692

        # ---- Step 4b: Strip VLM tokens from memory ----
        # Ref: deformable_transformer.py:326-331
        clip_patch_count = self.clip_grid * self.clip_grid  # 576
        decoder_memory = memory[:, :-clip_patch_count, :]   # [T, 4116, 256]
        decoder_shapes = spatial_shapes_enc[:3]              # P3, P4, P5 only

        # Un-flatten memory back to list of [T, 256, H, W] for decoder interface
        decoder_features = []
        offset = 0
        for H, W in decoder_shapes:
            H, W = int(H), int(W)
            feat = decoder_memory[:, offset:offset + H * W, :]  # [T, H*W, 256]
            feat = feat.transpose(1, 2).view(T, self.d_model, H, W)
            decoder_features.append(feat)
            offset += H * W

        # ---- Step 5: Decoder with per-layer CLS injection ----
        query_feats, pred_boxes, aux_outputs = self.decoder(
            decoder_features, decoder_shapes, clip_len=T,
            image_query=cls_token,  # [T, 768]
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

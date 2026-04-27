"""Exp2b — EfficientNet-FPN + Deformable DETR with VLM Semantic Fusion.

Full model: EfficientNet-B0 (spatial) + Qwen ViT (semantic) → FPN → gated
fusion → Deformable DETR decoder → per-frame boxes + classification heads.

Data flow for one 8-frame clip:
    1. EfficientNet extracts per-frame multi-scale features (C3/C4/C5)
    2. FPN merges to P3/P4/P5 at 256 channels
    3. Qwen ViT extracts semantic maps, gated-fused into each FPN level
    4. Features stacked across T frames per level
    5. Deformable DETR decoder: 300 queries attend to multi-scale tokens
    6. Per-frame box prediction: [300, 8, 4]
    7. Classification: 5 heads + agentness

Four optimizer param groups:
    - lora_parameters():     Qwen ViT LoRA adapters (LR=5e-5)
    - backbone_parameters(): EfficientNet trainable blocks (LR=2e-5)
    - decoder_parameters():  FPN + VLM fusion + deformable decoder (LR=1e-4)
    - head_parameters():     cls heads + agentness (LR=1e-4)
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torchvision.transforms import functional as TF

from backbone import EfficientNetBackbone
from fpn import FPN
from vlm_fusion import VLMSemanticFusion
from deformable_decoder import DeformableDETRDecoder


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


class EfficientNetFPNDETRModel(nn.Module):
    """Full Exp2b model combining spatial CNN + semantic VLM + deformable DETR.

    Args:
        model_id:   Qwen model ID for VLM semantic extraction.
        vit_dim:    Qwen ViT hidden dim (3584).
        d_model:    Shared feature dimension (256).
        head_sizes: Dict mapping head name → number of classes.
        clip_len:   Frames per clip (8).
        num_queries: Object queries (300).
        num_decoder_layers: Deformable decoder layers (6).
        nhead:      Attention heads (8).
        dim_ffn:    FFN hidden dim (1024).
        dropout:    Dropout rate (0.1).
        n_deform_points: Sampling points per head per level (4).
        backbone_name: EfficientNet variant ("efficientnet_b0").
        backbone_freeze_blocks: Freeze first N blocks (2).
        fpn_in_channels: EfficientNet stage output channels ([40, 112, 320]).
        gate_init: VLM fusion gate initial value (0.1).
        lora_r, lora_alpha, lora_dropout, lora_target_modules, lora_n_layers:
            LoRA configuration for Qwen ViT.
    """

    def __init__(
        self,
        model_id: str,
        vit_dim: int = 3584,
        d_model: int = 256,
        head_sizes: Dict[str, int] = None,
        clip_len: int = 8,
        num_queries: int = 300,
        num_decoder_layers: int = 6,
        nhead: int = 8,
        dim_ffn: int = 1024,
        dropout: float = 0.1,
        n_deform_points: int = 4,
        backbone_name: str = "efficientnet_b0",
        backbone_freeze_blocks: int = 2,
        fpn_in_channels: list[int] = None,
        gate_init: float = 0.1,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        lora_n_layers: int = 8,
    ):
        super().__init__()
        if head_sizes is None:
            head_sizes = {"agent": 10, "action": 22, "loc": 16, "duplex": 49, "triplet": 86}
        if fpn_in_channels is None:
            fpn_in_channels = [40, 112, 320]

        self.clip_len = clip_len
        self.d_model = d_model
        n_levels = len(fpn_in_channels)

        # ---- Spatial branch: EfficientNet + FPN ----
        self.backbone = EfficientNetBackbone(
            backbone_name, freeze_blocks=backbone_freeze_blocks, pretrained=True,
        )
        self.fpn = FPN(in_channels=fpn_in_channels, out_channels=d_model)

        # ---- Semantic branch: Qwen ViT + gated fusion ----
        self.vlm_fusion = VLMSemanticFusion(
            model_id=model_id,
            vit_dim=vit_dim,
            d_model=d_model,
            n_levels=n_levels,
            gate_init=gate_init,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            lora_n_layers=lora_n_layers,
        )

        # ---- Deformable DETR decoder ----
        self.decoder = DeformableDETRDecoder(
            d_model=d_model,
            n_heads=nhead,
            num_layers=num_decoder_layers,
            dim_ffn=dim_ffn,
            dropout=dropout,
            num_queries=num_queries,
            clip_len=clip_len,
            n_levels=n_levels,
            n_points=n_deform_points,
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

    def lora_parameters(self) -> list[nn.Parameter]:
        """Qwen ViT LoRA adapters."""
        return self.vlm_fusion.lora_parameters()

    def backbone_parameters(self) -> list[nn.Parameter]:
        """EfficientNet trainable blocks."""
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def decoder_parameters(self) -> list[nn.Parameter]:
        """FPN + VLM fusion (non-LoRA) + deformable decoder."""
        params = list(self.fpn.parameters())
        params += self.vlm_fusion.fusion_parameters()
        params += list(self.decoder.parameters())
        return params

    def head_parameters(self) -> list[nn.Parameter]:
        """Classification heads + agentness."""
        return list(self.cls_heads.parameters()) + list(self.agentness_head.parameters())

    def _normalize_for_efficientnet(self, frames: torch.Tensor) -> torch.Tensor:
        """ImageNet-normalize a batch of [0,1] RGB frames.

        Args:
            frames: [B, 3, H, W] in [0, 1] float.

        Returns:
            [B, 3, H, W] normalized.
        """
        return (frames - self._img_mean.to(frames.dtype)) / self._img_std.to(frames.dtype)

    def forward(
        self,
        pil_frames: list,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for one clip.

        Args:
            pil_frames:     List of T PIL images (used for EfficientNet after
                            resize + normalize).
            pixel_values:   Qwen-processed patch tensor (for VLM branch).
            image_grid_thw: Qwen grid info tensor.

        Returns:
            dict with:
                pred_boxes:  [N_queries, T, 4] — sigmoid [cx,cy,w,h] in [0,1]
                pred_logits: {head: [N_queries, C]} — raw logits
                query_feats: [N_queries, d_model]
                T: int
        """
        device = pixel_values.device
        dtype = next(self.backbone.parameters()).dtype

        # ---- Step 1: EfficientNet per-frame features ----
        # Convert PIL → tensor, resize to INPUT_SIZE, normalize
        frame_tensors = []
        for img in pil_frames:
            t = TF.to_tensor(img)  # [3, H, W] in [0, 1]
            t = TF.resize(t, [448, 448], antialias=True)
            frame_tensors.append(t)

        # Stack and normalize: [T, 3, 448, 448]
        frames_batch = torch.stack(frame_tensors).to(device=device, dtype=dtype)
        frames_batch = self._normalize_for_efficientnet(frames_batch)

        # Extract multi-scale features: dict with C3, C4, C5
        cnn_features = self.backbone(frames_batch)  # each [T, ch, H_i, W_i]

        # ---- Step 2: FPN ----
        fpn_features = self.fpn(cnn_features)  # [P3, P4, P5], each [T, 256, H_i, W_i]

        # ---- Step 3: VLM semantic fusion ----
        fused_features = self.vlm_fusion(
            pixel_values, image_grid_thw, fpn_features
        )  # [P3', P4', P5'], each [T, 256, H_i, W_i]

        # ---- Step 4: Per-frame multi-scale features for decoder ----
        # Pass fused features directly — no temporal stacking.
        # Each level is [T, 256, H_i, W_i] with B=T (standard 2D per frame).
        T = len(pil_frames)
        spatial_shapes_list = []
        for feat in fused_features:
            _, C, H_i, W_i = feat.shape
            spatial_shapes_list.append([H_i, W_i])

        spatial_shapes = torch.tensor(spatial_shapes_list, device=device, dtype=torch.long)

        # ---- Step 5: Deformable DETR decoder ----
        query_feats, pred_boxes, aux_outputs = self.decoder(
            fused_features, spatial_shapes, clip_len=T
        )
        # query_feats: [N_queries, d_model], pred_boxes: [N_queries, T, 4]
        # aux_outputs: list of (query_feats, pred_boxes) per earlier layer

        # ---- Step 6: Classification heads ----
        pred_logits = self.cls_heads(query_feats)
        pred_logits["agentness"] = self.agentness_head(query_feats.float())

        # Auxiliary outputs: apply shared cls heads to each layer's features
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

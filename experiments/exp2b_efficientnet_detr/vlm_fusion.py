"""VLM semantic fusion: Qwen2.5-VL ViT feature extraction + gated per-level injection.

The Qwen ViT provides rich semantic features (understands "pedestrian", "vehicle",
etc.) but at low spatial resolution (16×16). The gated fusion injects these
semantics into each FPN level without replacing the precise spatial features
from EfficientNet.

Follows the Frozen-DETR (NeurIPS 2024) paradigm: foundation model features
enrich a detection-native backbone rather than replacing it.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class QwenViTExtractor(nn.Module):
    """Extracts per-frame spatial feature maps from Qwen2.5-VL's frozen ViT.

    Output: List of T tensors, each [H', W', D] where D=3584.
    Copied from exp2 for checkpoint compatibility.
    """

    def __init__(self, model_id: str, freeze: bool = True):
        super().__init__()
        from transformers import Qwen2_5_VLForConditionalGeneration

        full = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cpu",
        )
        self.visual = full.model.visual
        del full

        if freeze:
            for p in self.visual.parameters():
                p.requires_grad_(False)

    def add_lora(
        self,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: list | None = None,
        n_layers: int = 8,
    ) -> None:
        """Wrap the first n_layers ViT attention blocks with LoRA adapters."""
        if not hasattr(torch, "float8_e8m0fnu") and hasattr(torch, "float8_e4m3fn"):
            torch.float8_e8m0fnu = torch.float8_e4m3fn

        from peft import LoraConfig, get_peft_model

        block_range = "|".join(str(i) for i in range(n_layers))
        attn_modules = "|".join(target_modules or ["qkv", "proj"])
        regex = rf"blocks\.({block_range})\.attn\.({attn_modules})"

        lora_config = LoraConfig(
            r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=regex, bias="none", inference_mode=False,
        )
        self.visual = get_peft_model(self.visual, lora_config)

    def lora_parameters(self):
        return [p for p in self.visual.parameters() if p.requires_grad]

    def forward(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> List[torch.Tensor]:
        out = self.visual(pixel_values, image_grid_thw)
        merged = out.pooler_output

        base = getattr(self.visual, "base_model", self.visual)
        base = getattr(base, "model", base)
        merge_size = base.spatial_merge_size

        frame_feats: List[torch.Tensor] = []
        offset = 0
        for row in image_grid_thw:
            t, h, w = int(row[0]), int(row[1]), int(row[2])
            h_m, w_m = h // merge_size, w // merge_size
            n = t * h_m * w_m
            chunk = merged[offset : offset + n].view(t, h_m, w_m, -1)
            for frame in chunk:
                frame_feats.append(frame)  # [H', W', D]
            offset += n

        return frame_feats


class FeatureProjection(nn.Module):
    """Projects ViT features from D=3584 down to d_model=256."""

    def __init__(self, d_in: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x.float()))


class VLMSemanticFusion(nn.Module):
    """Gated fusion of Qwen ViT semantic features into FPN spatial features.

    For each FPN level:
        1. Bilinear-interpolate VLM map to match FPN level's spatial size
        2. Apply per-level 1×1 conv projection
        3. Multiply by learned sigmoid gate (initialized small)
        4. Add to FPN features

    Args:
        model_id:   Qwen model ID for ViT extraction.
        vit_dim:    ViT output dimension (3584).
        d_model:    FPN channel dimension (256).
        n_levels:   Number of FPN levels to fuse into (3).
        gate_init:  Initial gate value in [0, 1] (default 0.1).
        lora_r:     LoRA rank.
        lora_alpha: LoRA alpha.
        lora_dropout: LoRA dropout.
        lora_target_modules: LoRA target module names.
        lora_n_layers: Number of ViT blocks to apply LoRA to.
    """

    def __init__(
        self,
        model_id: str,
        vit_dim: int = 3584,
        d_model: int = 256,
        n_levels: int = 3,
        gate_init: float = 0.1,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        lora_n_layers: int = 8,
    ):
        super().__init__()
        self.n_levels = n_levels

        # Qwen ViT extractor + LoRA
        self.vit = QwenViTExtractor(model_id, freeze=True)
        self.vit.add_lora(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=lora_target_modules, n_layers=lora_n_layers,
        )

        # Project VLM features to FPN dimension
        self.feat_proj = FeatureProjection(vit_dim, d_model)

        # Per-level fusion: 1×1 conv + learned gate
        self.level_projections = nn.ModuleList([
            nn.Conv2d(d_model, d_model, 1) for _ in range(n_levels)
        ])

        # Gate parameters: init to logit(gate_init) so sigmoid(gate_bias) ≈ gate_init
        gate_bias = math.log(gate_init / (1.0 - gate_init))
        self.gate_params = nn.ParameterList([
            nn.Parameter(torch.tensor(gate_bias)) for _ in range(n_levels)
        ])

    def lora_parameters(self) -> list[nn.Parameter]:
        """LoRA adapter parameters (for separate LR group)."""
        return self.vit.lora_parameters()

    def fusion_parameters(self) -> list[nn.Parameter]:
        """Fusion-specific parameters: projection, level convs, gates."""
        params = list(self.feat_proj.parameters())
        params += list(self.level_projections.parameters())
        params += list(self.gate_params)
        return params

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        fpn_features: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Fuse VLM semantics into FPN features.

        Args:
            pixel_values:    Qwen-processed patch tensor.
            image_grid_thw:  Qwen grid info.
            fpn_features:    [P3, P4, P5] from FPN, each [B, 256, H_i, W_i].
                             These are per-frame features (B = num_frames for
                             batched per-frame processing).

        Returns:
            Fused [P3', P4', P5'], same shapes as input fpn_features.
        """
        # Extract VLM features: list of T tensors, each [H_v, W_v, D]
        vlm_frames = self.vit(pixel_values, image_grid_thw)

        # Project to d_model: [T, H_v, W_v, d_model]
        vlm_maps = torch.stack([self.feat_proj(f) for f in vlm_frames])
        # → [T, d_model, H_v, W_v] for spatial ops
        vlm_maps = vlm_maps.permute(0, 3, 1, 2)

        # Fuse into each FPN level
        fused = []
        for lvl in range(self.n_levels):
            fpn_feat = fpn_features[lvl]  # [T, 256, H_i, W_i]
            H_i, W_i = fpn_feat.shape[2], fpn_feat.shape[3]

            # Interpolate VLM map to match this level's spatial size
            vlm_resized = F.interpolate(
                vlm_maps, size=(H_i, W_i), mode="bilinear", align_corners=False
            )

            # Project and gate
            vlm_proj = self.level_projections[lvl](vlm_resized)
            gate = torch.sigmoid(self.gate_params[lvl])

            fused.append(fpn_feat + gate * vlm_proj)

        return fused

"""Swin-L backbone with multi-scale feature extraction via timm.

Extracts features at three spatial scales for FPN consumption:
  C3: stride  8, 384 channels  (stage 1, 48×48 at 384 input)
  C4: stride 16, 768 channels  (stage 2, 24×24 at 384 input)
  C5: stride 32, 1536 channels (stage 3, 12×12 at 384 input)

Stages 0-1 (patch_embed + layers_0 + layers_1) are frozen by default.
"""

import torch
import torch.nn as nn
import timm


class SwinBackbone(nn.Module):
    """Swin-L feature extractor for multi-scale detection.

    Args:
        model_name: timm model name (default "swin_large_patch4_window12_384").
        freeze_stages: List of stage indices to freeze (default [0, 1]).
            Stage 0 = patch_embed + layers_0, Stage 1 = layers_1, etc.
        pretrained: Load ImageNet-22K pretrained weights.
        drop_path_rate: Stochastic depth rate (0.0 = disabled, 0.2 = standard for detection).
    """

    # Channel dimensions for each extracted level (used by FPN)
    out_channels = [384, 768, 1536]  # stages 1, 2, 3

    def __init__(
        self,
        model_name: str = "swin_large_patch4_window12_384",
        freeze_stages: list[int] | None = None,
        pretrained: bool = True,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        if freeze_stages is None:
            freeze_stages = [0, 1]

        self.body = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[1, 2, 3],  # stages 1,2,3 → C3,C4,C5
            drop_path_rate=drop_path_rate,
        )

        # Freeze specified stages
        # timm Swin children: patch_embed, layers_0, layers_1, layers_2, layers_3
        _stage_to_modules = {
            0: ["patch_embed", "layers_0"],
            1: ["layers_1"],
            2: ["layers_2"],
            3: ["layers_3"],
        }
        for stage_idx in freeze_stages:
            for mod_name in _stage_to_modules[stage_idx]:
                mod = getattr(self.body, mod_name)
                for param in mod.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract multi-scale features.

        Args:
            x: [B, 3, 384, 384] RGB images (ImageNet-normalized).

        Returns:
            dict with keys "C3", "C4", "C5" mapping to [B, C, H, W] tensors.
        """
        # timm Swin outputs NHWC [B, H, W, C] — convert to NCHW for FPN
        features = self.body(x)
        return {
            "C3": features[0].permute(0, 3, 1, 2).contiguous(),  # [B, 384, 48, 48]
            "C4": features[1].permute(0, 3, 1, 2).contiguous(),  # [B, 768, 24, 24]
            "C5": features[2].permute(0, 3, 1, 2).contiguous(),  # [B, 1536, 12, 12]
        }

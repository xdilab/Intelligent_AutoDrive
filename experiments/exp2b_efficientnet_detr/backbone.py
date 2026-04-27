"""EfficientNet-B0 backbone with multi-scale feature extraction.

Extracts features at three spatial scales for FPN consumption:
  C3: stride  8, 40 channels  (features.3, 56×56 at 448 input)
  C4: stride 16, 112 channels (features.5, 28×28 at 448 input)
  C5: stride 32, 320 channels (features.7, 14×14 at 448 input)

Stem + blocks 0-1 (features.0-2) are frozen by default.
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor


# Mapping from config backbone name to torchvision constructor + weights
_BACKBONE_REGISTRY = {
    "efficientnet_b0": (
        torchvision.models.efficientnet_b0,
        torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    ),
}

# Which feature blocks to extract and their output channel counts
_RETURN_NODES = {
    "features.3": "C3",   # stride 8,  40ch
    "features.5": "C4",   # stride 16, 112ch
    "features.7": "C5",   # stride 32, 320ch
}


class EfficientNetBackbone(nn.Module):
    """EfficientNet-B0 feature extractor for multi-scale detection.

    Args:
        backbone_name: Key into _BACKBONE_REGISTRY (default "efficientnet_b0").
        freeze_blocks: Freeze features.0 through features.{freeze_blocks-1}.
            Default 2 freezes stem + first two MBConv blocks.
        pretrained: Load ImageNet-pretrained weights.
    """

    # Channel dimensions for each extracted level (used by FPN)
    out_channels = [40, 112, 320]  # C3, C4, C5

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        freeze_blocks: int = 2,
        pretrained: bool = True,
    ):
        super().__init__()

        ctor, weights = _BACKBONE_REGISTRY[backbone_name]
        base = ctor(weights=weights if pretrained else None)

        self.body = create_feature_extractor(base, return_nodes=_RETURN_NODES)

        # Freeze early blocks
        for name, param in self.body.named_parameters():
            # name looks like "features.2.0.block.1.0.weight" etc.
            block_idx = int(name.split(".")[1])
            if block_idx < freeze_blocks:
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract multi-scale features.

        Args:
            x: [B, 3, H, W] RGB images (ImageNet-normalized).

        Returns:
            dict with keys "C3", "C4", "C5" mapping to feature tensors.
        """
        return self.body(x)

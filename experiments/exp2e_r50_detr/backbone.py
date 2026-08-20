"""ResNet-50 backbone with FrozenBatchNorm2d.

Follows Frozen-DETR/MS-DETR/models/backbone.py exactly:
  - torchvision ResNet-50 with FrozenBatchNorm2d as norm_layer
  - IntermediateLayerGetter returning layer2/3/4
  - Freeze everything except layer2, layer3, layer4
  - Output channels: [512, 1024, 2048], strides: [8, 16, 32]
  - Fully convolutional — accepts any input size
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d where the batch statistics and affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rsqrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.

    Source: Frozen-DETR/MS-DETR/models/backbone.py:27-64
    """

    def __init__(self, n: int, eps: float = 1e-5):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class ResNet50Backbone(nn.Module):
    """ResNet-50 with FrozenBatchNorm2d, returning layer2/3/4 features.

    Source: Frozen-DETR/MS-DETR/models/backbone.py:67-109

    Freeze policy (line 72): everything except layer2, layer3, layer4.
    This means conv1, bn1, layer1 are all frozen.

    Output dict keys match our FPN interface: {"C3", "C4", "C5"}.
    """

    # Channel dimensions for each returned layer
    num_channels = [512, 1024, 2048]
    strides = [8, 16, 32]

    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None,
            norm_layer=FrozenBatchNorm2d,
        )

        # Freeze everything except layer2, layer3, layer4
        # Source: backbone.py:72
        for name, parameter in backbone.named_parameters():
            if "layer2" not in name and "layer3" not in name and "layer4" not in name:
                parameter.requires_grad_(False)

        # Return layer2/3/4 (strides 8, 16, 32)
        # Source: backbone.py:76
        return_layers = {"layer2": "C3", "layer3": "C4", "layer4": "C5"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: [B, 3, H, W] — ImageNet-normalized, any spatial size.

        Returns:
            Dict with C3 [B, 512, H/8, W/8], C4 [B, 1024, H/16, W/16],
            C5 [B, 2048, H/32, W/32].
        """
        return self.body(x)

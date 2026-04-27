"""Feature Pyramid Network (FPN) with top-down pathway.

Takes multi-scale features from EfficientNet (C3, C4, C5) and produces
unified 256-channel feature maps at each level (P3, P4, P5) via lateral
connections and top-down merging.

Uses GroupNorm instead of BatchNorm for stability at batch_size=1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """Standard FPN: lateral 1x1 convs + top-down upsample-add + 3x3 smooth.

    Args:
        in_channels: Channel dimensions of input features [C3, C4, C5].
        out_channels: Output channel dimension for all levels (default 256).
        num_groups: Groups for GroupNorm (default 32).
    """

    def __init__(
        self,
        in_channels: list[int] = [40, 112, 320],
        out_channels: int = 256,
        num_groups: int = 32,
    ):
        super().__init__()
        assert len(in_channels) == 3, "FPN expects exactly 3 input levels (C3, C4, C5)"

        # Lateral 1x1 convolutions: project each level to out_channels
        self.lateral_c5 = nn.Conv2d(in_channels[2], out_channels, 1)
        self.lateral_c4 = nn.Conv2d(in_channels[1], out_channels, 1)
        self.lateral_c3 = nn.Conv2d(in_channels[0], out_channels, 1)

        # Smooth 3x3 convolutions after merging (reduce aliasing from upsampling)
        self.smooth_p5 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
        )
        self.smooth_p4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
        )
        self.smooth_p3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, features: dict[str, torch.Tensor]
    ) -> list[torch.Tensor]:
        """Build feature pyramid.

        Args:
            features: dict with "C3", "C4", "C5" from EfficientNet backbone.

        Returns:
            [P3, P4, P5] — list of feature maps, each [B, 256, H_i, W_i].
            P3 is the largest spatial resolution, P5 the smallest.
        """
        c3, c4, c5 = features["C3"], features["C4"], features["C5"]

        # Top-down pathway
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(
            p5, size=c4.shape[2:], mode="bilinear", align_corners=False
        )
        p3 = self.lateral_c3(c3) + F.interpolate(
            p4, size=c3.shape[2:], mode="bilinear", align_corners=False
        )

        # Smooth
        p5 = self.smooth_p5(p5)
        p4 = self.smooth_p4(p4)
        p3 = self.smooth_p3(p3)

        return [p3, p4, p5]

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SpatiotemporalPositionEncoding(nn.Module):
    """
    2D sinusoidal spatial encoding + learned temporal embedding.
    """

    def __init__(self, d_model: int, max_t: int = 16):
        super().__init__()
        self.d_model = d_model
        self.temporal = nn.Embedding(max_t, d_model)

    def _spatial_2d(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        d_half = self.d_model // 2
        y_dim = d_half
        x_dim = self.d_model - y_dim

        yy = torch.arange(h, device=device, dtype=torch.float32).unsqueeze(1).repeat(1, w)
        xx = torch.arange(w, device=device, dtype=torch.float32).unsqueeze(0).repeat(h, 1)

        yy = yy / max(h - 1, 1)
        xx = xx / max(w - 1, 1)

        def encode(coord: torch.Tensor, d: int) -> torch.Tensor:
            pe = torch.zeros(h, w, d, device=device, dtype=torch.float32)
            div_term = torch.exp(
                torch.arange(0, d, 2, device=device, dtype=torch.float32)
                * (-math.log(10000.0) / max(d, 1))
            )
            pe[..., 0::2] = torch.sin(coord.unsqueeze(-1) * div_term)
            if d > 1:
                pe[..., 1::2] = torch.cos(coord.unsqueeze(-1) * div_term[: pe[..., 1::2].shape[-1]])
            return pe

        pe_y = encode(yy, y_dim)
        pe_x = encode(xx, x_dim)
        return torch.cat([pe_y, pe_x], dim=-1).view(h * w, self.d_model)

    def forward(self, t: int, h: int, w: int, device: torch.device) -> torch.Tensor:
        spatial = self._spatial_2d(h, w, device)
        chunks = []
        for idx in range(t):
            chunks.append(spatial + self.temporal.weight[idx].to(device=device, dtype=torch.float32))
        return torch.cat(chunks, dim=0)


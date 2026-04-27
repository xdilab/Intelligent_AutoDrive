from __future__ import annotations

import math

import torch
import torch.nn as nn


class SpatiotemporalPositionEncoding(nn.Module):
    """
    Tells the DETR decoder WHERE in space and time each memory token came from.

    Why we need this:
        Attention is permutation-invariant — if you shuffle the 2048 tokens, the
        cross-attention output is identical. Without position information the decoder
        can't reason about spatial proximity ("the agent near the left edge") or
        temporal order ("the agent that was visible in frame 3 but not frame 7").

    Design: 2D sinusoidal spatial encoding (fixed, no parameters) +
            learned temporal embedding (one vector per frame index).

    Sinusoidal vs learned spatial:
        Sinusoidal is chosen because it generalises to unseen H/W sizes — learned
        embeddings overfit to training shapes. Temporal uses learned because the
        8-frame clip is fixed and learning temporal semantics is valuable.

    Output: one encoding vector [d_model] per spatiotemporal token, returned as
            [T*H*W, d_model] — added to the projected ViT features before the
            decoder cross-attends to them.
    """

    def __init__(self, d_model: int, max_t: int = 16):
        super().__init__()
        self.d_model = d_model
        # One learned vector per possible frame index.
        # max_t=16 gives room for clips up to 16 frames; exp2 uses 8.
        self.temporal = nn.Embedding(max_t, d_model)

    def _spatial_2d(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """
        Build a 2D sinusoidal position grid of shape [h*w, d_model].

        Standard sinusoidal PE (Vaswani 2017) handles 1D sequences. For 2D
        we split the d_model budget evenly: first half encodes the y-axis
        (row), second half encodes the x-axis (column).

        Each axis uses the same sinusoidal formula:
            PE(pos, 2i)   = sin(pos / 10000^(2i/d))
            PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

        Positions are normalised to [0,1] so the encoding is scale-invariant
        across different input resolutions (after Qwen's spatial merge the
        grid dimensions change with image resolution).
        """
        d_half = self.d_model // 2
        y_dim = d_half               # dimensions allocated to y-axis
        x_dim = self.d_model - y_dim  # dimensions allocated to x-axis (handles odd d_model)

        # Build coordinate grids, both [h, w], normalised to [0,1]
        yy = torch.arange(h, device=device, dtype=torch.float32).unsqueeze(1).repeat(1, w)
        xx = torch.arange(w, device=device, dtype=torch.float32).unsqueeze(0).repeat(h, 1)
        yy = yy / max(h - 1, 1)  # avoid div-by-zero when h=1
        xx = xx / max(w - 1, 1)

        def encode(coord: torch.Tensor, d: int) -> torch.Tensor:
            # coord is [h, w]; output is [h, w, d]
            pe = torch.zeros(h, w, d, device=device, dtype=torch.float32)
            # div_term[i] = 10000^(2i/d) — controls the frequency of each dimension.
            # Low-index dims oscillate quickly (fine-grained position), high-index
            # dims oscillate slowly (coarse position). Together they form a unique
            # binary-like code for each position.
            div_term = torch.exp(
                torch.arange(0, d, 2, device=device, dtype=torch.float32)
                * (-math.log(10000.0) / max(d, 1))
            )
            pe[..., 0::2] = torch.sin(coord.unsqueeze(-1) * div_term)
            if d > 1:
                # div_term may have one more element than pe[...,1::2] when d is odd
                pe[..., 1::2] = torch.cos(coord.unsqueeze(-1) * div_term[: pe[..., 1::2].shape[-1]])
            return pe

        pe_y = encode(yy, y_dim)  # [h, w, y_dim]
        pe_x = encode(xx, x_dim)  # [h, w, x_dim]
        # Concatenate along the feature axis → [h, w, d_model], then flatten spatial
        return torch.cat([pe_y, pe_x], dim=-1).view(h * w, self.d_model)

    def forward(self, t: int, h: int, w: int, device: torch.device) -> torch.Tensor:
        """
        Returns [T*H*W, d_model] — one position vector per spatiotemporal token.

        For each frame idx we add the shared spatial grid to the frame's learned
        temporal embedding. Addition (not concatenation) keeps d_model fixed.
        """
        spatial = self._spatial_2d(h, w, device)  # [h*w, d_model], shared across frames
        chunks = []
        for idx in range(t):
            # temporal.weight[idx] is [d_model]; broadcast-add to [h*w, d_model]
            chunks.append(spatial + self.temporal.weight[idx].to(device=device, dtype=torch.float32))
        # Stack frames along the token axis → [T*H*W, d_model]
        return torch.cat(chunks, dim=0)

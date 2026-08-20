"""Deformable DETR encoder + decoder with multi-scale deformable attention.

Uses the official CUDA kernel (SenseTime) when available for ~2-3x speedup
on MSDeformAttn forward/backward. Falls back to pure PyTorch grid_sample
if the CUDA extension is not installed.

Architecture (Frozen-DETR, Fu et al., NeurIPS 2024):

Encoder (new in exp2c):
  - 6-layer post-norm deformable self-attention encoder
  - Operates on 4 scales: P3, P4, P5 + CLIP patch tokens
  - After encoding, CLIP tokens are stripped — decoder sees only CNN scales
  - Reference: DINO-coco/models/dino/deformable_transformer.py:278-331, 819-874

Decoder (from exp2b, with per-layer CLS injection):
  - 6-layer pre-norm decoder with deformable cross-attention + temporal self-attention
  - Per-layer CLIP CLS token injection: concat → layer → strip
  - Iterative box refinement, auxiliary outputs
  - Reference: DINO-coco/models/dino/deformable_transformer.py:604-782
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

# Try to load the official CUDA kernel for MSDeformAttn.
# Falls back to pure-PyTorch grid_sample if not available.
try:
    import MultiScaleDeformableAttention as MSDA
    _CUDA_MSDA = True
except ImportError:
    _CUDA_MSDA = False


class _MSDeformAttnCUDA(Function):
    """Autograd wrapper for the official CUDA MSDeformAttn kernel."""

    @staticmethod
    def forward(ctx, value, spatial_shapes, level_start_index,
                sampling_locations, attention_weights):
        ctx.save_for_backward(value, spatial_shapes, level_start_index,
                              sampling_locations, attention_weights)
        return MSDA.ms_deform_attn_forward(
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights, 64)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (value, spatial_shapes, level_start_index,
         sampling_locations, attention_weights) = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, spatial_shapes, level_start_index,
                sampling_locations, attention_weights,
                grad_output.contiguous(), 64)
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Inverse of sigmoid: log(x / (1 - x)). Used for iterative box refinement."""
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


# ---------------------------------------------------------------------------
# Multi-Scale Deformable Attention (CUDA kernel with PyTorch fallback)
# ---------------------------------------------------------------------------

class MSDeformAttn(nn.Module):
    """Multi-Scale Deformable Attention.

    For each query, samples K points per head per feature level.
    Offsets and attention weights are predicted from the query.

    Args:
        d_model: Hidden dimension (256).
        n_heads: Number of attention heads (8).
        n_levels: Number of FPN levels (3 for decoder, 4 for encoder).
        n_points: Sampling points per head per level (4).
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 n_levels: int = 3, n_points: int = 4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        # Predict sampling offsets: each head samples n_points per level,
        # each point has (dx, dy) offset from the reference point.
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)

        # Predict attention weights: one weight per sampling point, softmaxed
        # across all (levels * points) for each head.
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)

        # Value projection applied to the flattened multi-scale features.
        self.value_proj = nn.Linear(d_model, d_model)

        # Output projection after aggregation.
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        # Initialize offsets to sample in a small grid around reference
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.n_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        # shape: [n_heads, n_levels, n_points, 2]
        grid_init = grid_init.view(self.n_heads, 1, 1, 2).repeat(
            1, self.n_levels, self.n_points, 1
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query:            [B, N_q, d_model]
            reference_points: [B, N_q, n_levels, 2] — normalized (x, y) in [0, 1]
            value:            [B, sum(H_i*W_i), d_model] — flattened multi-scale features
            spatial_shapes:   [n_levels, 2] — (H_i, W_i) for each level
            level_start_index:[n_levels] — cumulative start index per level in value

        Returns: [B, N_q, d_model]
        """
        B, N_q, _ = query.shape
        B, N_v, _ = value.shape

        value = self.value_proj(value)
        # [B, N_v, n_heads, head_dim]
        value = value.view(B, N_v, self.n_heads, self.head_dim)

        # Predict offsets: [B, N_q, n_heads, n_levels, n_points, 2]
        offsets = self.sampling_offsets(query).view(
            B, N_q, self.n_heads, self.n_levels, self.n_points, 2
        )

        # Predict attention weights: [B, N_q, n_heads, n_levels * n_points]
        attn_weights = self.attention_weights(query).view(
            B, N_q, self.n_heads, self.n_levels * self.n_points
        )
        attn_weights = F.softmax(attn_weights, dim=-1).view(
            B, N_q, self.n_heads, self.n_levels, self.n_points
        )

        # Compute sampling locations in [0,1] coordinate space
        # reference_points: [B, N_q, n_levels, 2] → [B, N_q, 1, n_levels, 1, 2]
        ref = reference_points[:, :, None, :, None, :]
        # spatial_shapes: [n_levels, 2] → offset_normalizer: [1, 1, 1, n_levels, 1, 2]
        offset_normalizer = spatial_shapes.flip(-1)[None, None, None, :, None, :].float()
        # Sampling locations in [0,1]: reference + offset / spatial_size
        sampling_locations = ref + offsets / offset_normalizer
        # → [B, N_q, n_heads, n_levels, n_points, 2]

        if _CUDA_MSDA:
            output = self._forward_cuda(
                value, spatial_shapes, level_start_index,
                sampling_locations, attn_weights)
        else:
            output = self._forward_pytorch(
                value, spatial_shapes, level_start_index,
                sampling_locations, attn_weights, B, N_q)

        return self.output_proj(output)

    def _forward_cuda(
        self,
        value: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        sampling_locations: torch.Tensor,
        attn_weights: torch.Tensor,
    ) -> torch.Tensor:
        """CUDA kernel path — fused im2col, no Python-level loop over levels.

        The kernel requires float32 (AT_DISPATCH_FLOATING_TYPES), so we cast
        from bf16/fp16 if needed and cast back after.
        """
        orig_dtype = value.dtype
        needs_cast = orig_dtype not in (torch.float32, torch.float64)

        if needs_cast:
            value = value.float()
            sampling_locations = sampling_locations.float()
            attn_weights = attn_weights.float()

        # Kernel expects contiguous tensors and int64 shapes/indices
        output = _MSDeformAttnCUDA.apply(
            value.contiguous(),
            spatial_shapes.to(torch.int64).contiguous(),
            level_start_index.to(torch.int64).contiguous(),
            sampling_locations.contiguous(),
            attn_weights.contiguous(),
        )
        # output: [B, N_q, n_heads * head_dim] — kernel merges heads internally
        if needs_cast:
            output = output.to(orig_dtype)
        return output

    def _forward_pytorch(
        self,
        value: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        sampling_locations: torch.Tensor,
        attn_weights: torch.Tensor,
        B: int,
        N_q: int,
    ) -> torch.Tensor:
        """Pure-PyTorch fallback — loops over levels with F.grid_sample."""
        # Convert from [0,1] to grid_sample coords [-1,1]
        sampling_grid = 2.0 * sampling_locations - 1.0

        output = torch.zeros(B, N_q, self.n_heads, self.head_dim,
                             device=value.device, dtype=value.dtype)

        for lvl in range(self.n_levels):
            H_l, W_l = spatial_shapes[lvl]
            start = level_start_index[lvl]
            end = start + H_l * W_l

            val_lvl = value[:, start:end, :, :].view(B, H_l, W_l, self.n_heads, self.head_dim)
            val_lvl = val_lvl.permute(0, 3, 4, 1, 2).reshape(
                B * self.n_heads, self.head_dim, H_l.item(), W_l.item()
            )

            grid_lvl = sampling_grid[:, :, :, lvl, :, :]
            grid_lvl = grid_lvl.permute(0, 2, 1, 3, 4).reshape(
                B * self.n_heads, N_q, self.n_points, 2
            )

            sampled = F.grid_sample(
                val_lvl, grid_lvl,
                mode="bilinear", padding_mode="zeros", align_corners=False,
            )
            sampled = sampled.view(B, self.n_heads, self.head_dim, N_q, self.n_points)

            w = attn_weights[:, :, :, lvl, :]  # [B, N_q, n_heads, n_points]
            w = w.permute(0, 2, 1, 3)          # [B, n_heads, N_q, n_points]
            agg = (sampled * w.unsqueeze(2)).sum(dim=-1)
            output += agg.permute(0, 3, 1, 2)

        output = output.reshape(B, N_q, self.d_model)
        return output


# ---------------------------------------------------------------------------
# Sinusoidal 2D Positional Encoding (for encoder)
# ---------------------------------------------------------------------------

def _get_sinusoidal_pos_embed(
    spatial_shapes: torch.Tensor,
    d_model: int,
    temperature: float = 10000.0,
    device: torch.device = None,
) -> list[torch.Tensor]:
    """Generate normalized 2D sinusoidal positional embeddings per level.

    Matches PositionEmbeddingSine from Frozen-DETR:
      DINO-coco/models/dino/position_encoding.py:24-60

    Args:
        spatial_shapes: [n_levels, 2] — (H, W) per level
        d_model: embedding dimension (256)
        temperature: frequency base (10000)
        device: target device

    Returns:
        List of [1, H*W, d_model] tensors, one per level.
    """
    pos_list = []
    half = d_model // 2
    dim_t = torch.arange(half, dtype=torch.float32, device=device)
    dim_t = temperature ** (2 * (dim_t // 2) / half)

    for H, W in spatial_shapes:
        H, W = int(H), int(W)
        # Normalized coordinates in [0, 2π]
        scale = 2 * math.pi
        y_embed = torch.arange(0.5, H + 0.5, dtype=torch.float32, device=device)
        x_embed = torch.arange(0.5, W + 0.5, dtype=torch.float32, device=device)
        y_embed = y_embed / H * scale  # [H]
        x_embed = x_embed / W * scale  # [W]

        # Outer product → [H, W]
        y_embed = y_embed[:, None].expand(H, W)
        x_embed = x_embed[None, :].expand(H, W)

        # Divide by frequency: [H, W, half]
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = x_embed[:, :, None] / dim_t

        # Interleave sin/cos: [H, W, half] → [H, W, half]
        pos_y = torch.stack([pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()], dim=3).flatten(2)
        pos_x = torch.stack([pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()], dim=3).flatten(2)

        # Concat y + x: [H, W, d_model] → [1, H*W, d_model]
        pos = torch.cat([pos_y, pos_x], dim=2).view(H * W, d_model).unsqueeze(0)
        pos_list.append(pos)

    return pos_list


# ---------------------------------------------------------------------------
# Encoder Layer + Full Encoder
# ---------------------------------------------------------------------------

class DeformableEncoderLayer(nn.Module):
    """Post-norm deformable encoder layer.

    Self-attention uses (src + pos) as query and raw src as value.
    Reference: DINO-coco/models/dino/deformable_transformer.py:819-874
    """

    def __init__(self, d_model: int = 256, d_ffn: int = 1024,
                 dropout: float = 0.1, n_levels: int = 4,
                 n_heads: int = 8, n_points: int = 4):
        super().__init__()
        # Self-attention
        self.self_attn = MSDeformAttn(d_model, n_heads, n_levels, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor, pos: torch.Tensor,
                reference_points: torch.Tensor,
                spatial_shapes: torch.Tensor,
                level_start_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src:              [B, sum(H_i*W_i), d_model]
            pos:              [B, sum(H_i*W_i), d_model]
            reference_points: [B, sum(H_i*W_i), n_levels, 2]
            spatial_shapes:   [n_levels, 2]
            level_start_index:[n_levels]
        Returns: [B, sum(H_i*W_i), d_model]
        """
        # Self-attention: query = src + pos, value = src (ref: line 863)
        src2 = self.self_attn(src + pos, reference_points, src,
                              spatial_shapes, level_start_index)
        src = src + self.dropout1(src2)
        src = self.norm1(src)                    # post-norm (ref: line 865)

        # FFN (ref: lines 856-858)
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)                    # post-norm

        return src


class DeformableDETREncoder(nn.Module):
    """Multi-scale deformable encoder.

    Flattens multi-scale features, adds sinusoidal positional encoding +
    learned level embeddings, then runs 6 layers of deformable self-attention.

    Reference: DINO-coco/models/dino/deformable_transformer.py:278-313, 500-570

    Args:
        d_model:    Hidden dimension (256).
        n_heads:    Attention heads (8).
        num_layers: Encoder layers (6).
        d_ffn:      FFN hidden dim (1024).
        dropout:    Dropout rate (0.1).
        n_levels:   Number of feature levels (4: P3+P4+P5+CLIP).
        n_points:   Deformable sampling points per head per level (4).
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 num_layers: int = 6, d_ffn: int = 1024,
                 dropout: float = 0.1, n_levels: int = 4,
                 n_points: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels

        self.layers = nn.ModuleList([
            DeformableEncoderLayer(d_model, d_ffn, dropout, n_levels, n_heads, n_points)
            for _ in range(num_layers)
        ])

        # Learned per-level embedding (ref: uses nn.Parameter, line 164)
        self.level_embed = nn.Parameter(torch.Tensor(n_levels, d_model))
        nn.init.normal_(self.level_embed)

    @staticmethod
    def _get_reference_points(spatial_shapes: torch.Tensor,
                              device: torch.device) -> torch.Tensor:
        """Grid reference points — each token references its own normalized position.

        Simplified from Frozen-DETR (no valid_ratios / padding masks — our
        inputs are unpadded fixed-size feature maps).

        Reference: deformable_transformer.py:502-514

        Args:
            spatial_shapes: [n_levels, 2] — (H, W) per level

        Returns:
            reference_points: [1, sum(H_i*W_i), n_levels, 2] — (x, y) in [0, 1]
        """
        reference_points_list = []
        for H, W in spatial_shapes:
            H, W = int(H), int(W)
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1) / H  # normalize to [0, 1]
            ref_x = ref_x.reshape(-1) / W
            ref = torch.stack([ref_x, ref_y], dim=-1)  # [H*W, 2]
            reference_points_list.append(ref)

        # [sum(H_i*W_i), 2] → [1, sum(H_i*W_i), 1, 2] → [1, sum(H_i*W_i), n_levels, 2]
        reference_points = torch.cat(reference_points_list, dim=0)
        reference_points = reference_points.unsqueeze(0).unsqueeze(2)
        reference_points = reference_points.expand(-1, -1, len(spatial_shapes), -1)
        return reference_points

    def forward(
        self,
        multi_scale_features: list[torch.Tensor],
        spatial_shapes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            multi_scale_features: list of [T, d_model, H_i, W_i], one per level.
                4 levels for exp2c: P3(56×56), P4(28×28), P5(14×14), CLIP(24×24).
            spatial_shapes: [n_levels, 2] — (H_i, W_i) per level.

        Returns:
            memory:           [T, sum(H_i*W_i), d_model]
            spatial_shapes:   [n_levels, 2] (pass-through)
            level_start_index:[n_levels]
        """
        device = multi_scale_features[0].device

        # Flatten + add level_embed + sinusoidal pos embed (ref: lines 272-293)
        pos_embeds = _get_sinusoidal_pos_embed(
            spatial_shapes, self.d_model, device=device
        )

        src_flatten = []
        pos_flatten = []
        for lvl, feat in enumerate(multi_scale_features):
            # feat: [T, d_model, H, W]
            feat_flat = feat.flatten(2).transpose(1, 2)       # [T, H*W, d_model]
            pos = pos_embeds[lvl]                              # [1, H*W, d_model]
            pos = pos + self.level_embed[lvl].view(1, 1, -1)  # add level embed
            src_flatten.append(feat_flat)
            pos_flatten.append(pos.expand(feat_flat.shape[0], -1, -1))

        src = torch.cat(src_flatten, dim=1)   # [T, sum(H_i*W_i), d_model]
        pos = torch.cat(pos_flatten, dim=1)   # [T, sum(H_i*W_i), d_model]

        # Build level_start_index
        level_sizes = spatial_shapes[:, 0] * spatial_shapes[:, 1]
        level_start_index = torch.cat([
            torch.zeros(1, device=device, dtype=torch.long),
            level_sizes.cumsum(0)[:-1],
        ])

        # Reference points: each token at its own grid position (ref: line 549)
        reference_points = self._get_reference_points(spatial_shapes, device)
        # Expand for batch: [1, N, n_levels, 2] → [T, N, n_levels, 2]
        reference_points = reference_points.expand(src.shape[0], -1, -1, -1)

        # Encoder layers
        output = src
        for layer in self.layers:
            output = layer(output, pos, reference_points,
                           spatial_shapes, level_start_index)

        return output, spatial_shapes, level_start_index


# ---------------------------------------------------------------------------
# Decoder Layer + Full Decoder
# ---------------------------------------------------------------------------

class DeformableDecoderLayer(nn.Module):
    """Single decoder layer with four stages (pre-norm residual):

    1. Per-frame self-attention: queries attend to each other within each frame
    2. Deformable cross-attention: queries attend to multi-scale features (per-frame)
    3. Temporal self-attention: each query attends to itself across T frames
    4. FFN
    """

    def __init__(self, d_model: int, n_heads: int, dim_ffn: int,
                 dropout: float, n_levels: int, n_points: int, clip_len: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        # Per-frame self-attention among queries
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Multi-scale deformable cross-attention (per-frame)
        self.cross_attn = MSDeformAttn(d_model, n_heads, n_levels, n_points)

        # Temporal self-attention across frames per query
        self.temporal_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Learned temporal position encoding for temporal attention
        self.temporal_pos = nn.Parameter(torch.randn(clip_len, d_model) * 0.02)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ffn),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_ffn, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            queries:          [T, N_q, d_model]  (B=T, per-frame)
            memory:           [T, sum(H_i*W_i), d_model]
            reference_points: [T, N_q, n_levels, 2]
            spatial_shapes:   [n_levels, 2]
            level_start_index:[n_levels]

        Returns: [T, N_q, d_model]
        """
        T = queries.shape[0]

        # 1. Per-frame self-attention among queries
        q = self.norm1(queries)
        sa_out, _ = self.self_attn(q, q, q, need_weights=False)
        queries = queries + self.dropout(sa_out)

        # 2. Deformable cross-attention (per-frame, standard 2D)
        q = self.norm2(queries)
        ca_out = self.cross_attn(q, reference_points, memory,
                                 spatial_shapes, level_start_index)
        queries = queries + self.dropout(ca_out)

        # 3. Temporal self-attention (across T frames per query)
        q = self.norm3(queries)
        # Reshape: [T, N_q, D] → [N_q, T, D] (batch=N_q, seq=T)
        q_t = q.transpose(0, 1)
        # Add temporal position encoding to Q and K (not V)
        pos = self.temporal_pos[:T].unsqueeze(0)  # [1, T, D]
        q_t_with_pos = q_t + pos
        # V uses raw features without position encoding
        v_t = q.transpose(0, 1)
        ta_out, _ = self.temporal_attn(q_t_with_pos, q_t_with_pos, v_t,
                                       need_weights=False)
        # Reshape back: [N_q, T, D] → [T, N_q, D]
        queries = queries + self.dropout(ta_out.transpose(0, 1))

        # 4. FFN
        q = self.norm4(queries)
        queries = queries + self.dropout(self.ffn(q))

        return queries


class MLP(nn.Module):
    """Simple multi-layer perceptron (used for box regression)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(d_in, d_out) for d_in, d_out in zip(dims[:-1], dims[1:])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x, inplace=True)
        return x


class DeformableDETRDecoder(nn.Module):
    """Full deformable DETR decoder with per-frame processing, iterative
    box refinement, per-layer CLS injection, and auxiliary outputs.

    300 learnable queries attend to per-frame multi-scale features via
    deformable cross-attention. Temporal self-attention in each layer
    provides inter-frame reasoning. Each layer predicts boxes and updates
    reference points for the next layer (coarse-to-fine).

    Per-layer CLS injection (Frozen-DETR):
      Each layer: concat CLS query → run layer → strip CLS.
      Fresh projection + LayerNorm per layer.
      Reference: DINO-coco/models/dino/deformable_transformer.py:604-782

    Args:
        d_model:      Hidden dimension (256).
        n_heads:      Attention heads (8).
        num_layers:   Decoder layers (6).
        dim_ffn:      FFN hidden dim (1024).
        dropout:      Dropout rate (0.1).
        num_queries:  Number of object queries (300).
        clip_len:     Frames per clip (8).
        n_levels:     FPN levels for decoder cross-attn (3: P3, P4, P5).
        n_points:     Deformable sampling points per head per level (4).
        clip_cls_dim: CLIP CLS token dimension (768 for ViT-L/14).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        num_layers: int = 6,
        dim_ffn: int = 1024,
        dropout: float = 0.1,
        num_queries: int = 300,
        clip_len: int = 8,
        n_levels: int = 3,
        n_points: int = 4,
        clip_cls_dim: int = 768,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.n_levels = n_levels
        self.num_layers = num_layers

        # Learnable object queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Learned reference points: each query gets an (x, y) in [0, 1]
        self.reference_points_head = nn.Linear(d_model, 2)

        # Per-level embedding added to memory tokens (so decoder knows which scale)
        self.level_embed = nn.Embedding(n_levels, d_model)

        # Decoder layers (with temporal attention)
        self.layers = nn.ModuleList([
            DeformableDecoderLayer(
                d_model, n_heads, dim_ffn, dropout, n_levels, n_points, clip_len
            )
            for _ in range(num_layers)
        ])

        # Per-layer box heads for iterative refinement
        self.box_heads = nn.ModuleList([
            MLP(d_model, d_model, 4, num_layers=3)
            for _ in range(num_layers)
        ])

        # Per-layer CLIP CLS token projection + norm (ref: lines 621-626)
        # CLS token: 768-dim → d_model via per-layer Linear + LayerNorm
        self.image_query_proj = nn.ModuleList([
            nn.Linear(clip_cls_dim, d_model) for _ in range(num_layers)
        ])
        self.image_query_norm = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.reference_points_head.weight)
        nn.init.constant_(self.reference_points_head.bias, 0.0)
        # Initialize box head final layers to near-zero for stable iterative refinement
        for box_head in self.box_heads:
            nn.init.constant_(box_head.layers[-1].weight, 0.0)
            nn.init.constant_(box_head.layers[-1].bias, 0.0)

    def forward(
        self,
        multi_scale_features: list[torch.Tensor],
        spatial_shapes: torch.Tensor,
        clip_len: int,
        image_query: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list]:
        """
        Args:
            multi_scale_features: list of [T, C, H_i, W_i] per FPN level.
                Per-frame features (B=T), standard 2D spatial shapes.
                Decoder sees 3 levels: P3, P4, P5 (CLIP tokens already stripped).
            spatial_shapes: [n_levels, 2] — (H_i, W_i) per level (single-frame)
            clip_len: number of frames T
            image_query: [T, 768] — CLIP CLS tokens (optional, for per-layer injection)

        Returns:
            query_feats: [N_queries, d_model] — temporally pooled
            pred_boxes:  [N_queries, T, 4] — sigmoid [cx, cy, w, h] in [0, 1]
            aux_outputs: list of (query_feats, pred_boxes) for layers 0..N-2
        """
        T = clip_len
        device = multi_scale_features[0].device

        # Flatten per-frame multi-scale features into a single sequence
        # with level embeddings. Each frame is a batch element (B=T).
        src_flatten = []
        for lvl, feat in enumerate(multi_scale_features):
            # feat: [T, d_model, H_i, W_i]
            feat_flat = feat.flatten(2).transpose(1, 2)  # [T, H_i*W_i, C]
            feat_flat = feat_flat + self.level_embed.weight[lvl].unsqueeze(0).unsqueeze(0)
            src_flatten.append(feat_flat)

        # [T, sum(H_i*W_i), d_model] — per-frame memory
        memory = torch.cat(src_flatten, dim=1)

        # Build level_start_index from per-frame spatial shapes
        level_sizes = spatial_shapes[:, 0] * spatial_shapes[:, 1]
        level_start_index = torch.cat([
            torch.zeros(1, device=device, dtype=torch.long),
            level_sizes.cumsum(0)[:-1],
        ])

        # Queries: [N_q, d_model] → [T, N_q, d_model] (shared across frames)
        queries = self.query_embed.weight.unsqueeze(0).expand(T, -1, -1)

        # Reference points: [N_q, 2] → sigmoid → [T, N_q, 2]
        ref_points = self.reference_points_head(self.query_embed.weight).sigmoid()
        ref_points = ref_points.unsqueeze(0).expand(T, -1, -1)  # [T, N_q, 2]

        # CLS reference point: center of image [0.5, 0.5]
        if image_query is not None:
            image_ref = torch.full((T, 1, 2), 0.5, device=device, dtype=ref_points.dtype)

        # Run decoder layers with iterative box refinement + CLS injection
        all_boxes = []
        all_query_feats = []

        for lid, layer in enumerate(self.layers):
            # Per-layer CLS injection (ref: lines 713-717)
            if image_query is not None:
                cls_q = self.image_query_norm[lid](
                    self.image_query_proj[lid](image_query)
                ).unsqueeze(1)                              # [T, 1, d_model]
                queries_aug = torch.cat([queries, cls_q], dim=1)  # [T, N_q+1, d_model]
                ref_aug = torch.cat([ref_points, image_ref], dim=1)  # [T, N_q+1, 2]
            else:
                queries_aug = queries
                ref_aug = ref_points

            # Expand reference points for all levels: [T, N_q(+1), n_levels, 2]
            ref_for_attn = ref_aug.unsqueeze(2).expand(
                -1, -1, self.n_levels, -1
            )

            queries_aug = layer(queries_aug, memory, ref_for_attn,
                                spatial_shapes, level_start_index)

            # Strip CLS token from output (ref: lines 777-779)
            queries = queries_aug[:, :self.num_queries, :]

            # Box prediction: delta in inverse-sigmoid space
            delta = self.box_heads[lid](queries)  # [T, N_q, 4]

            # Iterative box refinement
            pred_center = (inverse_sigmoid(ref_points) + delta[..., :2]).sigmoid()
            pred_size = delta[..., 2:].sigmoid()
            pred_box = torch.cat([pred_center, pred_size], dim=-1)  # [T, N_q, 4]

            all_boxes.append(pred_box)
            all_query_feats.append(queries.clone())

            # Update reference points for next layer (detached)
            if lid < self.num_layers - 1:
                ref_points = pred_center.detach()

        # Final output: transpose to [N_q, T, 4] and pool features over time
        final_boxes = all_boxes[-1].permute(1, 0, 2)    # [N_q, T, 4]
        final_feats = queries.mean(dim=0)                # [N_q, d_model]

        # Auxiliary outputs for layers 0..N-2
        aux = []
        for i in range(self.num_layers - 1):
            aux_boxes = all_boxes[i].permute(1, 0, 2)       # [N_q, T, 4]
            aux_feats = all_query_feats[i].mean(dim=0)       # [N_q, d_model]
            aux.append((aux_feats, aux_boxes))

        return final_feats, final_boxes, aux

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
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query:            [B, N_q, d_model]
            reference_points: [B, N_q, n_levels, 2] or [B, N_q, n_levels, 4]
                              normalized in [0, 1]; 4D = (x, y, w, h)
            value:            [B, sum(H_i*W_i), d_model] — flattened multi-scale features
            spatial_shapes:   [n_levels, 2] — (H_i, W_i) for each level
            level_start_index:[n_levels] — cumulative start index per level in value
            padding_mask:     [B, sum(H_i*W_i)] bool — True = padded position

        Returns: [B, N_q, d_model]
        """
        B, N_q, _ = query.shape
        B, N_v, _ = value.shape

        value = self.value_proj(value)
        if padding_mask is not None:
            value = value.masked_fill(padding_mask[..., None], 0.0)
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
        if reference_points.shape[-1] == 4:
            # 4D: (cx, cy, w, h) — offsets scaled by box size (MS-DETR/DINO)
            ref_xy = reference_points[:, :, None, :, None, :2]   # [B, Nq, 1, L, 1, 2]
            ref_wh = reference_points[:, :, None, :, None, 2:]   # [B, Nq, 1, L, 1, 2]
            sampling_locations = ref_xy + offsets / self.n_points * ref_wh * 0.5
        else:
            # 2D: (cx, cy) — offsets normalized by spatial shape
            ref = reference_points[:, :, None, :, None, :]
            offset_normalizer = spatial_shapes.flip(-1)[None, None, None, :, None, :].float()
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
                level_start_index: torch.Tensor,
                padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            src:              [B, sum(H_i*W_i), d_model]
            pos:              [B, sum(H_i*W_i), d_model]
            reference_points: [B, sum(H_i*W_i), n_levels, 2]
            spatial_shapes:   [n_levels, 2]
            level_start_index:[n_levels]
            padding_mask:     [B, sum(H_i*W_i)] bool — True = padded
        Returns: [B, sum(H_i*W_i), d_model]
        """
        # Self-attention: query = src + pos, value = src (ref: line 863)
        src2 = self.self_attn(src + pos, reference_points, src,
                              spatial_shapes, level_start_index, padding_mask)
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
                              valid_ratios: torch.Tensor,
                              device: torch.device) -> torch.Tensor:
        """Grid reference points scaled by valid_ratios.

        Reference: Frozen-DETR/MS-DETR deformable_transformer.py:275-287

        Args:
            spatial_shapes: [n_levels, 2] — (H, W) per level
            valid_ratios:   [B, n_levels, 2] — (w_ratio, h_ratio) per level

        Returns:
            reference_points: [B, sum(H_i*W_i), n_levels, 2] — (x, y) scaled
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            H, W = int(H), int(W)
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
                indexing="ij",
            )
            # Normalize by valid_ratios * spatial_size (ref: lines 281-282)
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack([ref_x, ref_y], dim=-1)  # [B, H*W, 2]
            reference_points_list.append(ref)

        reference_points = torch.cat(reference_points_list, dim=1)  # [B, sum(H*W), 2]
        # Scale by valid_ratios for cross-level (ref: line 286)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points  # [B, sum(H*W), n_levels, 2]

    @staticmethod
    def get_valid_ratio(mask: torch.Tensor) -> torch.Tensor:
        """Compute valid (non-padded) ratio for one feature level.

        Reference: Frozen-DETR/MS-DETR deformable_transformer.py:124-131

        Args:
            mask: [B, H, W] bool — True = padded position

        Returns:
            valid_ratio: [B, 2] — (w_ratio, h_ratio)
        """
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], dim=1)  # [B]
        valid_W = torch.sum(~mask[:, 0, :], dim=1)   # [B]
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        return torch.stack([valid_ratio_w, valid_ratio_h], dim=-1)  # [B, 2]

    def forward(
        self,
        multi_scale_features: list[torch.Tensor],
        spatial_shapes: torch.Tensor,
        masks: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            multi_scale_features: list of [T, d_model, H_i, W_i], one per level.
                4 levels: P3, P4, P5, CLIP(24×24).
            spatial_shapes: [n_levels, 2] — (H_i, W_i) per level.
            masks: list of [T, H_i, W_i] bool per level (True=padded). If None,
                   assumes no padding (all-False masks).

        Returns:
            memory:           [T, sum(H_i*W_i), d_model]
            spatial_shapes:   [n_levels, 2] (pass-through)
            level_start_index:[n_levels]
            valid_ratios:     [T, n_levels, 2]
            mask_flatten:     [T, sum(H_i*W_i)]
        """
        device = multi_scale_features[0].device
        T = multi_scale_features[0].shape[0]

        # Build masks if not provided (no padding)
        if masks is None:
            masks = [
                torch.zeros(T, int(H), int(W), dtype=torch.bool, device=device)
                for H, W in spatial_shapes
            ]

        # Compute valid_ratios from masks (ref: line 157)
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in masks], dim=1
        )  # [T, n_levels, 2]

        # Flatten + add level_embed + sinusoidal pos embed (ref: lines 272-293)
        pos_embeds = _get_sinusoidal_pos_embed(
            spatial_shapes, self.d_model, device=device
        )

        src_flatten = []
        mask_flatten_list = []
        pos_flatten = []
        for lvl, feat in enumerate(multi_scale_features):
            # feat: [T, d_model, H, W]
            feat_flat = feat.flatten(2).transpose(1, 2)       # [T, H*W, d_model]
            mask_flat = masks[lvl].flatten(1)                  # [T, H*W]
            pos = pos_embeds[lvl]                              # [1, H*W, d_model]
            pos = pos + self.level_embed[lvl].view(1, 1, -1)  # add level embed
            src_flatten.append(feat_flat)
            mask_flatten_list.append(mask_flat)
            pos_flatten.append(pos.expand(T, -1, -1))

        src = torch.cat(src_flatten, dim=1)            # [T, sum(H_i*W_i), d_model]
        mask_flatten = torch.cat(mask_flatten_list, dim=1)  # [T, sum(H_i*W_i)]
        pos = torch.cat(pos_flatten, dim=1)            # [T, sum(H_i*W_i), d_model]

        # Build level_start_index
        level_sizes = spatial_shapes[:, 0] * spatial_shapes[:, 1]
        level_start_index = torch.cat([
            torch.zeros(1, device=device, dtype=torch.long),
            level_sizes.cumsum(0)[:-1],
        ])

        # Reference points scaled by valid_ratios (ref: line 291)
        reference_points = self._get_reference_points(
            spatial_shapes, valid_ratios, device
        )  # [T, sum(H*W), n_levels, 2]

        # Encoder layers
        output = src
        for layer in self.layers:
            output = layer(output, pos, reference_points,
                           spatial_shapes, level_start_index, mask_flatten)

        return output, spatial_shapes, level_start_index, valid_ratios, mask_flatten


# ---------------------------------------------------------------------------
# Decoder Layer + Full Decoder
# ---------------------------------------------------------------------------

class DeformableDecoderLayer(nn.Module):
    """Single decoder layer with MS-DETR dual-path support.

    When use_ms_detr=True (MS-DETR attention order):
      1. Deformable cross-attention (queries attend to multi-scale features)
      2. Auxiliary FFN → produces O2M features (before self-attn mixes queries)
      3. Per-frame self-attention (queries attend to each other)
      4. Temporal self-attention (each query across T frames)
      5. Main FFN → produces O2O features

    When use_ms_detr=False (original order):
      1. Per-frame self-attention
      2. Deformable cross-attention
      3. Temporal self-attention
      4. FFN
    """

    def __init__(self, d_model: int, n_heads: int, dim_ffn: int,
                 dropout: float, n_levels: int, n_points: int, clip_len: int,
                 use_ms_detr: bool = False, use_aux_ffn: bool = False):
        super().__init__()
        self.use_ms_detr = use_ms_detr

        self.norm_sa = nn.LayerNorm(d_model)
        self.norm_ca = nn.LayerNorm(d_model)
        self.norm_ta = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

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

        # Auxiliary FFN for O2M branch (MS-DETR)
        if use_ms_detr and use_aux_ffn:
            self.aux_ffn = nn.Sequential(
                nn.Linear(d_model, dim_ffn),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(dim_ffn, d_model),
            )
            self.norm_aux = nn.LayerNorm(d_model)

    def forward(
        self,
        queries: torch.Tensor,
        query_pos: torch.Tensor | None,
        memory: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        src_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            queries:          [T, N_q, d_model]  (B=T, per-frame)
            query_pos:        [T, N_q, d_model] or None (positional embedding for queries)
            memory:           [T, sum(H_i*W_i), d_model]
            reference_points: [T, N_q, n_levels, 2] or [T, N_q, n_levels, 4]
            spatial_shapes:   [n_levels, 2]
            level_start_index:[n_levels]
            src_padding_mask: [T, sum(H_i*W_i)] bool — True = padded

        Returns: (tgt_o2o, tgt_o2m) both [T, N_q, d_model]
        """
        T = queries.shape[0]

        def _with_pos(tensor, pos):
            return tensor if pos is None else tensor + pos

        if self.use_ms_detr:
            # --- MS-DETR order: cross → aux_ffn(O2M) → self → temporal → ffn(O2O) ---

            # 1. Deformable cross-attention
            q = self.norm_ca(_with_pos(queries, query_pos))
            ca_out = self.cross_attn(q, reference_points, memory,
                                     spatial_shapes, level_start_index,
                                     src_padding_mask)
            queries = queries + self.dropout(ca_out)

            # 2. Auxiliary FFN → O2M output (before self-attn mixes queries)
            if hasattr(self, "aux_ffn"):
                q_aux = self.norm_aux(queries)
                tgt_o2m = queries + self.dropout(self.aux_ffn(q_aux))
            else:
                tgt_o2m = queries

            # 3. Per-frame self-attention
            q = self.norm_sa(queries)
            q_pos = _with_pos(q, query_pos if query_pos is not None else None)
            sa_out, _ = self.self_attn(q_pos, q_pos, q, need_weights=False)
            queries = queries + self.dropout(sa_out)

            # 4. Temporal self-attention
            q = self.norm_ta(queries)
            q_t = q.transpose(0, 1)
            pos = self.temporal_pos[:T].unsqueeze(0)
            q_t_with_pos = q_t + pos
            v_t = q.transpose(0, 1)
            ta_out, _ = self.temporal_attn(q_t_with_pos, q_t_with_pos, v_t,
                                           need_weights=False)
            queries = queries + self.dropout(ta_out.transpose(0, 1))

            # 5. Main FFN → O2O output
            q = self.norm_ffn(queries)
            tgt_o2o = queries + self.dropout(self.ffn(q))

        else:
            # --- Original order: self → cross → temporal → ffn ---

            # 1. Per-frame self-attention
            q = self.norm_sa(queries)
            q_pos = _with_pos(q, query_pos if query_pos is not None else None)
            sa_out, _ = self.self_attn(q_pos, q_pos, q, need_weights=False)
            queries = queries + self.dropout(sa_out)

            # 2. Deformable cross-attention
            q = self.norm_ca(_with_pos(queries, query_pos))
            ca_out = self.cross_attn(q, reference_points, memory,
                                     spatial_shapes, level_start_index,
                                     src_padding_mask)
            queries = queries + self.dropout(ca_out)

            # 3. Temporal self-attention
            q = self.norm_ta(queries)
            q_t = q.transpose(0, 1)
            pos = self.temporal_pos[:T].unsqueeze(0)
            q_t_with_pos = q_t + pos
            v_t = q.transpose(0, 1)
            ta_out, _ = self.temporal_attn(q_t_with_pos, q_t_with_pos, v_t,
                                           need_weights=False)
            queries = queries + self.dropout(ta_out.transpose(0, 1))

            # 4. FFN
            q = self.norm_ffn(queries)
            tgt_o2o = queries + self.dropout(self.ffn(q))
            tgt_o2m = tgt_o2o

        return tgt_o2o, tgt_o2m


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
    box refinement, per-layer CLS injection, dual O2O/O2M outputs, and
    optional two-stage query initialization.

    Supports two modes:
    - **Learned queries** (two_stage=False): N_q learnable embeddings + learned ref points
    - **Two-stage** (two_stage=True): queries and ref points provided externally from
      encoder proposals. With mixed_selection: learnable content + proposal positions.

    Per-layer CLS injection (Frozen-DETR):
      Each layer: concat CLS query → run layer → strip CLS.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        num_layers: int = 6,
        dim_ffn: int = 2048,
        dropout: float = 0.0,
        num_queries: int = 900,
        clip_len: int = 8,
        n_levels: int = 3,
        n_points: int = 4,
        clip_cls_dim: int = 768,
        two_stage: bool = False,
        mixed_selection: bool = False,
        use_ms_detr: bool = False,
        use_aux_ffn: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.n_levels = n_levels
        self.num_layers = num_layers
        self.two_stage = two_stage
        self.mixed_selection = mixed_selection
        self.use_ms_detr = use_ms_detr

        # Learnable queries: used when not two_stage, or content-only in mixed_selection
        if not two_stage or mixed_selection:
            self.query_embed = nn.Embedding(num_queries, d_model)

        # Learned reference points (only when not two_stage)
        if not two_stage:
            self.reference_points_head = nn.Linear(d_model, 2)

        # Per-level embedding added to memory tokens
        self.level_embed = nn.Embedding(n_levels, d_model)

        # Decoder layers
        self.layers = nn.ModuleList([
            DeformableDecoderLayer(
                d_model, n_heads, dim_ffn, dropout, n_levels, n_points, clip_len,
                use_ms_detr=use_ms_detr, use_aux_ffn=use_aux_ffn,
            )
            for _ in range(num_layers)
        ])

        # Per-layer box heads for iterative refinement
        self.box_heads = nn.ModuleList([
            MLP(d_model, d_model, 4, num_layers=3)
            for _ in range(num_layers)
        ])

        # Per-layer CLIP CLS token projection + norm
        self.image_query_proj = nn.ModuleList([
            nn.Linear(clip_cls_dim, d_model) for _ in range(num_layers)
        ])
        self.image_query_norm = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        if hasattr(self, "reference_points_head"):
            nn.init.xavier_uniform_(self.reference_points_head.weight)
            nn.init.constant_(self.reference_points_head.bias, 0.0)
        for box_head in self.box_heads:
            nn.init.constant_(box_head.layers[-1].weight, 0.0)
            nn.init.constant_(box_head.layers[-1].bias, 0.0)

    def forward(
        self,
        multi_scale_features: list[torch.Tensor],
        spatial_shapes: torch.Tensor,
        clip_len: int,
        image_query: torch.Tensor | None = None,
        tgt: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
        init_ref_points: torch.Tensor | None = None,
        src_valid_ratios: torch.Tensor | None = None,
        src_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, list]:
        """
        Args:
            multi_scale_features: list of [T, d_model, H_i, W_i] per FPN level
            spatial_shapes:       [n_levels, 2] — (H, W) per level
            clip_len:             number of frames T
            image_query:          [T, clip_cls_dim] CLIP CLS token (optional)
            tgt:                  [T, N_q, d_model] content queries (two-stage)
            query_pos:            [T, N_q, d_model] position queries (two-stage)
            init_ref_points:      [T, N_q, 4] reference points (two-stage)
            src_valid_ratios:     [T, n_levels, 2] valid ratios per level
            src_padding_mask:     [T, sum(H_i*W_i)] bool — True = padded

        Returns:
            o2o_feats:  [N_q, d_model]        — time-pooled O2O query features
            o2m_feats:  [N_q, d_model] | None — time-pooled O2M features
            pred_boxes: [N_q, T, 4]           — final per-frame boxes
            aux_outputs: list of (o2o_feats, o2m_feats, boxes) for layers 0..N-2
        """
        T = clip_len
        device = multi_scale_features[0].device

        # Flatten multi-scale features with level embeddings
        src_flatten = []
        for lvl, feat in enumerate(multi_scale_features):
            feat_flat = feat.flatten(2).transpose(1, 2)
            feat_flat = feat_flat + self.level_embed.weight[lvl].unsqueeze(0).unsqueeze(0)
            src_flatten.append(feat_flat)
        memory = torch.cat(src_flatten, dim=1)

        level_sizes = spatial_shapes[:, 0] * spatial_shapes[:, 1]
        level_start_index = torch.cat([
            torch.zeros(1, device=device, dtype=torch.long),
            level_sizes.cumsum(0)[:-1],
        ])

        # --- Query initialization ---
        if self.two_stage:
            assert tgt is not None and init_ref_points is not None
            queries = tgt
            ref_points = init_ref_points
            # query_pos passed through to layers
        else:
            queries = self.query_embed.weight.unsqueeze(0).expand(T, -1, -1)
            ref_points = self.reference_points_head(self.query_embed.weight).sigmoid()
            ref_points = ref_points.unsqueeze(0).expand(T, -1, -1)
            query_pos = None

        # CLS reference point (match ref_points dimensionality)
        if image_query is not None:
            ref_dim = ref_points.shape[-1]  # 2 or 4
            image_ref = torch.full((T, 1, ref_dim), 0.5, device=device, dtype=ref_points.dtype)

        # Run decoder layers
        all_boxes = []
        all_o2o_feats = []
        all_o2m_feats = []

        for lid, layer in enumerate(self.layers):
            # Per-layer CLS injection
            if image_query is not None:
                cls_q = self.image_query_norm[lid](
                    self.image_query_proj[lid](image_query)
                ).unsqueeze(1)
                queries_aug = torch.cat([queries, cls_q], dim=1)
                ref_aug = torch.cat([ref_points, image_ref], dim=1)
                if query_pos is not None:
                    cls_pos = torch.zeros(T, 1, self.d_model, device=device,
                                          dtype=query_pos.dtype)
                    qpos_aug = torch.cat([query_pos, cls_pos], dim=1)
                else:
                    qpos_aug = None
            else:
                queries_aug = queries
                ref_aug = ref_points
                qpos_aug = query_pos

            # Scale reference points by valid_ratios (ref: lines 431-436)
            if src_valid_ratios is not None and ref_aug.shape[-1] == 4:
                vr = torch.cat([src_valid_ratios, src_valid_ratios], dim=-1)  # [T, L, 4]
                ref_for_attn = ref_aug[:, :, None, :] * vr[:, None, :, :]    # [T, N, L, 4]
            elif src_valid_ratios is not None and ref_aug.shape[-1] == 2:
                ref_for_attn = ref_aug[:, :, None, :] * src_valid_ratios[:, None, :, :]  # [T, N, L, 2]
            else:
                ref_for_attn = ref_aug.unsqueeze(2).expand(-1, -1, self.n_levels, -1)

            o2o_out, o2m_out = layer(
                queries_aug, qpos_aug, memory, ref_for_attn,
                spatial_shapes, level_start_index, src_padding_mask,
            )

            # Strip CLS token
            queries = o2o_out[:, :self.num_queries, :]
            queries_o2m = o2m_out[:, :self.num_queries, :]

            # Box prediction from O2O path (iterative refinement)
            delta = self.box_heads[lid](queries)
            if ref_points.shape[-1] == 4:
                # 4D iterative refinement: refine all 4 coords (cx, cy, w, h)
                pred_box = (inverse_sigmoid(ref_points) + delta).sigmoid()
            else:
                # 2D: only refine center, predict size from scratch
                pred_center = (inverse_sigmoid(ref_points) + delta[..., :2]).sigmoid()
                pred_size = delta[..., 2:].sigmoid()
                pred_box = torch.cat([pred_center, pred_size], dim=-1)

            all_boxes.append(pred_box)
            all_o2o_feats.append(queries.clone())
            all_o2m_feats.append(queries_o2m.clone())

            if lid < self.num_layers - 1:
                ref_points = pred_box.detach()  # pass full 4D for next layer

        # Final output
        final_boxes = all_boxes[-1].permute(1, 0, 2)
        final_o2o = queries.mean(dim=0)
        final_o2m = queries_o2m.mean(dim=0) if self.use_ms_detr else None

        # Auxiliary outputs for layers 0..N-2
        aux = []
        for i in range(self.num_layers - 1):
            aux_boxes = all_boxes[i].permute(1, 0, 2)
            aux_o2o = all_o2o_feats[i].mean(dim=0)
            aux_o2m = all_o2m_feats[i].mean(dim=0) if self.use_ms_detr else None
            aux.append((aux_o2o, aux_o2m, aux_boxes))

        return final_o2o, final_o2m, final_boxes, aux

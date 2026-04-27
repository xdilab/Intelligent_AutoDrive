"""Deformable DETR decoder with multi-scale deformable attention.

Pure PyTorch implementation (no CUDA custom ops). Follows Zhu et al.,
"Deformable DETR" (ICLR 2021) with all standard components:

1. Per-frame decoding: B=T, each frame uses standard 2D spatial shapes.
   Temporal self-attention inside each layer for inter-frame reasoning.
2. Iterative box refinement: each layer predicts box deltas and updates
   reference points for the next layer (coarse-to-fine localization).
3. Auxiliary outputs: every layer produces boxes for auxiliary loss
   supervision (direct gradient signal to all layers).

Key concepts:
  - Each query has a learned reference point (x, y) per level.
  - Cross-attention samples K=4 points per head per level around the
    reference, with learned offsets and attention weights.
  - This is O(N_queries * N_levels * K * N_heads) instead of
    O(N_queries * N_tokens), making it tractable for large token counts.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Inverse of sigmoid: log(x / (1 - x)). Used for iterative box refinement."""
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


# ---------------------------------------------------------------------------
# Multi-Scale Deformable Attention (pure PyTorch)
# ---------------------------------------------------------------------------

class MSDeformAttn(nn.Module):
    """Multi-Scale Deformable Attention.

    For each query, samples K points per head per feature level.
    Offsets and attention weights are predicted from the query.

    Args:
        d_model: Hidden dimension (256).
        n_heads: Number of attention heads (8).
        n_levels: Number of FPN levels (3).
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

        # Normalize offsets by spatial shape so they're in [0,1] coordinate space
        # reference_points: [B, N_q, n_levels, 2] → [B, N_q, 1, n_levels, 1, 2]
        ref = reference_points[:, :, None, :, None, :]  # [B, N_q, 1, n_levels, 1, 2]
        # spatial_shapes: [n_levels, 2] → offset_normalizer: [1, 1, 1, n_levels, 1, 2]
        offset_normalizer = spatial_shapes.flip(-1)[None, None, None, :, None, :].float()
        # Sampling locations in [0,1]: reference + offset / spatial_size
        sampling_locations = ref + offsets / offset_normalizer
        # Convert from [0,1] to grid_sample coords [-1,1]
        sampling_grid = 2.0 * sampling_locations - 1.0  # [B, N_q, n_heads, n_levels, n_points, 2]

        # Sample from each level using F.grid_sample
        output = torch.zeros(B, N_q, self.n_heads, self.head_dim,
                             device=query.device, dtype=query.dtype)

        for lvl in range(self.n_levels):
            H_l, W_l = spatial_shapes[lvl]
            start = level_start_index[lvl]
            end = start + H_l * W_l

            # Extract this level's values: [B, H_l, W_l, n_heads, head_dim]
            # → [B*n_heads, head_dim, H_l, W_l] for grid_sample
            val_lvl = value[:, start:end, :, :].view(B, H_l, W_l, self.n_heads, self.head_dim)
            val_lvl = val_lvl.permute(0, 3, 4, 1, 2).reshape(
                B * self.n_heads, self.head_dim, H_l.item(), W_l.item()
            )

            # Grid for this level: [B, N_q, n_heads, n_points, 2]
            grid_lvl = sampling_grid[:, :, :, lvl, :, :]
            # → [B*n_heads, N_q, n_points, 2]
            grid_lvl = grid_lvl.permute(0, 2, 1, 3, 4).reshape(
                B * self.n_heads, N_q, self.n_points, 2
            )

            # F.grid_sample: input [B*H, C, H_l, W_l], grid [B*H, N_q, n_points, 2]
            # → [B*H, head_dim, N_q, n_points]
            sampled = F.grid_sample(
                val_lvl, grid_lvl,
                mode="bilinear", padding_mode="zeros", align_corners=False,
            )
            # → [B, n_heads, head_dim, N_q, n_points]
            sampled = sampled.view(B, self.n_heads, self.head_dim, N_q, self.n_points)

            # Weighted sum over points: attn_weights[:, :, :, lvl, :] is [B, N_q, n_heads, n_points]
            w = attn_weights[:, :, :, lvl, :]  # [B, N_q, n_heads, n_points]
            w = w.permute(0, 2, 1, 3)          # [B, n_heads, N_q, n_points]

            # sampled: [B, n_heads, head_dim, N_q, n_points]
            # w:       [B, n_heads, N_q, n_points]
            # sum over points → [B, n_heads, head_dim, N_q]
            agg = (sampled * w.unsqueeze(2)).sum(dim=-1)  # [B, n_heads, head_dim, N_q]
            # → [B, N_q, n_heads, head_dim]
            output += agg.permute(0, 3, 1, 2)

        # Merge heads and project
        output = output.reshape(B, N_q, self.d_model)
        return self.output_proj(output)


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
    box refinement, and auxiliary outputs for per-layer supervision.

    300 learnable queries attend to per-frame multi-scale features via
    deformable cross-attention. Temporal self-attention in each layer
    provides inter-frame reasoning. Each layer predicts boxes and updates
    reference points for the next layer (coarse-to-fine).

    Args:
        d_model:    Hidden dimension (256).
        n_heads:    Attention heads (8).
        num_layers: Decoder layers (6).
        dim_ffn:    FFN hidden dim (1024).
        dropout:    Dropout rate (0.1).
        num_queries: Number of object queries (300).
        clip_len:   Frames per clip (8).
        n_levels:   FPN levels (3).
        n_points:   Deformable sampling points per head per level (4).
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
        # Each layer has its own box MLP: query_feat → 4 values (delta in inv-sigmoid space)
        self.box_heads = nn.ModuleList([
            MLP(d_model, d_model, 4, num_layers=3)
            for _ in range(num_layers)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.reference_points_head.weight)
        nn.init.constant_(self.reference_points_head.bias, 0.0)
        # Initialize box head final layers to near-zero for stable iterative refinement:
        # early layers predict small deltas from reference points
        for box_head in self.box_heads:
            nn.init.constant_(box_head.layers[-1].weight, 0.0)
            nn.init.constant_(box_head.layers[-1].bias, 0.0)

    def forward(
        self,
        multi_scale_features: list[torch.Tensor],
        spatial_shapes: torch.Tensor,
        clip_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list]:
        """
        Args:
            multi_scale_features: list of [T, C, H_i, W_i] per FPN level.
                Per-frame features (B=T), standard 2D spatial shapes.
            spatial_shapes: [n_levels, 2] — (H_i, W_i) per level (single-frame)
            clip_len: number of frames T

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

        # Build level_start_index from per-frame spatial shapes (no temporal scaling)
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

        # Run decoder layers with iterative box refinement
        all_boxes = []
        all_query_feats = []

        for lid, layer in enumerate(self.layers):
            # Expand reference points for all levels: [T, N_q, n_levels, 2]
            ref_points_for_attn = ref_points.unsqueeze(2).expand(
                -1, -1, self.n_levels, -1
            )

            queries = layer(queries, memory, ref_points_for_attn,
                            spatial_shapes, level_start_index)

            # Box prediction: delta in inverse-sigmoid space
            delta = self.box_heads[lid](queries)  # [T, N_q, 4]

            # Iterative box refinement:
            # Center = sigmoid(inverse_sigmoid(ref_xy) + delta_xy)
            # Size = sigmoid(delta_wh)  — starts at sigmoid(0)=0.5
            pred_center = (inverse_sigmoid(ref_points) + delta[..., :2]).sigmoid()
            pred_size = delta[..., 2:].sigmoid()
            pred_box = torch.cat([pred_center, pred_size], dim=-1)  # [T, N_q, 4]

            all_boxes.append(pred_box)
            all_query_feats.append(queries.clone())

            # Update reference points for next layer (detached — no grad through matching)
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

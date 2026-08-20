"""Exp2g — R50-FPN + Frozen-DETR + CLIP ViT-L/14 + MS-DETR (Two-Stage + O2M).

Builds on exp2f (flat 184-dim head) with the paper's core contributions:
  - Two-stage query init: encoder proposes regions, top-k become decoder queries
  - One-to-many (O2M) loss: dual decoder path, aux FFN, 6x positive training signals
  - Encoder loss: binary objectness + box regression on encoder proposals
  - Softmax agent head: softmax CE on agent (single-label), sigmoid on rest
"""

from __future__ import annotations

import copy
import math
from typing import Dict

import torch
import torch.nn as nn
from torchvision.transforms import functional as TF

from backbone import ResNet50Backbone
from fpn import FPN
from clip_encoder import CLIPVisualEncoder, PatchTokenProjection, get_clip_preprocess
from deformable_decoder import (
    DeformableDETREncoder, DeformableDETRDecoder, MLP,
)


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _resize_shorter_side(img, short_side: int, max_size: int):
    """Resize PIL image so shorter side = short_side, respecting max_size."""
    w, h = img.size
    min_side = float(min(w, h))
    max_side = float(max(w, h))
    scale = short_side / min_side
    if max_side * scale > max_size:
        scale = max_size / max_side
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    return TF.resize(img, [new_h, new_w], antialias=True)


class FrozenDETRModel(nn.Module):
    """Exp2g: R50-FPN + Frozen-DETR encoder + CLIP ViT-L/14 + MS-DETR."""

    def __init__(
        self,
        clip_model: str = "ViT-L/14@336px",
        clip_cache_dir: str | None = None,
        clip_cls_dim: int = 768,
        clip_patch_dim: int = 1024,
        clip_grid: int = 24,
        d_model: int = 256,
        num_classes: int = 184,
        clip_len: int = 8,
        num_queries: int = 900,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        nhead: int = 8,
        dim_ffn: int = 2048,
        dropout: float = 0.0,
        n_deform_points: int = 4,
        encoder_n_levels: int = 4,
        decoder_n_levels: int = 3,
        fpn_in_channels: list[int] | None = None,
        val_short_side: int = 800,
        val_max_size: int = 1333,
        # MS-DETR flags
        two_stage: bool = False,
        mixed_selection: bool = False,
        use_ms_detr: bool = False,
        use_aux_ffn: bool = False,
    ):
        super().__init__()
        if fpn_in_channels is None:
            fpn_in_channels = [512, 1024, 2048]

        self.clip_len = clip_len
        self.d_model = d_model
        self.clip_grid = clip_grid
        self.val_short_side = val_short_side
        self.val_max_size = val_max_size
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.two_stage = two_stage
        self.mixed_selection = mixed_selection
        self.use_ms_detr = use_ms_detr

        # ---- Spatial branch: ResNet-50 + FPN ----
        self.backbone = ResNet50Backbone(pretrained=True)
        self.fpn = FPN(in_channels=fpn_in_channels, out_channels=d_model)

        # ---- Semantic branch: Frozen CLIP ViT-L/14 ----
        self.clip_encoder = CLIPVisualEncoder(
            model_name=clip_model, cache_dir=clip_cache_dir,
        )
        self.patch_proj = PatchTokenProjection(
            clip_dim=clip_patch_dim, d_model=d_model,
        )
        self._clip_preprocess = get_clip_preprocess(
            self.clip_encoder.input_resolution
        )

        # ---- Deformable encoder (4 scales: P3+P4+P5+CLIP) ----
        self.encoder = DeformableDETREncoder(
            d_model=d_model,
            n_heads=nhead,
            num_layers=num_encoder_layers,
            d_ffn=dim_ffn,
            dropout=dropout,
            n_levels=encoder_n_levels,
            n_points=n_deform_points,
        )

        # ---- Deformable decoder (3 scales: P3+P4+P5, with CLS injection) ----
        self.decoder = DeformableDETRDecoder(
            d_model=d_model,
            n_heads=nhead,
            num_layers=num_decoder_layers,
            dim_ffn=dim_ffn,
            dropout=dropout,
            num_queries=num_queries,
            clip_len=clip_len,
            n_levels=decoder_n_levels,
            n_points=n_deform_points,
            clip_cls_dim=clip_cls_dim,
            two_stage=two_stage,
            mixed_selection=mixed_selection,
            use_ms_detr=use_ms_detr,
            use_aux_ffn=use_aux_ffn,
        )

        # ---- Two-stage encoder proposal modules ----
        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
            self.enc_box_head = MLP(d_model, d_model, 4, num_layers=3)

        # ---- Per-layer O2O classification heads ----
        # +1 clone for encoder proposals when two_stage (last clone = encoder head)
        num_cls_heads = num_decoder_layers + (1 if two_stage else 0)
        _cls_head = nn.Linear(d_model, num_classes)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)  # -4.595
        nn.init.constant_(_cls_head.bias, bias_value)
        self.cls_heads = nn.ModuleList(
            [copy.deepcopy(_cls_head) for _ in range(num_cls_heads)]
        )

        # ---- Per-layer O2M classification heads (MS-DETR) ----
        if use_ms_detr:
            _cls_head_o2m = nn.Linear(d_model, num_classes)
            nn.init.constant_(_cls_head_o2m.bias, bias_value)
            self.cls_heads_o2m = nn.ModuleList(
                [copy.deepcopy(_cls_head_o2m) for _ in range(num_decoder_layers)]
            )

        # Register ImageNet normalization as buffers
        self.register_buffer(
            "_img_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_img_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        )

    # ---- Two-stage proposal generation ----

    @staticmethod
    def get_proposal_pos_embed(proposals: torch.Tensor) -> torch.Tensor:
        """Sinusoidal positional encoding for 4D proposals.

        Args:
            proposals: [..., 4] in inverse-sigmoid space

        Returns:
            [..., 512] sinusoidal encoding (128 features x 4 coords)
        """
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32,
                             device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

        proposals = proposals.sigmoid() * scale  # [..., 4]
        pos = proposals[..., None] / dim_t        # [..., 4, 128]
        pos = torch.stack(
            (pos[..., 0::2].sin(), pos[..., 1::2].cos()), dim=-1
        ).flatten(-3)                              # [..., 512]
        return pos

    def gen_encoder_output_proposals(
        self,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate grid-based proposals from encoder spatial memory.

        Reference: Frozen-DETR/MS-DETR deformable_transformer.py:92-122

        Args:
            memory:              [T, N_spatial, d_model]
            memory_padding_mask: [T, N_spatial] bool — True = padded
            spatial_shapes:      [3, 2] — (H, W) per level

        Returns:
            output_memory:    [T, N_spatial, d_model] — processed through enc_output
            output_proposals: [T, N_spatial, 4] — in inverse-sigmoid space
        """
        T = memory.shape[0]
        device = memory.device

        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            H_, W_ = int(H_), int(W_)
            # Extract per-level mask and compute valid extent (ref: lines 98-100)
            mask_lvl = memory_padding_mask[:, _cur:_cur + H_ * W_].view(T, H_, W_, 1)
            valid_H = torch.sum(~mask_lvl[:, :, 0, 0], dim=1)  # [T]
            valid_W = torch.sum(~mask_lvl[:, 0, :, 0], dim=1)  # [T]

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=device),
                torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=device),
                indexing="ij",
            )
            grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
            # Normalize by valid extent (ref: lines 102-109)
            scale = torch.cat([valid_W[:, None], valid_H[:, None]], dim=1)  # [T, 2]
            grid = (grid.unsqueeze(0).expand(T, -1, -1, -1) + 0.5) / scale[:, None, None, :]
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat([grid, wh], dim=-1).view(T, -1, 4)  # [T, H*W, 4]
            proposals.append(proposal)
            _cur += H_ * W_

        output_proposals = torch.cat(proposals, dim=1)  # [T, N_spatial, 4]
        # Filter invalid proposals
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(
            dim=-1, keepdim=True
        )
        # Inverse sigmoid
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        # Mask out padded positions and invalid proposals (ref: lines 115-116)
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float("inf")
        )
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        # Process memory — zero out padded and invalid (ref: lines 119-120)
        output_memory = memory.masked_fill(memory_padding_mask.unsqueeze(-1), 0.0)
        output_memory = output_memory.masked_fill(~output_proposals_valid, 0.0)
        output_memory = self.enc_output_norm(self.enc_output(output_memory))

        return output_memory, output_proposals

    # ---- Parameter groups for optimizer ----

    def backbone_parameters(self) -> list[nn.Parameter]:
        """R50 trainable layers (layer2/3/4)."""
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def encoder_decoder_parameters(self) -> list[nn.Parameter]:
        """FPN + patch_proj + encoder + decoder + enc_output/norm (DINO-pretrained),
        excluding deformable params, heads, and fresh-init modules."""
        deform_set = set(id(p) for p in self.deformable_parameters())
        head_set = set(id(p) for p in self.head_parameters())
        fresh_set = set(id(p) for p in self.fresh_parameters())
        params = []
        modules = [self.fpn, self.patch_proj, self.encoder, self.decoder]
        if self.two_stage:
            # enc_output and enc_output_norm are DINO-pretrained; keep here
            modules += [self.enc_output, self.enc_output_norm]
        for module in modules:
            for p in module.parameters():
                if p.requires_grad and id(p) not in deform_set \
                        and id(p) not in head_set and id(p) not in fresh_set:
                    params.append(p)
        return params

    def head_parameters(self) -> list[nn.Parameter]:
        """O2O classification heads (decoder cls_heads — includes encoder cls as last clone)."""
        return list(self.cls_heads.parameters())

    def fresh_parameters(self) -> list[nn.Parameter]:
        """Fresh-init components that need higher LR:
        pos_trans, O2M heads, encoder box head."""
        params = []
        if self.two_stage:
            params += list(self.pos_trans.parameters())
            params += list(self.pos_trans_norm.parameters())
            params += list(self.enc_box_head.parameters())
        if self.use_ms_detr:
            params += list(self.cls_heads_o2m.parameters())
        return params

    def deformable_parameters(self) -> list[nn.Parameter]:
        """reference_points and sampling_offsets — get 0.1x LR per paper."""
        params = []
        for name, p in self.named_parameters():
            if p.requires_grad and (
                "reference_points" in name or "sampling_offsets" in name
            ):
                params.append(p)
        return params

    def _normalize_for_backbone(self, frames: torch.Tensor) -> torch.Tensor:
        return (frames - self._img_mean.to(frames.dtype)) / self._img_std.to(
            frames.dtype
        )

    def _preprocess_for_clip(
        self, pil_frames: list, device: torch.device
    ) -> torch.Tensor:
        clip_tensors = [self._clip_preprocess(img) for img in pil_frames]
        return torch.stack(clip_tensors).to(device)

    def forward(
        self,
        pil_frames: list,
    ) -> Dict[str, torch.Tensor]:
        device = next(self.backbone.parameters()).device
        dtype = next(self.backbone.parameters()).dtype

        # ---- Step 1: R50 per-frame features ----
        if not self.training:
            pil_frames = [
                _resize_shorter_side(img, self.val_short_side, self.val_max_size)
                for img in pil_frames
            ]

        frame_tensors = [TF.to_tensor(img) for img in pil_frames]
        frames_batch = torch.stack(frame_tensors).to(device=device, dtype=dtype)
        frames_batch = self._normalize_for_backbone(frames_batch)
        cnn_features = self.backbone(frames_batch)

        # ---- Step 2: FPN ----
        fpn_features = self.fpn(cnn_features)

        # ---- Step 3: CLIP features ----
        clip_frames = self._preprocess_for_clip(pil_frames, device)
        cls_token, patch_tokens = self.clip_encoder(clip_frames)
        vlm_spatial = self.patch_proj(patch_tokens)

        # ---- Step 4: Encoder — 4 scales (P3, P4, P5, CLIP patches) ----
        T = len(pil_frames)
        encoder_input = fpn_features + [vlm_spatial]

        spatial_shapes_list = []
        for feat in encoder_input:
            _, _, H_i, W_i = feat.shape
            spatial_shapes_list.append([H_i, W_i])
        spatial_shapes_enc = torch.tensor(
            spatial_shapes_list, device=device, dtype=torch.long
        )

        memory, _, level_start_index, enc_valid_ratios, enc_mask_flatten = self.encoder(
            encoder_input, spatial_shapes_enc
        )

        # ---- Step 4b: Strip VLM tokens from memory ----
        clip_patch_count = self.clip_grid * self.clip_grid
        decoder_memory = memory[:, :-clip_patch_count, :]
        decoder_shapes = spatial_shapes_enc[:3]
        # Strip CLIP tokens from mask and valid_ratios too
        decoder_mask_flatten = enc_mask_flatten[:, :-clip_patch_count]
        decoder_valid_ratios = enc_valid_ratios[:, :3, :]  # 3 spatial levels only

        # ---- Step 4c: Two-stage proposal generation ----
        enc_outputs = None
        tgt = None
        query_pos = None
        init_ref_points = None

        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(
                decoder_memory, decoder_mask_flatten, decoder_shapes,
            )

            # Score proposals with full cls head (last clone = encoder head)
            enc_class_logits = self.cls_heads[-1](output_memory)   # [T, N_spatial, C]
            # Refine proposals
            enc_box_delta = self.enc_box_head(output_memory)       # [T, N_spatial, 4]
            enc_box_unact = enc_box_delta + output_proposals       # inverse-sigmoid space

            # Select top-k proposals (mean score of dim 0 across T frames)
            mean_score = enc_class_logits[..., 0].mean(dim=0)     # [N_spatial]
            topk = min(self.num_queries, mean_score.shape[0])
            _, topk_indices = mean_score.topk(topk, dim=0)

            # Gather selected proposals
            topk_coords_unact = enc_box_unact[:, topk_indices, :]  # [T, Q, 4]
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()         # [T, Q, 4]

            # Transform proposal positions to query position embeddings
            pos_embed = self.get_proposal_pos_embed(topk_coords_unact)  # [T, Q, 512]
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(pos_embed)
            )  # [T, Q, 512]

            if self.mixed_selection:
                # Content: learnable embeddings; Position: from proposals
                tgt = self.decoder.query_embed.weight.unsqueeze(0).expand(
                    T, -1, -1
                )  # [T, Q, 256]
                query_pos, _ = pos_trans_out.split(self.d_model, dim=-1)
            else:
                query_pos, tgt = pos_trans_out.split(self.d_model, dim=-1)

            init_ref_points = reference_points  # [T, Q, 4] — cx, cy, w, h

            # Encoder outputs for encoder loss
            enc_outputs = {
                "pred_logits": enc_class_logits,        # [T, N_spatial, C]
                "pred_boxes": enc_box_unact.sigmoid(),  # [T, N_spatial, 4]
            }

        # ---- Step 4d: Reshape memory back to spatial features for decoder ----
        decoder_features = []
        offset = 0
        for H, W in decoder_shapes:
            H, W = int(H), int(W)
            feat = decoder_memory[:, offset : offset + H * W, :]
            feat = feat.transpose(1, 2).view(T, self.d_model, H, W)
            decoder_features.append(feat)
            offset += H * W

        # ---- Step 5: Decoder ----
        o2o_feats, o2m_feats, pred_boxes, aux_outputs = self.decoder(
            decoder_features,
            decoder_shapes,
            clip_len=T,
            image_query=cls_token,
            tgt=tgt,
            query_pos=query_pos,
            init_ref_points=init_ref_points,
            src_valid_ratios=decoder_valid_ratios,
            src_padding_mask=decoder_mask_flatten,
        )

        # ---- Step 6: Classification heads ----
        pred_logits = self.cls_heads[-1](o2o_feats.float())

        aux_list = []
        for layer_idx, (aux_o2o, aux_o2m, aux_boxes) in enumerate(aux_outputs):
            aux_logits = self.cls_heads[layer_idx](aux_o2o.float())
            aux_entry = {
                "pred_boxes": aux_boxes,
                "pred_logits": aux_logits,
                "query_feats": aux_o2o,
                "T": T,
            }
            if self.use_ms_detr and aux_o2m is not None:
                aux_entry["pred_logits_o2m"] = self.cls_heads_o2m[layer_idx](
                    aux_o2m.float()
                )
            aux_list.append(aux_entry)

        result = {
            "pred_boxes": pred_boxes,
            "pred_logits": pred_logits,
            "query_feats": o2o_feats,
            "T": T,
            "aux_outputs": aux_list,
        }

        # O2M outputs
        if self.use_ms_detr and o2m_feats is not None:
            result["o2m_pred_logits"] = self.cls_heads_o2m[-1](o2m_feats.float())

        # Encoder outputs for encoder loss
        if enc_outputs is not None:
            result["enc_outputs"] = enc_outputs

        return result

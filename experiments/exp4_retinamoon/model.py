"""Exp4 — RetinaNet (frozen) + MoonViT/SigLIP (frozen) + Q-Former late fusion.

Components:
  RetinaNetWrapper  — frozen 3D-RetinaNet baseline producing K tube proposals
                       per clip: boxes [B, K, T, 4] (xyxy in pixel coords) and
                       spatial features [B, K, T, D_RETINA] (RoI-pooled).
  MoonViTWrapper    — frozen SigLIP-SO400M (smoke test) or Kimi-VL vision_tower.
                       Returns dict {"grid": [B,T,H_e,W_e,D_ENC],
                                     "global": [B,T,D_ENC]} — patch grid +
                       per-frame mean-pooled global scene token.
  QFormer           — N_QUERY_TOKENS=1 learned query per tube-frame; cross-attends
                       to KV bank of 6 tokens = [1 spatial + 4 RoI'd semantic +
                       1 global semantic]. Frame-index PE added to queries.
                       Output: tube embedding [B, K, T, D_MODEL] taking q[:,0].
  TemporalSelfAttn  — single nn.TransformerEncoderLayer mixing across T=8 frames
                       on the fused tube embedding, so action labels can see
                       temporal context the 2D semantic encoder lacks.
  FlatHead          — single nn.Linear -> 184 logits per tube, focal-on-all
                       (matched get GT, unmatched get all-zeros). Exp2f lesson.

This module wires the five together as RetinaMoonFusion(nn.Module). Detector and
encoder are frozen (eval() + requires_grad_(False)); only Q-Former + temporal
block + head + frame PE train.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms, roi_align

import config as C


# ---------------------------------------------------------------------------
# RetinaNet wrapper — produces tube proposals from the frozen baseline.
# ---------------------------------------------------------------------------
def _build_baseline_args():
    """Build the Namespace the baseline's build_retinanet() expects.

    Values hardcoded to match the trained-ckpt config (epoch 25, 17.76% agent
    f-mAP) — extracted from baseline's main.py defaults + ROAD-Waymo dataset
    constants. MODE='val' bypasses the kinetics weight load in backbone_models
    (we replace those weights with the trained ckpt right after build).
    """
    from argparse import Namespace
    a = Namespace()
    a.MODE = "val"
    a.ARCH = "resnet50"
    a.MODEL_TYPE = "I3D"
    a.model_subtype = "I3D"
    a.ANCHOR_TYPE = "RETINA"
    a.SEQ_LEN = 8
    a.TEST_SEQ_LEN = 8
    a.MIN_SEQ_STEP = 1
    a.MAX_SEQ_STEP = 1
    a.HEAD_LAYERS = 3
    a.NUM_FEATURE_MAPS = 5
    a.CLS_HEAD_TIME_SIZE = 3
    a.REG_HEAD_TIME_SIZE = 3
    a.MIN_SIZE = 600
    a.MAX_SIZE = 840
    a.BATCH_SIZE = 1
    a.Domain_Adaptation = False
    a.FBN = True
    a.FREEZE_UPTO = 1
    a.MULTI_GPUS = False
    a.head_size = 256
    # ROAD-Waymo class structure (matches our flat 184-dim).
    a.num_classes = C.NUM_CLASSES                    # 184
    a.num_classes_list = C.NUM_CLASSES_LIST          # [1,10,22,16,49,86]
    a.num_ego_classes = 6                            # ROAD-Waymo ego count (from ckpt shape)
    # backbone_models() reads model_perms/model_3d_layers in build path.
    a.model_perms = [3, 4, 6, 3]
    a.model_3d_layers = [[0, 1, 2], [0, 2], [0, 2, 4], [0, 1]]
    a.non_local_inds = [[], [], [], []]
    return a


class RetinaNetWrapper(nn.Module):
    """Frozen 3D-RetinaNet producing K tube proposals per clip.

    Loads the 17.76% baseline checkpoint, runs in eval mode under no_grad,
    captures FPN P3 via a forward hook for RoI-pooled per-tube features,
    selects top-K tubes by mean agentness across T.

    Output:
        boxes:        [B, K, T, 4]   xyxy in image pixels
        scores:       [B, K]         mean agentness across T per tube
        feat_spatial: [B, K, T, D_RETINA]   RoI-pooled P3 per frame
        logits_tube:  [B, K, T, 184] raw flat classification logits per tube
    """

    def __init__(self, ckpt_path: str, top_k: int = 100, d_retina: int = 256,
                 conf_thresh: float = 0.025, nms_thresh: float = 0.5):
        super().__init__()
        self.top_k = top_k
        self.d_retina = d_retina
        # Replicate the baseline's detection-dump path (filter_detections_for_dumping):
        # GEN_CONF_THRESH=0.025, GEN_NMS=0.5, GEN_TOPK=100. Without NMS the raw top-K
        # anchors collapse onto ~7 objects (87% dup), flooring recall/f-mAP.
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        # Bootstrap baseline imports. The baseline's models/ pkg uses relative
        # imports, so we add its parent dir to sys.path.
        import sys
        baseline_root = "/data/repos/PedestrianIntent++/ROAD_plus_plus_Baseline"
        if baseline_root not in sys.path:
            sys.path.insert(0, baseline_root)
        from models.retinanet import build_retinanet  # noqa: E402

        args = _build_baseline_args()
        net = build_retinanet(args)

        # Load the 17.76% checkpoint (CPU first to avoid double-allocation).
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
              for k, v in sd.items()}
        missing, unexpected = net.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[RetinaNetWrapper] missing={len(missing)} unexpected={len(unexpected)}")
            if missing[:3]:    print(f"  missing[:3]:    {missing[:3]}")
            if unexpected[:3]: print(f"  unexpected[:3]: {unexpected[:3]}")

        self.net = net
        # Capture P3 via hook so we can RoI-pool spatial features at the tubes.
        self._sources_cache: list[torch.Tensor] = []

        def _hook(_mod, _inp, out):
            # backbone returns (sources, ego_feat); sources = [p3,p4,p5,p6,p7]
            self._sources_cache.append(out[0])

        self._hook_handle = self.net.backbone.register_forward_hook(_hook)

        # Freeze everything.
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, clip: torch.Tensor) -> Dict[str, torch.Tensor]:
        """clip: [B, T, 3, H, W] normalized RGB (will be transposed to [B,3,T,H,W])."""
        B, T, _, H, W = clip.shape
        x = clip.permute(0, 2, 1, 3, 4).contiguous()    # baseline expects [B,C,T,H,W]

        self._sources_cache.clear()
        decoded, flat_conf, _ego = self.net(x)
        sources = self._sources_cache[-1]               # tuple of feature maps; index 0 = P3
        p3 = sources[0]                                 # [B, D_RETINA, T_p3, H_p3, W_p3]

        agentness = flat_conf[..., 0].sigmoid()         # [B, T, A_total]
        tube_scores = agentness.mean(dim=1)             # tube score = mean agentness over T
        # Baseline-faithful selection: conf-thresh -> class-agnostic NMS -> top-K,
        # over ALL anchors. NMS must precede the top-K cut (post-cut dedup cannot
        # recover distinct objects ranked below the duplicates). NMS uses a per-anchor
        # representative box = mean decoded box over T (tube-level dedup), scored by
        # mean agentness, matching the baseline's class-agnostic agentness NMS.
        A = tube_scores.shape[-1]
        rep_boxes = decoded.mean(dim=1)                 # [B, A, 4] xyxy pixel coords
        idx_rows, score_rows = [], []
        for b in range(B):
            s = tube_scores[b]                          # [A]
            keep = (s > self.conf_thresh).nonzero(as_tuple=False).squeeze(1)
            if keep.numel() == 0:                       # degenerate frame: fall back to top-K
                keep = torch.topk(s, k=min(self.top_k, A)).indices
            nk = nms(rep_boxes[b][keep], s[keep], self.nms_thresh)   # score-sorted survivors
            sel = keep[nk][: self.top_k]                 # [<=K] unique anchors, score-desc
            sc = s[sel]
            if sel.numel() < self.top_k:                 # pad to K to keep [B,K,T,4] contract
                pad = self.top_k - sel.numel()
                last = sel[-1:] if sel.numel() else torch.zeros(1, dtype=torch.long, device=s.device)
                sel = torch.cat([sel, last.expand(pad)])
                sc = torch.cat([sc, torch.zeros(pad, device=s.device)])   # pad score=0 -> ranks last
            idx_rows.append(sel); score_rows.append(sc)
        idx = torch.stack(idx_rows, 0)                  # [B, K]
        scores = torch.stack(score_rows, 0)             # [B, K]

        # Gather per-frame boxes at those anchor indices.
        # baseline `decode()` returns boxes in **pixel xyxy** in the input image's
        # coord frame (anchors are stride × cell coords, not normalized).
        idx_exp = idx[:, None, :, None].expand(B, T, idx.shape[1], 4)
        boxes = torch.gather(decoded, dim=2, index=idx_exp).permute(0, 2, 1, 3).contiguous()

        # Per-tube raw classification logits (184-dim flat) at the same anchors.
        # Exp6's fusion head consumes these; padded tubes (score 0) carry the
        # duplicated anchor's logits, same contract as `boxes`.
        Ccls = flat_conf.shape[-1]
        idx_exp_c = idx[:, None, :, None].expand(B, T, idx.shape[1], Ccls)
        logits_tube = torch.gather(flat_conf, dim=2, index=idx_exp_c).permute(0, 2, 1, 3).contiguous()

        # RoI-pool P3 at each per-frame tube box for D_RETINA-dim spatial feature.
        _, D, Tp3, Hp3, Wp3 = p3.shape
        if Tp3 != T:
            p3 = F.interpolate(p3, size=(T, Hp3, Wp3), mode="trilinear", align_corners=False)
        sx = Wp3 / float(W)
        sy = Hp3 / float(H)
        boxes_p3 = boxes.clone()
        boxes_p3[..., 0::2] *= sx
        boxes_p3[..., 1::2] *= sy
        p3_btchw = p3.permute(0, 2, 1, 3, 4).contiguous().view(B * T, D, Hp3, Wp3)
        K = boxes_p3.shape[1]
        boxes_btK = boxes_p3.permute(0, 2, 1, 3).contiguous().view(B * T, K, 4)
        frame_ids = torch.arange(B * T, device=clip.device).view(B * T, 1, 1).expand(B * T, K, 1)
        rois = torch.cat([frame_ids.float(), boxes_btK], dim=-1).view(B * T * K, 5)
        pooled = roi_align(p3_btchw, rois, output_size=(1, 1),
                           spatial_scale=1.0, aligned=True)               # [B*T*K, D, 1, 1]
        feat_spatial = pooled.view(B, T, K, D).permute(0, 2, 1, 3).contiguous()  # [B,K,T,D]
        return {"boxes": boxes, "scores": scores, "feat_spatial": feat_spatial,
                "logits_tube": logits_tube}


# ---------------------------------------------------------------------------
# Vision encoder wrapper — MoonViT (Kimi-VL vision_tower) at native resolution.
# ---------------------------------------------------------------------------
class MoonViTWrapper(nn.Module):
    """Frozen MoonViT (Kimi-VL vision_tower) — NVIDIA LocateAnything-style usage.

    Loads ONLY the vision_tower weights from the single shard that contains
    them (model-00001-of-00007.safetensors — we never download the 6 LLM shards).

    Forward expects clip [B, T, 3, H, W] at the **native** input resolution
    (no resize). MoonViT's 2D RoPE handles any (H, W) and its built-in
    pixel-shuffle 2×2 patch_merger bundles 4 fine patches per output token.

    Output (dict):
        grid:   [B, T, H_m, W_m, ENCODER_TOKEN_DIM=4608]
                where H_m = H // (patch=14 × merge=2), W_m likewise.
                Each cell holds 4 sub-patches flattened to a 4608-dim vector.
        global: [B, T, ENCODER_TOKEN_DIM]  mean-pooled scene token.

    Future Qwen-as-decoder hook: the same `grid` output, optionally
    flattened to (B, T*H_m*W_m, 4608), is exactly what a Qwen2.5
    multi-modal projector consumes. No wrapper change needed when we add it.
    """

    def __init__(self, arch_repo: str, weights_repo: str, weights_file: str,
                 weights_prefix: str, hidden: int, patch: int = 14, merge: int = 2):
        super().__init__()
        self.arch_repo    = arch_repo
        self.weights_repo = weights_repo
        self.hidden = hidden
        self.patch  = patch
        self.merge  = merge
        self.token_dim = hidden * (merge ** 2)   # post-merger token dim, 4 * 1152 = 4608

        # Trust-remote-code instantiation of MoonVitPretrainedModel using the
        # standalone MoonViT-SO-400M modeling file (clean, no LLM cruft).
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        config = AutoConfig.from_pretrained(arch_repo, trust_remote_code=True)
        # Auto-pick the strongest attention impl: flash_attention_2 > sdpa > eager.
        try:
            import flash_attn  # noqa: F401
            config._attn_implementation = "flash_attention_2"
        except ImportError:
            config._attn_implementation = "sdpa"
        self.attn_impl = config._attn_implementation
        MoonVitPretrainedModel = get_class_from_dynamic_module(
            "modeling_moonvit.MoonVitPretrainedModel",
            arch_repo,
        )
        self.encoder = MoonVitPretrainedModel(config)

        # Load NVIDIA LocateAnything's continual-pretrained vision_model weights.
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        shard = hf_hub_download(weights_repo, weights_file)
        sd_full = load_file(shard)
        sd_vt = {
            k.replace(weights_prefix, "", 1): v
            for k, v in sd_full.items() if k.startswith(weights_prefix)
        }
        missing, unexpected = self.encoder.load_state_dict(sd_vt, strict=True)
        print(f"[MoonViTWrapper] {weights_repo} → loaded {len(sd_vt)} weights "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
        del sd_full, sd_vt

        # Cast to bf16 (encoder was trained in bf16; HF defaults to fp32 init).
        self.encoder = self.encoder.to(torch.bfloat16)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

        # MoonViT was trained with SigLIP-style normalization (mean/std=0.5).
        self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, clip: torch.Tensor) -> Dict[str, torch.Tensor]:
        """clip: [B, T, 3, H, W] in [0,1], NORMALIZED INSIDE (since SigLIP mean/std
        differs from the ImageNet means already applied by our dataloader).

        Returns dict {"grid": [B, T, H_m, W_m, token_dim], "global": [B, T, token_dim]}.
        """
        B, T, _, H, W = clip.shape
        # Undo dataloader's ImageNet normalization, then apply SigLIP normalization.
        # dataloader applied: t = (raw - mean_in) / std_in
        # we want:            t' = (raw - mean_sl) / std_sl
        # Compose: t' = (t * std_in + mean_in - mean_sl) / std_sl
        # But this is a frozen wrapper kept clean: we just re-do it.
        # Cleanest: cast back to [0, 1], apply SigLIP normalization.
        mean_in = torch.tensor([0.485, 0.456, 0.406], device=clip.device).view(1, 1, 3, 1, 1)
        std_in  = torch.tensor([0.229, 0.224, 0.225], device=clip.device).view(1, 1, 3, 1, 1)
        raw01 = clip * std_in + mean_in            # [B, T, 3, H, W] in ~[0, 1]
        flat = raw01.view(B * T, 3, H, W)
        flat = (flat - self.mean) / self.std
        # Truncate to multiples of (patch * merge) so the patch_merger reshape works.
        cell = self.patch * self.merge             # 28
        new_H = (H // cell) * cell
        new_W = (W // cell) * cell
        if new_H != H or new_W != W:
            flat = flat[:, :, :new_H, :new_W]

        # Patching strategy depends on attention impl:
        #  - flash_attention_2: pack all B*T frames into one packed sequence
        #    (cu_seqlens restricts attention to within-frame tokens, O(L) memory).
        #  - sdpa / eager: process per-frame (sdpa builds an (L,L) mask before
        #    SDPA call, so packed input would OOM at L=20K).
        p = self.patch
        n_h, n_w = new_H // p, new_W // p
        B_eff = B * T
        m_h, m_w = n_h // self.merge, n_w // self.merge

        # [B*T, 3, n_h, p, n_w, p] -> [B*T, n_h, n_w, 3, p, p] -> [B*T, n_h*n_w, 3, p, p]
        patches_per_frame = flat.reshape(B_eff, 3, n_h, p, n_w, p).permute(0, 2, 4, 1, 3, 5).contiguous()
        patches_per_frame = patches_per_frame.reshape(B_eff, n_h * n_w, 3, p, p).to(torch.bfloat16)

        if self.attn_impl == "flash_attention_2":
            # Pack all frames into one sequence — single kernel launch.
            patches = patches_per_frame.reshape(B_eff * n_h * n_w, 3, p, p)
            grid_hws = torch.tensor([[n_h, n_w]] * B_eff, device=clip.device, dtype=torch.long)
            out_list = self.encoder(pixel_values=patches, grid_hws=grid_hws)
            merged = torch.stack(out_list, dim=0)
        else:
            # Per-frame fallback for sdpa/eager (mask is per-frame O(L²) → fits).
            grid_hws_single = torch.tensor([[n_h, n_w]], device=clip.device, dtype=torch.long)
            merged_list = []
            for i in range(B_eff):
                out_list = self.encoder(
                    pixel_values=patches_per_frame[i], grid_hws=grid_hws_single,
                )
                merged_list.append(out_list[0])
            merged = torch.stack(merged_list, dim=0)

        grid = merged.reshape(B, T, m_h, m_w, self.merge * self.merge * self.hidden).float()
        feat_global = grid.mean(dim=(2, 3))         # [B, T, token_dim]
        return {"grid": grid, "global": feat_global}


# ---------------------------------------------------------------------------
# Q-Former fusion block — single learned query cross-attends to KV bank of 6.
# ---------------------------------------------------------------------------
class QFormerLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ffn: int, dropout: float):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q: [N, Lq, D]   kv: [N, Lkv, D]
        q  = self.norm1(q + self.drop(self.self_attn(q, q, q, need_weights=False)[0]))
        q  = self.norm2(q + self.drop(self.cross_attn(q, kv, kv, need_weights=False)[0]))
        q  = self.norm3(q + self.drop(self.ffn(q)))
        return q


def _sinusoidal_pe(length: int, d_model: int) -> torch.Tensor:
    """Standard sinusoidal positional embedding, shape [length, d_model]."""
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                         * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class QFormer(nn.Module):
    """Q-Former with N_QUERY learned queries + 6-token KV bank.

    KV bank per tube-frame = [1 spatial(P3 RoI) + M=4 semantic(MoonViT RoI) + 1 global(scene)]
    Queries get a frame-index positional embedding added before layer 0.
    Output: q[:, 0] (CLS-style) — no mean-pool, since (a) N_QUERY=1 makes pool a
    no-op and (b) if N>1 we want specialization, not averaging.
    """

    def __init__(self, d_model: int, n_query: int, n_layers: int,
                 nhead: int, d_ffn: int, dropout: float,
                 d_retina: int, d_encoder: int,
                 clip_len: int = 8):
        super().__init__()
        self.n_query = n_query
        self.d_model = d_model
        self.clip_len = clip_len
        # Project frozen features into the fusion model dim.
        self.proj_spatial = nn.Linear(d_retina,  d_model)
        self.proj_enc     = nn.Linear(d_encoder, d_model)   # used for RoI patches + global
        # Learned queries (shared across tubes, broadcast at runtime).
        self.queries = nn.Parameter(torch.randn(1, n_query, d_model) * 0.02)
        # Frame-index PE: sinusoidal, registered as a buffer (non-trainable).
        # Added to queries so per-frame predictions can distinguish their t.
        self.register_buffer("frame_pe", _sinusoidal_pe(clip_len, d_model), persistent=False)
        self.layers = nn.ModuleList(
            QFormerLayer(d_model, nhead, d_ffn, dropout) for _ in range(n_layers)
        )

    def forward(self, feat_spatial: torch.Tensor, feat_enc_roi: torch.Tensor,
                feat_enc_global: torch.Tensor) -> torch.Tensor:
        """
        feat_spatial:    [B, K, T, D_RETINA]   per-tube spatial feature
        feat_enc_roi:    [B, K, T, M, D_ENC]    per-tube RoI'd semantic patches
        feat_enc_global: [B, T, D_ENC]          per-frame mean-pooled semantic
        returns:         [B, K, T, D_MODEL]     fused tube embedding (q[:, 0])
        """
        B, K, T, _ = feat_spatial.shape
        M = feat_enc_roi.shape[3]
        N = B * K * T

        s = self.proj_spatial(feat_spatial).view(N, 1, self.d_model)         # [N,1,D]
        e = self.proj_enc(feat_enc_roi).view(N, M, self.d_model)             # [N,M,D]
        # Global token: broadcast over K tubes that share each (b, t) frame.
        g = self.proj_enc(feat_enc_global)                                   # [B,T,D]
        g = g.unsqueeze(1).expand(B, K, T, self.d_model).contiguous()        # [B,K,T,D]
        g = g.view(N, 1, self.d_model)                                       # [N,1,D]
        kv = torch.cat([s, e, g], dim=1)                                     # [N, 1+M+1, D]

        # Queries with frame-index PE.
        # N indexes flatten in order [b, k, t] (matches feat_spatial.view above);
        # for entry n the frame is t = n % T.
        q = self.queries.expand(N, -1, -1).contiguous()                      # [N, Lq, D]
        t_idx = torch.arange(N, device=q.device) % T                         # [N]
        frame_emb = self.frame_pe[t_idx].unsqueeze(1)                        # [N, 1, D]
        q = q + frame_emb                                                    # broadcasts over Lq

        for layer in self.layers:
            q = layer(q, kv)
        # CLS-style output: take the first query token, no pool.
        fused = q[:, 0]                                                      # [N, D]
        return fused.view(B, K, T, self.d_model)


# ---------------------------------------------------------------------------
# Temporal self-attention — mixes T=8 frames per tube after Q-Former fusion.
# ---------------------------------------------------------------------------
class TemporalSelfAttn(nn.Module):
    """Single TransformerEncoderLayer over T=8 per tube.

    Restores temporal mixing for the semantic branch. The frozen RetinaNet's
    I3D backbone has temporal context built in; the per-frame 2D semantic
    features from MoonViT do not. Action labels (Moving-Away, Stopping) need
    this mixing. Keeps per-frame outputs intact for per-frame f-mAP eval.
    """

    def __init__(self, d_model: int, nhead: int, d_ffn: int, dropout: float,
                 clip_len: int = 8):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ffn,
            dropout=dropout, batch_first=True, norm_first=False,
        )
        # Frame PE (separate from the Q-Former's — different role, same scheme).
        self.register_buffer("frame_pe", _sinusoidal_pe(clip_len, d_model),
                             persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, K, T, D]  ->  [B, K, T, D]"""
        B, K, T, D = x.shape
        x = x.view(B * K, T, D)
        x = x + self.frame_pe.unsqueeze(0)                # broadcast over [B*K]
        x = self.layer(x)
        return x.view(B, K, T, D)


# ---------------------------------------------------------------------------
# Output head — flat 184-dim per tube (focal-on-all applied in train.py).
# ---------------------------------------------------------------------------
class FlatHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int = 184):
        super().__init__()
        self.cls = nn.Linear(d_model, num_classes)
        # Focal-friendly bias init (matches RetinaNet convention).
        prior = 0.01
        nn.init.constant_(self.cls.bias, -math.log((1 - prior) / prior))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, K, T, D]  ->  [B, K, T, 184]"""
        return self.cls(x)


# ---------------------------------------------------------------------------
# Top-level model: wires detector → encoder → RoIAlign → Q-Former → Temp → head.
# ---------------------------------------------------------------------------
class RetinaMoonFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.detector = RetinaNetWrapper(
            ckpt_path=C.RETINANET_CKPT,
            top_k=C.RETINANET_TOPK_TUBES,
            d_retina=C.RETINANET_SPATIAL_DIM,
        )
        self.encoder = MoonViTWrapper(
            arch_repo     = C.ENCODER_ARCH_REPO,
            weights_repo  = C.ENCODER_WEIGHTS_REPO,
            weights_file  = C.ENCODER_WEIGHTS_FILE,
            weights_prefix= C.ENCODER_WEIGHTS_PREFIX,
            hidden        = C.ENCODER_HIDDEN,         # MoonViT internal hidden (1152)
            patch         = C.ENCODER_PATCH,
            merge         = C.ENCODER_MERGE,
        )
        self.fusion = QFormer(
            d_model=C.D_MODEL,
            n_query=C.N_QUERY_TOKENS,
            n_layers=C.NUM_FUSION_LAYERS,
            nhead=C.NHEAD,
            d_ffn=C.DIM_FFN,
            dropout=C.DROPOUT,
            d_retina=C.RETINANET_SPATIAL_DIM,
            d_encoder=C.ENCODER_TOKEN_DIM,   # 4608 (post-merger token dim)
            clip_len=C.CLIP_LEN,
        )
        self.temporal = TemporalSelfAttn(
            d_model=C.D_MODEL,
            nhead=C.NHEAD,
            d_ffn=C.D_MODEL * 2,                # 256 -> 512 (lighter than fusion FFN)
            dropout=C.DROPOUT,
            clip_len=C.CLIP_LEN,
        ) if C.USE_TEMPORAL_SELFATTN else nn.Identity()
        self.head = FlatHead(C.D_MODEL, C.NUM_CLASSES)

    def forward(self, clip: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        clip: [B, T, 3, H, W] normalized RGB.
        returns:
            logits: [B, K, T, 184]
            boxes:  [B, K, T, 4]
            scores: [B, K]
        """
        B, T, _, H, W = clip.shape
        det = self.detector(clip)
        boxes        = det["boxes"]                # [B,K,T,4]
        scores       = det["scores"]               # [B,K]
        feat_spatial = det["feat_spatial"]         # [B,K,T,D_RETINA]

        enc = self.encoder(clip)                   # dict {grid, global}
        feat_enc_grid   = enc["grid"]              # [B,T,H_e,W_e,D_ENC]
        feat_enc_global = enc["global"]            # [B,T,D_ENC]
        K = boxes.shape[1]
        H_e, W_e = feat_enc_grid.shape[2], feat_enc_grid.shape[3]

        # RoI-Align MoonViT patches at each tube box, per frame. M=4 (2x2 pool).
        feat_enc_btchw = feat_enc_grid.permute(0, 1, 4, 2, 3).contiguous()  # [B,T,D,H_e,W_e]
        sx = W_e / float(W)
        sy = H_e / float(H)
        boxes_enc = boxes.clone()
        boxes_enc[..., 0::2] *= sx
        boxes_enc[..., 1::2] *= sy
        boxes_btK = boxes_enc.permute(0, 2, 1, 3).contiguous().view(B * T, K, 4)
        frame_ids = torch.arange(B * T, device=clip.device).view(B * T, 1, 1).expand(B * T, K, 1)
        rois = torch.cat([frame_ids.float(), boxes_btK], dim=-1).view(B * T * K, 5)
        feat_map = feat_enc_btchw.view(B * T, C.ENCODER_TOKEN_DIM, H_e, W_e)
        pooled = roi_align(feat_map, rois, output_size=(2, 2),
                           spatial_scale=1.0, aligned=True)              # [B*T*K, D, 2, 2]
        M = 4
        # Reorder spatial dims into the token dim: [B, K, T, M, D_ENC].
        pooled = pooled.view(B, T, K, C.ENCODER_TOKEN_DIM, M).permute(0, 2, 1, 4, 3).contiguous()

        fused = self.fusion(feat_spatial, pooled, feat_enc_global)        # [B,K,T,D]
        fused = self.temporal(fused)                                       # [B,K,T,D]
        logits = self.head(fused)                                          # [B,K,T,184]
        return {"logits": logits, "boxes": boxes, "scores": scores}

"""
Experiment 1b model (redesign): FCOS-style dense detection head.

Architecture change from original Exp1b:
  - Removes ROIAveragePool and TubeLinkingModule (no GT boxes at inference)
  - Every spatial ViT token independently predicts:
      agentness : sigmoid(Linear(D→1))   — is there an agent here?
      box       : Linear(D→4)            — FCOS (l,t,r,b) to GT box edges (normalized)
      agent     : sigmoid(Linear(D→10))
      action    : sigmoid(Linear(D→22))
      loc       : sigmoid(Linear(D→16))
      duplex    : sigmoid(Linear(D→49))
      triplet   : sigmoid(Linear(D→86))
  - Agentness is a real learned score fed into the t-norm flat vector
    (replaces the hardcoded 1.0 from oracle-box Exp1b)

Warm-start from Exp1 best.pt (strict=False):
  - ViT weights: load cleanly (same architecture)
  - heads.agent / action / loc / duplex / triplet: load cleanly (same Linear dims)
  - heads.agentness, heads.box: initialize fresh (not in Exp1 ckpt)
  - roi_pool.*, tube_link.*: in Exp1 ckpt but not in this model → ignored

LoRA workflow (same as original Exp1b):
  1. Instantiate with freeze_vit=True
  2. Load Exp1 best.pt
  3. Call model.add_lora() → adds adapters, preserves base ViT weights
  4. Two param groups: LoRA lr=5e-5, heads lr=1e-4
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


# ── ViT extractor with LoRA (identical to original Exp1b) ─────────────────────

class QwenViTExtractor(nn.Module):
    """
    Qwen2.5-VL vision encoder. Initially frozen for clean weight loading,
    then LoRA adapters are added via add_lora() before training begins.
    """

    def __init__(self, model_id: str, freeze: bool = True):
        super().__init__()
        from transformers import Qwen2_5_VLForConditionalGeneration
        full = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        self.visual = full.model.visual
        del full

        if freeze:
            for p in self.visual.parameters():
                p.requires_grad_(False)

    def add_lora(
        self,
        r:               int   = 8,
        lora_alpha:      int   = 16,
        lora_dropout:    float = 0.05,
        target_modules:  list  = None,
        n_layers:        int   = 8,
    ) -> None:
        """
        Wrap the first n_layers ViT blocks with LoRA adapters via regex targeting.
        Only lora_A / lora_B params will have requires_grad=True after this call.
        """
        from peft import LoraConfig, get_peft_model

        block_range  = "|".join(str(i) for i in range(n_layers))
        attn_modules = "|".join(target_modules or ["qkv", "proj"])
        regex = rf"blocks\.({block_range})\.attn\.({attn_modules})"

        lora_config = LoraConfig(
            r              = r,
            lora_alpha     = lora_alpha,
            lora_dropout   = lora_dropout,
            target_modules = regex,
            bias           = "none",
            inference_mode = False,
        )
        self.visual = get_peft_model(self.visual, lora_config)

        n_lora  = sum(p.numel() for p in self.visual.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.visual.parameters())
        print(f"  LoRA added: {n_lora:,} trainable / {n_total:,} total ViT params "
              f"({100 * n_lora / n_total:.2f}%)")

    def lora_parameters(self):
        """Return only LoRA adapter parameters (requires_grad=True)."""
        return [p for p in self.visual.parameters() if p.requires_grad]

    def forward(
        self,
        pixel_values:   torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Returns list of T tensors [H'_t, W'_t, D]."""
        out    = self.visual(pixel_values, image_grid_thw)
        merged = out.pooler_output   # [total_merged_tokens, D]

        # spatial_merge_size lives on the base model; peft wraps under base_model.model
        base = getattr(self.visual, "base_model", self.visual)
        base = getattr(base, "model", base)
        m    = base.spatial_merge_size

        frame_feats: List[torch.Tensor] = []
        offset = 0
        for row in image_grid_thw:
            t, h, w  = int(row[0]), int(row[1]), int(row[2])
            h_m, w_m = h // m, w // m
            n        = t * h_m * w_m
            chunk    = merged[offset: offset + n].view(t, h_m, w_m, -1)
            frame_feats.append(chunk.squeeze(0))   # [H'_t, W'_t, D]
            offset  += n

        return frame_feats


# ── Dense detection heads ──────────────────────────────────────────────────────

class DetectionHeads(nn.Module):
    """
    Seven linear heads applied per spatial token.

    Input:  [N, D]  (N = T * H' * W' flattened tokens)
    Output: dict of per-token predictions
    """

    def __init__(
        self,
        d_in:       int = 1280,
        n_agents:   int = 10,
        n_actions:  int = 22,
        n_locs:     int = 16,
        n_duplexes: int = 49,
        n_triplets: int = 86,
    ):
        super().__init__()
        self.agentness = nn.Linear(d_in, 1)
        self.box       = nn.Linear(d_in, 4)        # FCOS (l,t,r,b) distances, normalized
        self.agent     = nn.Linear(d_in, n_agents)
        self.action    = nn.Linear(d_in, n_actions)
        self.loc       = nn.Linear(d_in, n_locs)
        self.duplex    = nn.Linear(d_in, n_duplexes)
        self.triplet   = nn.Linear(d_in, n_triplets)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: [N, D]
        Returns:
            dict:
              agentness [N, 1]   — sigmoid probability of foreground agent
              box       [N, 4]   — raw ltrb distances (no activation; SmoothL1 target)
              agent     [N, 10]  — sigmoid multi-label
              action    [N, 22]
              loc       [N, 16]
              duplex    [N, 49]
              triplet   [N, 86]
        """
        x = x.float()
        return {
            "agentness": torch.sigmoid(self.agentness(x)),
            "box":       self.box(x),
            "agent":     torch.sigmoid(self.agent(x)),
            "action":    torch.sigmoid(self.action(x)),
            "loc":       torch.sigmoid(self.loc(x)),
            "duplex":    torch.sigmoid(self.duplex(x)),
            "triplet":   torch.sigmoid(self.triplet(x)),
        }


# ── Full model ─────────────────────────────────────────────────────────────────

class QwenROADModel(nn.Module):
    """
    Experiment 1b full model — FCOS dense detection.

    No GT boxes required at forward time: every spatial token predicts
    independently. FCOS-style assignment happens in train.py/assign.py.

    forward(pixel_values, image_grid_thw) → preds dict
    """

    def __init__(
        self,
        model_id:   str,
        d_model:    int  = 1280,
        freeze_vit: bool = True,
        tube_heads: int  = 8,    # retained for call-site compatibility; unused
    ):
        super().__init__()
        self.vit   = QwenViTExtractor(model_id, freeze=freeze_vit)
        self.heads = DetectionHeads(d_model)

    def add_lora(self, **kwargs) -> None:
        self.vit.add_lora(**kwargs)

    def head_parameters(self):
        """Return all detection head parameters (for optimizer param group)."""
        return list(self.heads.parameters())

    def forward(
        self,
        pixel_values:   torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> dict:
        """
        Args:
            pixel_values:   [total_patches, C*pH*pW]  from Qwen image_processor
            image_grid_thw: [T, 3]  one row per frame: [1, H', W']

        Returns:
            preds dict:
              agentness     [T*H'*W', 1]
              box           [T*H'*W', 4]
              agent         [T*H'*W', 10]
              action        [T*H'*W', 22]
              loc           [T*H'*W', 16]
              duplex        [T*H'*W', 49]
              triplet       [T*H'*W', 86]
              token_counts  list[int]  — H'_t * W'_t per frame (for assignment)
              frame_shapes  list[(H'_t, W'_t)]
        """
        vit_feats = self.vit(pixel_values, image_grid_thw)   # list of T [H', W', D]

        token_counts = [f.shape[0] * f.shape[1] for f in vit_feats]
        frame_shapes = [(f.shape[0], f.shape[1]) for f in vit_feats]

        all_tokens = torch.cat(
            [f.reshape(-1, f.shape[-1]) for f in vit_feats], dim=0
        )   # [T*H'*W', D]

        preds = self.heads(all_tokens)
        preds["token_counts"] = token_counts
        preds["frame_shapes"] = frame_shapes
        return preds

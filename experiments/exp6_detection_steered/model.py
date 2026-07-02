"""Exp6 model — DetectionSteeredFusion: the "few layer to be trained".

Moradi 2026-06-02: "use language embedding to fused with output of 3d retina
net and then re predict with adding few layer to be trained". This is
architecture (a) of his 2026-06-11 note (new trainable layers on top of the
fused representation; (b) — training RetinaNet's remaining post-fusion layers
— is the sibling variant, not built here).

Three input tracks per (frame, box) sample, each linearly projected to D_MODEL:
  det_logits [184]   frozen RetinaNet flat classification logits
  struct     [54]    vectorized Qwen fields (agent/actions/locations/risk)
  rationale  [1153]  frozen SigLIP text embedding of the rationale (+has flag)

Fusion: concat(3 x D_MODEL) → LayerNorm → Linear → GELU → Dropout → Linear
→ delta [184];   output logits = det_logits + delta  (residual).

Floor guarantee: the delta layer is zero-initialized, so at init the model
reproduces the detector's logits exactly — training starts AT the
detector-over-top-K control and gradient descent moves it from there.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config as C


class DetectionSteeredFusion(nn.Module):
    def __init__(self):
        super().__init__()
        d = C.D_MODEL
        self.det_proj = nn.Linear(C.NUM_CLASSES, d)
        self.struct_proj = nn.Linear(C.STRUCT_DIM, d)
        self.rat_proj = nn.Linear(C.RATIONALE_DIM + 1, d)
        self.head = nn.Sequential(
            nn.LayerNorm(3 * d),
            nn.Linear(3 * d, C.D_FFN),
            nn.GELU(),
            nn.Dropout(C.DROPOUT),
            nn.Linear(C.D_FFN, C.NUM_CLASSES),
        )
        # Zero-init the delta layer → identity at init (floor guarantee).
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, det_logits: torch.Tensor, struct: torch.Tensor,
                rationale: torch.Tensor) -> torch.Tensor:
        """All inputs [B, ·] → fused flat logits [B, 184]."""
        z = torch.cat([
            self.det_proj(det_logits),
            self.struct_proj(struct),
            self.rat_proj(rationale),
        ], dim=-1)
        return det_logits + self.head(z)

"""Frozen CLIP ViT-L/14 feature extractor + patch token projection.

Extracts two feature types from a frozen CLIP visual encoder:
1. CLS token (768-dim) — scene-level descriptor, used as per-layer decoder query
2. Patch tokens (1024-dim, 24×24) — spatial features, used as extra encoder scale

Reference: Frozen-DETR (Fu et al., NeurIPS 2024)
  - CLIP integration: DINO-coco/models/dino/dino.py:195-257
  - CLIP model: DINO-coco/myclip/model.py:288-306
  - Patch projection: DINO-coco/models/dino/dino.py:201-203
"""

from __future__ import annotations

import hashlib
import os
import urllib
import warnings
from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import (
    CenterCrop, Compose, Normalize, Resize, ToTensor,
)

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# CLIP normalization constants
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_CLIP_URLS = {
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}


# ---------------------------------------------------------------------------
# Minimal CLIP ViT building blocks (from OpenAI CLIP, via Frozen-DETR myclip/)
# ---------------------------------------------------------------------------

class LayerNorm(nn.LayerNorm):
    """LayerNorm that handles fp16 inputs by casting to fp32 internally."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), need_weights=False)[0]
        x = x + self.mlp(self.ln_2(x))
        return x


class CLIPTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads) for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x


class CLIPVisionTransformer(nn.Module):
    """CLIP ViT that returns both CLS token and patch tokens.

    For ViT-L/14@336px:
      - width = 1024, layers = 24, heads = 16
      - patch_size = 14, input_resolution = 336
      - grid = 336 // 14 = 24 → 576 patches
      - output_dim (embed_dim) = 768
      - CLS token: [B, 768] (after ln_post + projection)
      - Patch tokens: [B, 1024, 24, 24] (raw width, reshaped to spatial grid)
    """

    def __init__(self, input_resolution: int, patch_size: int, width: int,
                 layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.width = width
        self.grid_size = input_resolution // patch_size  # 24 for 336/14

        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size,
                               stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = LayerNorm(width)
        self.transformer = CLIPTransformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 3, input_resolution, input_resolution]
        Returns:
            cls_token:    [B, output_dim] (768 for ViT-L/14)
            patch_tokens: [B, width, grid, grid] (1024, 24, 24 for ViT-L/14@336)
        """
        # Ref: myclip/model.py:288-306
        x = self.conv1(x)                                    # [B, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)           # [B, width, grid**2]
        x = x.permute(0, 2, 1)                               # [B, grid**2, width]

        # Prepend CLS token
        cls = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls, x], dim=1)                       # [B, grid**2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        # Transformer: NLD -> LND -> NLD
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)                               # [B, grid**2 + 1, width]

        # CLS: apply ln_post + projection → [B, output_dim]
        # Ref: myclip/model.py:301,303-304
        cls_token = self.ln_post(x[:, 0, :])
        cls_token = cls_token @ self.proj                     # [B, output_dim=768]

        # Patches: raw width, reshaped to spatial grid
        # Ref: myclip/model.py:300, dino.py:381
        patch_tokens = x[:, 1:, :]                            # [B, grid**2, width=1024]
        g = self.grid_size
        patch_tokens = patch_tokens.permute(0, 2, 1).reshape(
            x.shape[0], self.width, g, g
        )                                                     # [B, 1024, 24, 24]

        return cls_token, patch_tokens


# ---------------------------------------------------------------------------
# Model loading (adapted from myclip/clip.py)
# ---------------------------------------------------------------------------

def _download_clip(url: str, root: str) -> str:
    """Download CLIP checkpoint if not cached."""
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        warnings.warn(f"SHA256 mismatch for {download_target}; re-downloading")

    print(f"Downloading CLIP to {download_target} ...")
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as out:
        while True:
            buf = source.read(8192)
            if not buf:
                break
            out.write(buf)

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("SHA256 mismatch after download")
    return download_target


def _build_clip_vit(state_dict: dict) -> CLIPVisionTransformer:
    """Build CLIPVisionTransformer from a CLIP state_dict (full model or visual-only)."""
    # Extract visual keys
    vit_state = {}
    for k, v in state_dict.items():
        if k.startswith("visual."):
            vit_state[k[len("visual."):]] = v

    # Infer architecture from state dict shapes
    width = vit_state["conv1.weight"].shape[0]            # 1024 for ViT-L
    patch_size = vit_state["conv1.weight"].shape[-1]      # 14
    grid_size = round(
        (vit_state["positional_embedding"].shape[0] - 1) ** 0.5
    )                                                      # 24
    input_resolution = grid_size * patch_size              # 336
    layers = len([
        k for k in vit_state
        if k.startswith("transformer.resblocks.") and k.endswith(".attn.in_proj_weight")
    ])                                                     # 24
    heads = width // 64                                    # 16
    output_dim = vit_state["proj"].shape[1]                # 768

    model = CLIPVisionTransformer(
        input_resolution=input_resolution,
        patch_size=patch_size,
        width=width,
        layers=layers,
        heads=heads,
        output_dim=output_dim,
    )
    model.load_state_dict(vit_state, strict=True)
    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class CLIPVisualEncoder(nn.Module):
    """Frozen CLIP ViT-L/14 feature extractor.

    Loads the CLIP visual encoder, freezes all parameters, and extracts:
      - CLS token: [B, 768] — scene-level embedding
      - Patch tokens: [B, 1024, 24, 24] — spatial features

    Args:
        model_name: CLIP model identifier (default: "ViT-L/14@336px")
        cache_dir:  Directory for downloaded checkpoints (default: ~/.cache/clip)
    """

    def __init__(self, model_name: str = "ViT-L/14@336px",
                 cache_dir: str | None = None):
        super().__init__()

        # Download / load checkpoint
        if model_name in _CLIP_URLS:
            ckpt_path = _download_clip(
                _CLIP_URLS[model_name],
                cache_dir or os.path.expanduser("~/.cache/clip"),
            )
        elif os.path.isfile(model_name):
            ckpt_path = model_name
        else:
            raise ValueError(f"Unknown CLIP model: {model_name}")

        try:
            model = torch.jit.load(ckpt_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(ckpt_path, map_location="cpu",
                                    weights_only=False)

        self.visual = _build_clip_vit(state_dict)
        self.visual.float().eval()

        # Freeze everything
        for p in self.visual.parameters():
            p.requires_grad_(False)

        # Store metadata
        self.cls_dim = self.visual.output_dim      # 768
        self.patch_dim = self.visual.width          # 1024
        self.grid_size = self.visual.grid_size      # 24
        self.input_resolution = self.visual.input_resolution  # 336

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract CLIP features from pre-processed frames.

        Args:
            frames: [B, 3, 336, 336] CLIP-normalized RGB frames.

        Returns:
            cls_token:    [B, 768] — CLS embedding
            patch_tokens: [B, 1024, 24, 24] — spatial patch features
        """
        return self.visual(frames)


class PatchTokenProjection(nn.Module):
    """Project CLIP patch tokens to detection feature dimension.

    Conv2d(1024, 256, 1) + GroupNorm(32, 256).
    Matches Frozen-DETR: dino.py:201-203.
    """

    def __init__(self, clip_dim: int = 1024, d_model: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(clip_dim, d_model, kernel_size=1),
            nn.GroupNorm(32, d_model),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: [B, clip_dim, H, W] (e.g., [B, 1024, 24, 24])
        Returns:
            [B, d_model, H, W] (e.g., [B, 256, 24, 24])
        """
        return self.proj(patch_tokens)


def get_clip_preprocess(input_resolution: int = 336) -> Compose:
    """CLIP preprocessing transform for PIL images.

    Resize → CenterCrop → ToTensor → Normalize with CLIP stats.
    """
    return Compose([
        Resize(input_resolution, interpolation=BICUBIC),
        CenterCrop(input_resolution),
        lambda img: img.convert("RGB"),
        ToTensor(),
        Normalize(_CLIP_MEAN, _CLIP_STD),
    ])

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from position_encoding import SpatiotemporalPositionEncoding


class QwenViTExtractor(nn.Module):
    """
    Copied from exp1b so we can warm-start the same visual encoder + LoRA path.
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
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: list | None = None,
        n_layers: int = 8,
    ) -> None:
        if not hasattr(torch, "float8_e8m0fnu") and hasattr(torch, "float8_e4m3fn"):
            torch.float8_e8m0fnu = torch.float8_e4m3fn

        from peft import LoraConfig, get_peft_model

        block_range = "|".join(str(i) for i in range(n_layers))
        attn_modules = "|".join(target_modules or ["qkv", "proj"])
        regex = rf"blocks\.({block_range})\.attn\.({attn_modules})"

        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=regex,
            bias="none",
            inference_mode=False,
        )
        self.visual = get_peft_model(self.visual, lora_config)

    def lora_parameters(self):
        return [p for p in self.visual.parameters() if p.requires_grad]

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> List[torch.Tensor]:
        out = self.visual(pixel_values, image_grid_thw)
        merged = out.pooler_output

        base = getattr(self.visual, "base_model", self.visual)
        base = getattr(base, "model", base)
        merge_size = base.spatial_merge_size

        frame_feats: List[torch.Tensor] = []
        offset = 0
        for row in image_grid_thw:
            t, h, w = int(row[0]), int(row[1]), int(row[2])
            h_m, w_m = h // merge_size, w // merge_size
            n = t * h_m * w_m
            chunk = merged[offset : offset + n].view(t, h_m, w_m, -1)
            for frame in chunk:
                frame_feats.append(frame)
            offset += n

        return frame_feats


class FeatureProjection(nn.Module):
    def __init__(self, d_in: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x.float()))


class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, num_layers: int = 3):
        super().__init__()
        dims = [d_in] + [d_hidden] * (num_layers - 1) + [d_out]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ffn: int, dropout: float):
        super().__init__()
        self.norm_q1 = nn.LayerNorm(d_model)
        self.norm_q2 = nn.LayerNorm(d_model)
        self.norm_q3 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ffn),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_ffn, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        q = self.norm_q1(queries)
        sa_out, _ = self.self_attn(q, q, q, need_weights=False)
        queries = queries + self.dropout(sa_out)

        q = self.norm_q2(queries)
        mem = memory + pos_embed
        ca_out, _ = self.cross_attn(q, mem, mem, need_weights=False)
        queries = queries + self.dropout(ca_out)

        q = self.norm_q3(queries)
        queries = queries + self.dropout(self.ffn(q))
        return queries


class DETRDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_ffn: int,
        dropout: float,
        num_queries: int,
        clip_len: int,
    ):
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.temporal_embed = nn.Embedding(clip_len, d_model)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, nhead, dim_ffn, dropout) for _ in range(num_layers)]
        )
        self.box_head = MLP(d_model, d_model, 4, num_layers=3)

    def forward(self, memory: torch.Tensor, pos_embed: torch.Tensor, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        queries = self.query_embed.weight.unsqueeze(0)
        memory = memory.unsqueeze(0)
        pos_embed = pos_embed.unsqueeze(0)

        for layer in self.layers:
            queries = layer(queries, memory, pos_embed)

        query_feats = queries.squeeze(0)

        boxes = []
        for frame_idx in range(t):
            temp = self.temporal_embed.weight[frame_idx].unsqueeze(0)
            boxes.append(torch.sigmoid(self.box_head(query_feats + temp)))
        pred_boxes = torch.stack(boxes, dim=1)
        return query_feats, pred_boxes


class ClassificationHeads(nn.Module):
    def __init__(self, d_model: int, head_sizes: Dict[str, int]):
        super().__init__()
        self.heads = nn.ModuleDict({name: nn.Linear(d_model, size) for name, size in head_sizes.items()})

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: head(x.float()) for name, head in self.heads.items()}


class DETRROADModel(nn.Module):
    def __init__(
        self,
        model_id: str,
        vit_dim: int,
        d_model: int,
        head_sizes: Dict[str, int],
        clip_len: int,
        num_queries: int,
        num_decoder_layers: int,
        nhead: int,
        dim_ffn: int,
        dropout: float,
        freeze_vit: bool = True,
    ):
        super().__init__()
        self.clip_len = clip_len
        self.vit = QwenViTExtractor(model_id, freeze=freeze_vit)
        self.projection = FeatureProjection(vit_dim, d_model)
        self.position = SpatiotemporalPositionEncoding(d_model, max_t=clip_len)
        self.decoder = DETRDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_ffn=dim_ffn,
            dropout=dropout,
            num_queries=num_queries,
            clip_len=clip_len,
        )
        self.cls_heads = ClassificationHeads(d_model, head_sizes)

    def add_lora(self, **kwargs) -> None:
        self.vit.add_lora(**kwargs)

    def lora_parameters(self):
        return self.vit.lora_parameters()

    def decoder_parameters(self):
        return list(self.projection.parameters()) + list(self.position.parameters()) + list(self.decoder.parameters())

    def head_parameters(self):
        return list(self.cls_heads.parameters())

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> Dict[str, torch.Tensor]:
        frame_feats = self.vit(pixel_values, image_grid_thw)
        if not frame_feats:
            raise RuntimeError("QwenViTExtractor returned no frame features")

        t = len(frame_feats)
        h, w, _ = frame_feats[0].shape
        flat_tokens = torch.cat([f.reshape(-1, f.shape[-1]) for f in frame_feats], dim=0)
        memory = self.projection(flat_tokens)
        pos = self.position(t=t, h=h, w=w, device=memory.device).to(memory.dtype)
        query_feats, pred_boxes = self.decoder(memory.float(), pos.float(), t=t)
        pred_logits = self.cls_heads(query_feats)

        return {
            "pred_boxes": pred_boxes,
            "pred_logits": pred_logits,
            "query_feats": query_feats,
            "T": t,
            "frame_shape": (h, w),
        }


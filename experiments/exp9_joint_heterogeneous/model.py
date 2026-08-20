"""Exp9 model — one Qwen2.5-VL backbone, two output paths, per-corpus losses.

ROAD leg      frame (+drawn boxes) → ViT (LoRA) → merged token map →
              ROIAveragePool per detector box → Linear(1280→184) flat logits.
              No LLM pass, no text. Trains: ViT LoRA + head.
Language legs standard chat-template forward with prompt tokens masked to -100
              → LM cross-entropy. Trains: ViT LoRA + LLM LoRA.

The ViT LoRA is the shared surface between legs (DESIGN.md); the flat head is
the exp2f recipe (single Linear over the full 184-dim label space, logits out
— focal wants logits, the t-norm sigmoids the slices it needs).

ROIAveragePool is exp1's mechanism (exp1_road_r/model.py), reimplemented here
minus the GT-box assumptions: boxes arrive normalized to [0,1] frame coords.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

import config as C


def _pool_one(feat_map: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """feat_map [H',W',D]; box [4] normalized xyxy → [D] mean over the region.
    Same index arithmetic as exp1's ROIAveragePool._pool_one."""
    H, W, _ = feat_map.shape
    x1, y1, x2, y2 = box.tolist()
    col_lo = max(0, int(x1 * W))
    col_hi = min(W, max(col_lo + 1, round(x2 * W + 0.5)))
    row_lo = max(0, int(y1 * H))
    row_hi = min(H, max(row_lo + 1, round(y2 * H + 0.5)))
    return feat_map[row_lo:row_hi, col_lo:col_hi, :].mean(dim=(0, 1))


class Exp9Model(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from peft import LoraConfig, get_peft_model

        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            C.MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": device},
        )
        base.config.use_cache = False
        lora_cfg = LoraConfig(
            r=C.LORA_R, lora_alpha=C.LORA_ALPHA, lora_dropout=C.LORA_DROPOUT,
            target_modules=C.LORA_TARGET_REGEX, bias="none",
        )
        self.vlm = get_peft_model(base, lora_cfg)
        self.processor = AutoProcessor.from_pretrained(
            C.MODEL_ID, min_pixels=C.MIN_PIXELS, max_pixels=C.MAX_PIXELS,
        )
        self.head = nn.Linear(C.D_VIT, C.NUM_CLASSES).to(device).float()
        self.device = device

        n_vit = sum(p.numel() for n, p in self.vlm.named_parameters()
                    if "lora" in n and "visual" in n)
        n_llm = sum(p.numel() for n, p in self.vlm.named_parameters()
                    if "lora" in n and "visual" not in n)
        assert n_vit > 0, "no LoRA params landed on the ViT — check LORA_TARGET_REGEX"
        assert n_llm > 0, "no LoRA params landed on the LLM — check LORA_TARGET_REGEX"
        print(f"[model] LoRA params: ViT {n_vit:,} | LLM {n_llm:,} | "
              f"head {sum(p.numel() for p in self.head.parameters()):,}", flush=True)

    # ---- ROAD leg ---------------------------------------------------------
    def _visual(self):
        """The (LoRA-wrapped) Qwen vision tower."""
        return self.vlm.base_model.model.model.visual

    def road_forward(self, image, boxes_norm: torch.Tensor) -> torch.Tensor:
        """image: PIL.Image; boxes_norm: [n,4] normalized xyxy.
        Returns [n, 184] logits."""
        inputs = self.processor.image_processor(images=[image], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, torch.bfloat16)
        grid_thw = inputs["image_grid_thw"].to(self.device)

        visual = self._visual()
        merged = visual(pixel_values, grid_thw)          # [n_tokens, D]
        if not isinstance(merged, torch.Tensor):          # transformers>=5 returns output obj
            merged = merged.pooler_output
        m = visual.spatial_merge_size
        t, h, w = (int(v) for v in grid_thw[0])
        feat_map = merged.view(t * (h // m), w // m, -1)  # [H', W', D], t==1

        feats = torch.stack([_pool_one(feat_map, b) for b in boxes_norm.to(self.device)])
        return self.head(feats.float())

    # ---- language legs ----------------------------------------------------
    def lang_forward(self, image_path: str, user_text: str, target_text: str) -> torch.Tensor:
        """One SFT sample → LM loss with prompt tokens masked."""
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        messages = [{"role": "user", "content": [
            {"type": "image", "image": Image.open(image_path).convert("RGB")},
            {"type": "text", "text": user_text},
        ]}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)

        full = prompt + target_text + self.processor.tokenizer.eos_token
        enc = self.processor(text=[full], images=image_inputs,
                             return_tensors="pt", padding=False)
        enc = {k: v.to(self.device) for k, v in enc.items()}

        n_prompt = self.processor(text=[prompt], images=image_inputs,
                                  return_tensors="pt")["input_ids"].shape[1]
        labels = enc["input_ids"].clone()
        labels[:, :n_prompt] = -100
        out = self.vlm(**enc, labels=labels)
        return out.loss

    # ---- checkpointing ----------------------------------------------------
    def save(self, ckpt_dir):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.vlm.save_pretrained(str(ckpt_dir / "adapter"))
        torch.save(self.head.state_dict(), ckpt_dir / "head.pt")

    def load_head_and_adapter(self, ckpt_dir):
        from peft import PeftModel  # noqa: F401  (adapter loaded via load_adapter)
        self.vlm.load_adapter(str(ckpt_dir / "adapter"), adapter_name="default")
        self.head.load_state_dict(torch.load(ckpt_dir / "head.pt", weights_only=True))

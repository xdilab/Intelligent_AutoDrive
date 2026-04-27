#!/usr/bin/env python3
"""
Exp3 — BDD-X captioning fine-tune on Qwen2.5-VL-7B-Instruct.

Trainable: LoRA r=16 on LLM attention + MLP layers, plus merger (projector).
Frozen:    ViT (model.visual.patch_embed + transformer blocks).
Loss:      Cross-entropy on assistant tokens only (label mask on prompt).
Val:       Teacher-forced val loss; best checkpoint saved on lowest val loss.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import config as C
from dataset import BDDXDataset


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a driving behavior analyst."
USER_QUESTION = "Describe what the vehicle is doing and explain why."


def build_messages(image, action: str, justification: str) -> tuple[list, list]:
    """Return (messages_full, messages_prompt) for a single example."""
    messages_full = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_QUESTION},
            ],
        },
        {
            "role": "assistant",
            "content": f"Action: {action}\nJustification: {justification}",
        },
    ]
    messages_prompt = messages_full[:-1]  # everything except assistant turn
    return messages_full, messages_prompt


def encode_example(processor, image, action: str, justification: str, device, dtype):
    """
    Tokenize one example, return (input_ids, pixel_values, image_grid_thw,
    attention_mask, labels) all on device.

    Label masking: process prompt-only with the same image to get exact
    prompt token length (including expanded image tokens), then mask those
    positions with -100.
    """
    messages_full, messages_prompt = build_messages(image, action, justification)

    text_full = processor.apply_chat_template(
        messages_full, tokenize=False, add_generation_prompt=False
    )
    text_prompt = processor.apply_chat_template(
        messages_prompt, tokenize=False, add_generation_prompt=True
    )

    enc_full = processor(
        text=[text_full],
        images=[image],
        return_tensors="pt",
        min_pixels=C.MIN_PIXELS,
        max_pixels=C.MAX_PIXELS,
    )
    enc_prompt = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
        min_pixels=C.MIN_PIXELS,
        max_pixels=C.MAX_PIXELS,
    )

    input_ids = enc_full["input_ids"][0].to(device)
    pixel_values = enc_full["pixel_values"].to(device=device, dtype=dtype)
    image_grid_thw = enc_full["image_grid_thw"].to(device)
    attention_mask = enc_full["attention_mask"][0].to(device)

    prompt_len = enc_prompt["input_ids"].shape[1]
    labels = input_ids.clone()
    labels[:prompt_len] = -100  # mask prompt; loss on response tokens only

    return input_ids, pixel_values, image_grid_thw, attention_mask, labels


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def build_model(device):
    print(f"Loading {C.MODEL_ID} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        C.MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )

    # Freeze everything, then selectively unfreeze via LoRA + merger
    model.requires_grad_(False)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=C.LORA_R,
        lora_alpha=C.LORA_ALPHA,
        lora_dropout=C.LORA_DROPOUT,
        target_modules=C.LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    # Unfreeze merger (projector: ViT → LLM embedding space)
    for param in model.base_model.model.visual.merger.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")

    return model


# ---------------------------------------------------------------------------
# Optimizer + scheduler
# ---------------------------------------------------------------------------

def build_optimizer(model, n_steps_total: int):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=C.LR,
        weight_decay=C.WEIGHT_DECAY,
    )

    def lr_lambda(step: int) -> float:
        if step < C.WARMUP_STEPS:
            return step / max(1, C.WARMUP_STEPS)
        progress = (step - C.WARMUP_STEPS) / max(1, n_steps_total - C.WARMUP_STEPS)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler, processor, device, dtype, epoch):
    model.train()
    total_loss = 0.0
    n_tokens = 0
    optimizer.zero_grad()
    t0 = time.time()

    for step, batch in enumerate(loader, 1):
        image = batch["image"]
        action = batch["action"]
        justification = batch["justification"]

        input_ids, pixel_values, image_grid_thw, attention_mask, labels = encode_example(
            processor, image, action, justification, device, dtype
        )

        out = model(
            input_ids=input_ids.unsqueeze(0),
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask.unsqueeze(0),
            labels=labels.unsqueeze(0),
        )

        loss = out.loss / C.GRAD_ACCUM
        loss.backward()

        n_resp = (labels != -100).sum().item()
        total_loss += out.loss.item() * n_resp
        n_tokens += n_resp

        if step % C.GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], C.GRAD_CLIP
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % 200 == 0:
            elapsed = time.time() - t0
            avg = total_loss / max(n_tokens, 1)
            print(
                f"  [train] ep{epoch}/{C.MAX_EPOCHS} step {step}/{len(loader)}"
                f" | loss={avg:.4f} | {elapsed:.0f}s elapsed"
            )

    # Final grad step if leftover accumulation
    if len(loader) % C.GRAD_ACCUM != 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], C.GRAD_CLIP
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / max(n_tokens, 1)


@torch.no_grad()
def val_epoch(model, loader, processor, device, dtype, epoch):
    model.eval()
    total_loss = 0.0
    n_tokens = 0

    for batch in loader:
        image = batch["image"]
        action = batch["action"]
        justification = batch["justification"]

        input_ids, pixel_values, image_grid_thw, attention_mask, labels = encode_example(
            processor, image, action, justification, device, dtype
        )

        out = model(
            input_ids=input_ids.unsqueeze(0),
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask.unsqueeze(0),
            labels=labels.unsqueeze(0),
        )

        n_resp = (labels != -100).sum().item()
        total_loss += out.loss.item() * n_resp
        n_tokens += n_resp

    return total_loss / max(n_tokens, 1)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch, val_loss, path: str, tag: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
    print(f"  Saved {tag} checkpoint → {path}  (val_loss={val_loss:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(C.CKPT_DIR, exist_ok=True)
    os.makedirs(C.LOG_DIR, exist_ok=True)

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    print("Loading processor ...")
    processor = AutoProcessor.from_pretrained(C.MODEL_ID)

    train_ds = BDDXDataset(C.CSV_PATH, C.IMG_DIR, f"{C.SPLIT_DIR}/train.txt")
    val_ds = BDDXDataset(C.CSV_PATH, C.IMG_DIR, f"{C.SPLIT_DIR}/val.txt")
    print(f"Dataset: train={len(train_ds)} val={len(val_ds)} examples")

    # batch_size=1 always; collate returns single dict (not a list)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=2, collate_fn=lambda b: b[0])
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=2, collate_fn=lambda b: b[0])

    model = build_model(device)

    n_optimizer_steps = (len(train_ds) // C.GRAD_ACCUM) * C.MAX_EPOCHS
    optimizer, scheduler = build_optimizer(model, n_optimizer_steps)

    best_val_loss = float("inf")
    log_rows = []

    for epoch in range(1, C.MAX_EPOCHS + 1):
        print(f"\nStarting epoch {epoch}/{C.MAX_EPOCHS} ...")
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, processor, device, dtype, epoch
        )
        val_loss = val_epoch(model, val_loader, processor, device, dtype, epoch)

        print(
            f"Epoch {epoch}/{C.MAX_EPOCHS} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )
        log_rows.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        )
        with open(f"{C.LOG_DIR}/metrics.json", "w") as f:
            json.dump(log_rows, f, indent=2)

        latest_path = f"{C.CKPT_DIR}/latest.pt"
        save_checkpoint(model, optimizer, epoch, val_loss, latest_path, "latest")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = f"{C.CKPT_DIR}/best.pt"
            save_checkpoint(model, optimizer, epoch, val_loss, best_path, "best")

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()

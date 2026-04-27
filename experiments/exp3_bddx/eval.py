#!/usr/bin/env python3
"""
Exp3 BDD-X evaluation — generates action+justification text and computes BLEU-4.

Usage:
    python eval.py --ckpt checkpoints/best.pt --split test
    python eval.py --ckpt checkpoints/best.pt --split val

Outputs per-head BLEU-4 (action, justification, combined) and saves
generated text to logs/eval_{split}.json for qualitative review.

Requires sacrebleu (installed by run.sh).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import sacrebleu
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

EXP_DIR = Path(__file__).resolve().parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import config as C
from dataset import BDDXDataset
from train import SYSTEM_PROMPT, USER_QUESTION, build_model


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def build_prompt_text(processor, image) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_QUESTION},
            ],
        },
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def parse_output(text: str) -> tuple[str, str]:
    """
    Extract action and justification from generated text.
    Expected format: "Action: ...\nJustification: ..."
    Returns (action_text, justification_text); falls back to full text if unparseable.
    """
    text = text.strip()
    action = text
    justification = text
    if "Justification:" in text:
        parts = text.split("Justification:", 1)
        justification = parts[1].strip()
        action_part = parts[0]
        if "Action:" in action_part:
            action = action_part.split("Action:", 1)[1].strip().rstrip("\n")
    elif "Action:" in text:
        action = text.split("Action:", 1)[1].strip()
    return action, justification


@torch.no_grad()
def generate_one(model, processor, image, device, dtype) -> str:
    text_prompt = build_prompt_text(processor, image)
    enc = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
        min_pixels=C.MIN_PIXELS,
        max_pixels=C.MAX_PIXELS,
    )
    input_ids = enc["input_ids"].to(device)
    pixel_values = enc["pixel_values"].to(device=device, dtype=dtype)
    image_grid_thw = enc["image_grid_thw"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    generated = model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
        max_new_tokens=C.MAX_NEW_TOKENS,
        num_beams=C.NUM_BEAMS,
        do_sample=False,
    )
    new_tokens = generated[0][input_ids.shape[1]:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    print("Loading processor ...")
    processor = AutoProcessor.from_pretrained(C.MODEL_ID)

    print("Loading model ...")
    model = build_model(device)

    print(f"Loading checkpoint from {args.ckpt} ...")
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  missing={len(missing)} unexpected={len(unexpected)}")

    model.eval()

    split_file = f"{C.SPLIT_DIR}/{args.split}.txt"
    dataset = BDDXDataset(C.CSV_PATH, C.IMG_DIR, split_file)
    print(f"Evaluating on {args.split}: {len(dataset)} examples")

    hyp_action, ref_action = [], []
    hyp_just, ref_just = [], []
    hyp_combined, ref_combined = [], []
    records = []

    for i, ex in enumerate(dataset):
        image = ex["image"]
        gt_action = ex["action"]
        gt_just = ex["justification"]

        generated = generate_one(model, processor, image, device, dtype)
        pred_action, pred_just = parse_output(generated)

        hyp_action.append(pred_action)
        ref_action.append([gt_action])

        hyp_just.append(pred_just)
        ref_just.append([gt_just])

        hyp_combined.append(generated)
        ref_combined.append([f"Action: {gt_action}\nJustification: {gt_just}"])

        records.append(
            {
                "gt_action": gt_action,
                "gt_justification": gt_just,
                "pred_action": pred_action,
                "pred_justification": pred_just,
                "raw_generated": generated,
            }
        )

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(dataset)}] generated so far ...")

    # BLEU-4
    bleu_action = sacrebleu.corpus_bleu(hyp_action, ref_action).score
    bleu_just = sacrebleu.corpus_bleu(hyp_just, ref_just).score
    bleu_combined = sacrebleu.corpus_bleu(hyp_combined, ref_combined).score

    print(f"\n=== BDD-X Eval ({args.split}) ===")
    print(f"  BLEU-4 action:        {bleu_action:.2f}")
    print(f"  BLEU-4 justification: {bleu_just:.2f}")
    print(f"  BLEU-4 combined:      {bleu_combined:.2f}")

    os.makedirs(C.LOG_DIR, exist_ok=True)
    out_path = f"{C.LOG_DIR}/eval_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "split": args.split,
                "n_examples": len(dataset),
                "bleu4_action": bleu_action,
                "bleu4_justification": bleu_just,
                "bleu4_combined": bleu_combined,
                "samples": records[:50],  # first 50 for qualitative review
            },
            f,
            indent=2,
        )
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()

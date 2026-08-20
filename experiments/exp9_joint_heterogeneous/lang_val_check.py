"""Language-leg forgetting check — did ROAD training damage language capability?

Computes mean next-token val loss (prompt tokens masked, identical masking to
train.py's lang_forward) on the held-out BDD-X (545) and CoVLA (250) SFT val
sets, for one of four model variants:

  base  Qwen/Qwen2.5-VL-7B-Instruct, untouched          (floor / reference)
  exp8  exp8 joint-LoRA merged_checkpoint_9596           (language-trained, no heads)
  r3    exp9 R3 ep003 adapter (joint, no t-norm)         (the question)
  r4    exp9 R4 ep003 adapter (joint, t-norm λ=10)       (the question)

Reading: r3/r4 ≈ exp8  → language survived joint heterogeneous training.
         r3/r4 ≫ exp8/base → ROAD training degraded language (two-way interference).

Usage:  CUDA_VISIBLE_DEVICES=0 python -u lang_val_check.py --model base --out lang_val_base.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import torch

import config as C

BDDX_VAL = C.EXP_DIR.parent / "exp7_bddx_lora" / "cache" / "bddx_sft_val.json"
COVLA_VAL = C.EXP8_DIR / "cache" / "covla_sft_val.json"
EXP8_MERGED = C.EXP8_DIR / "checkpoints" / "merged_checkpoint_9596"


@torch.no_grad()
def lm_loss(vlm, processor, device, image_path, user_text, target_text) -> float:
    """Identical masking scheme to Exp9Model.lang_forward."""
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    messages = [{"role": "user", "content": [
        {"type": "image", "image": Image.open(image_path).convert("RGB")},
        {"type": "text", "text": user_text},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    full = prompt + target_text + processor.tokenizer.eos_token
    enc = processor(text=[full], images=image_inputs, return_tensors="pt", padding=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    n_prompt = processor(text=[prompt], images=image_inputs,
                         return_tensors="pt")["input_ids"].shape[1]
    labels = enc["input_ids"].clone()
    labels[:, :n_prompt] = -100
    return float(vlm(**enc, labels=labels).loss)


def load_variant(name: str, device):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    if name in ("base", "exp8"):
        src = C.MODEL_ID if name == "base" else str(EXP8_MERGED)
        vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            src, torch_dtype=torch.bfloat16, device_map={"": device}).eval()
        proc = AutoProcessor.from_pretrained(src, min_pixels=C.MIN_PIXELS,
                                             max_pixels=C.MAX_PIXELS)
        return vlm, proc
    from model import Exp9Model
    m = Exp9Model(device)
    m.load_head_and_adapter(C.CKPT_DIR / f"{name}_ep003")
    m.vlm.eval()
    return m.vlm, m.processor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("base", "exp8", "r3", "r4"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    vlm, proc = load_variant(args.model, device)
    print(f"[langval] {args.model} loaded", flush=True)

    from dataset import LangDataset
    out = {"model": args.model}
    for name, path in (("bddx", BDDX_VAL), ("covla", COVLA_VAL)):
        ds = LangDataset(path, f"{name}-val")
        losses, t0 = [], time.time()
        for i in range(len(ds)):
            losses.append(lm_loss(vlm, proc, device, *ds[i]))
            if (i + 1) % 100 == 0:
                print(f"[langval] {args.model} {name} {i + 1}/{len(ds)} "
                      f"| mean so far {sum(losses)/len(losses):.4f} "
                      f"| {(time.time()-t0)/(i+1):.2f}s/sample", flush=True)
        out[name] = {"mean_loss": sum(losses) / len(losses), "n": len(losses)}
        print(f"[langval] {args.model} {name}: mean {out[name]['mean_loss']:.4f} "
              f"over {len(losses)}", flush=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[langval] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

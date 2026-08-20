"""Exp7 stage 3 — export the tuned VLM in the exact form exp5 loads.

exp5/qwen_infer.py loads the model with
    Qwen2_5_VLForConditionalGeneration.from_pretrained(C.QWEN_MODEL, ...)
where QWEN_MODEL is a HF id *or local path*. This script merges the LoRA
adapter into the base weights and writes a plain HF model directory, so
re-caching is a one-line swap of QWEN_MODEL in exp5/config.py — no exp5 code
changes.

Checkpoint selection: by default the epoch checkpoint with the LOWEST BDD-X
val loss (read from each checkpoint's trainer_state.json) is exported — never
selected on the downstream ROAD-Waymo test split.

Usage:
  /home/brandon/miniconda3/bin/python -u export_for_exp5.py            # auto-pick by val loss
  /home/brandon/miniconda3/bin/python -u export_for_exp5.py \
      --adapter checkpoints/checkpoint-3255 --out checkpoints/merged_ep3
Runs on CPU (host RAM); no GPU needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import config as C


def _last_eval_loss(ckpt: Path) -> float | None:
    state = ckpt / "trainer_state.json"
    if not state.exists():
        return None
    hist = json.loads(state.read_text()).get("log_history", [])
    losses = [h["eval_loss"] for h in hist if "eval_loss" in h]
    return losses[-1] if losses else None


def _pick_adapter() -> Path:
    ckpts = sorted(
        (d for d in C.CKPT_DIR.glob("checkpoint-*") if (d / "adapter_config.json").exists()),
        key=lambda d: int(d.name.split("-")[-1]),
    )
    if not ckpts:
        raise SystemExit(f"no checkpoint-*/adapter_config.json under {C.CKPT_DIR} — train first")
    print("[export] candidate checkpoints (BDD-X val loss):")
    scored = []
    for d in ckpts:
        vl = _last_eval_loss(d)
        print(f"[export]   {d.name}: eval_loss={vl if vl is not None else 'n/a'}")
        if vl is not None:
            scored.append((vl, d))
    if not scored:
        print("[export] no eval_loss recorded — falling back to the last checkpoint")
        return ckpts[-1]
    best = min(scored, key=lambda t: t[0])
    print(f"[export] selected {best[1].name} (val loss {best[0]:.4f})")
    return best[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None,
                    help="Adapter dir (contains adapter_config.json). Default: best-val-loss checkpoint.")
    ap.add_argument("--out", default=None,
                    help="Output dir for the merged HF model. Default: checkpoints/merged_<ckpt>.")
    args = ap.parse_args()

    adapter = Path(args.adapter) if args.adapter else _pick_adapter()
    assert (adapter / "adapter_config.json").exists(), f"{adapter} is not a PEFT adapter dir"
    out = Path(args.out) if args.out else C.CKPT_DIR / f"merged_{adapter.name.replace('-', '_')}"

    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"[export] loading base {C.MODEL_ID} (bf16, CPU) ...")
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(C.MODEL_ID, dtype=torch.bfloat16)
    print(f"[export] applying adapter {adapter} ...")
    model = PeftModel.from_pretrained(base, str(adapter))
    print("[export] merging LoRA into base weights ...")
    merged = model.merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    AutoProcessor.from_pretrained(C.MODEL_ID).save_pretrained(str(out))
    n_files = len(list(out.iterdir()))
    print(f"[export] wrote merged model ({n_files} files) -> {out}")

    print("\n[export] to re-cache with the tuned VLM (exp5 pipeline):")
    print(f"  1. in exp5_qwen_reasoning/config.py set  QWEN_MODEL = \"{out}\"")
    print("  2. point QWEN_CACHE_DIR / QWEN_STEERED_CACHE_DIR at fresh dirs "
          "(e.g. cache/qwen_bddx_lora/) so the zero-shot cache stays the control")
    print("  3. rerun qwen_infer.py for train+val, then re-embed rationales and "
          "retrain the exp6 fusion head on the new cache")


if __name__ == "__main__":
    main()

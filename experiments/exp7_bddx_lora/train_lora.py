"""Exp7 stage 2 — LoRA SFT via the reference trainer (thin launcher).

This does NOT reimplement training. It imports the official QwenLM/Qwen2.5-VL
finetune trainer (reference/Qwen2.5-VL/qwen-vl-finetune) unmodified and only:
  1. registers the BDD-X annotation JSONs in its dataset registry at runtime
     (its design expects you to edit qwenvl/data/__init__.py; we inject instead
     so the reference clone stays pristine);
  2. wraps its make_supervised_data_module to attach a BDD-X *val* dataset so
     HF Trainer reports eval_loss each epoch (lab rule: BDD-X val is for SFT
     validation loss only — checkpoint selection happens on it);
  3. assembles the reference CLI args from config.py and calls train().

Resumable: the reference train() auto-resumes from the newest checkpoints/
checkpoint-* if one exists — just rerun the same command.

Usage (single GPU; do not launch while the loss-diagnostic job holds a GPU):
  CUDA_VISIBLE_DEVICES=1 /home/brandon/miniconda3/bin/python -u train_lora.py
  # fallback if the flash-attn varlen path misbehaves at runtime:
  CUDA_VISIBLE_DEVICES=1 /home/brandon/miniconda3/bin/python -u train_lora.py \
      --attn sdpa --no-flatten
"""

from __future__ import annotations

import argparse
import copy
import sys

sys.stdout.reconfigure(line_buffering=True)

import config as C


def _add_reference_to_path() -> None:
    root = C.REFERENCE_ROOT
    assert root.is_dir(), (
        f"reference repo missing at {root} — clone it first (see README)"
    )
    # qwen-vl-finetune root for `qwenvl.*`; qwenvl/train for its
    # script-relative `from trainer import ...`.
    for p in (str(root), str(root / "qwenvl" / "train")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _build_argv(attn: str, flatten: bool) -> list[str]:
    """Reference scripts/sft_30a3b_lora.sh args; deviations documented in README."""
    return [
        "train_lora.py",
        "--model_name_or_path", C.MODEL_ID,
        "--dataset_use", "bddx_train",
        "--data_flatten", str(flatten),
        "--lora_enable", "True",
        "--lora_r", str(C.LORA_R),
        "--lora_alpha", str(C.LORA_ALPHA),
        "--lora_dropout", str(C.LORA_DROPOUT),
        "--bf16", "True",
        "--output_dir", str(C.CKPT_DIR),
        "--num_train_epochs", str(C.NUM_EPOCHS),
        "--per_device_train_batch_size", str(C.PER_DEVICE_BS),
        "--per_device_eval_batch_size", str(C.PER_DEVICE_EVAL_BS),
        "--gradient_accumulation_steps", str(C.GRAD_ACCUM),
        "--max_pixels", str(C.MAX_PIXELS),
        "--min_pixels", str(C.MIN_PIXELS),
        "--eval_strategy", C.EVAL_STRATEGY,
        "--prediction_loss_only", "True",
        "--save_strategy", C.SAVE_STRATEGY,
        "--learning_rate", str(C.LR),
        "--weight_decay", str(C.WEIGHT_DECAY),
        "--warmup_ratio", str(C.WARMUP_RATIO),
        "--max_grad_norm", str(C.MAX_GRAD_NORM),
        "--lr_scheduler_type", C.LR_SCHEDULER,
        "--logging_steps", str(C.LOGGING_STEPS),
        "--model_max_length", str(C.MODEL_MAX_LENGTH),
        "--gradient_checkpointing", "True",
        "--dataloader_num_workers", "4",
        "--run_name", "exp7-bddx-lora",
        "--report_to", "none",
        # NOTE: --logging_dir omitted (deprecated in transformers 5.x); stdout
        # is teed to logs/train.log per the pipeline command. warmup_ratio is
        # deprecation-warned but verified functional on 5.3.0
        # (get_warmup_steps(3255) == 98).
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", choices=("flash_attention_2", "sdpa"),
                    default=C.ATTN_IMPLEMENTATION)
    ap.add_argument("--no-flatten", action="store_true",
                    help="Disable data_flatten (use with --attn sdpa fallback).")
    args = ap.parse_args()
    flatten = C.DATA_FLATTEN and not args.no_flatten
    if flatten and args.attn != "flash_attention_2":
        raise SystemExit("data_flatten needs flash-attn varlen; add --no-flatten with --attn sdpa")

    for f in (C.SFT_TRAIN_JSON, C.SFT_VAL_JSON):
        assert f.exists(), f"{f} missing — run data_prep.py first"
    C.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    C.LOG_DIR.mkdir(parents=True, exist_ok=True)

    _add_reference_to_path()
    import qwenvl.data as qd

    qd.data_dict["bddx_train"] = {"annotation_path": str(C.SFT_TRAIN_JSON), "data_path": ""}
    qd.data_dict["bddx_val"] = {"annotation_path": str(C.SFT_VAL_JSON), "data_path": ""}

    import qwenvl.train.train_qwen as tq
    from qwenvl.data.data_processor import LazySupervisedDataset

    _orig_make = tq.make_supervised_data_module

    def _make_with_val(processor, data_args):
        module = _orig_make(processor, data_args=data_args)
        val_args = copy.copy(data_args)
        val_args.dataset_use = "bddx_val"
        module["eval_dataset"] = LazySupervisedDataset(processor, data_args=val_args)
        print(f"[train] eval dataset attached: {len(module['eval_dataset'])} BDD-X val samples",
              flush=True)
        return module

    tq.make_supervised_data_module = _make_with_val

    # Compat shim for transformers >= 5.5: the reference's return_mask() bypass
    # (trainer.py) names its 2nd param `input_embeds`, but the 5.5.x call site
    # passes `inputs_embeds` — the kwarg lands in **kwargs and the positional
    # goes unfilled (TypeError at step 0). replace_qwen2_vl_attention_class()
    # runs *inside* train() and assigns its module-global `return_mask`, so we
    # swap that symbol beforehand; same behavior (return the mask untouched —
    # flash-attn packing routes causality through position_ids, not the mask).
    import trainer as ref_trainer

    def _return_mask_compat(*_args, attention_mask=None, **_kwargs):
        return attention_mask

    ref_trainer.return_mask = _return_mask_compat
    print("[train] applied transformers>=5.5 return_mask signature shim", flush=True)

    sys.argv = _build_argv(args.attn, flatten)
    print(f"[train] launching reference trainer (attn={args.attn}, flatten={flatten})",
          flush=True)
    print(f"[train] argv: {' '.join(sys.argv[1:])}", flush=True)
    tq.train(attn_implementation=args.attn)


if __name__ == "__main__":
    main()

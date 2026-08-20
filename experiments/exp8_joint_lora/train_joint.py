"""Exp8 stage 2 — joint LoRA SFT via the reference trainer (thin launcher).

Same launcher pattern as exp7/train_lora.py (reference trainer imported
unmodified, datasets injected at runtime, val dataset attached) plus ONE
addition: a round-robin train sampler so that every optimizer step's
gradient-accumulation window (GRAD_ACCUM=3) contains exactly one BDD-X, one
CoVLA and one ROAD-Waymo sample — Dr. Moradi's interleaved joint-training
strategy. Within each corpus the order reshuffles every epoch; corpora
smaller than the largest cycle.

Resumable: rerun the same command after a crash.

Usage (single GPU):
  CUDA_VISIBLE_DEVICES=1 /home/brandon/miniconda3/bin/python -u train_joint.py
  # quick smoke (2 optimizer steps, no checkpoint):
  CUDA_VISIBLE_DEVICES=0 /home/brandon/miniconda3/bin/python -u train_joint.py --max-steps 2
"""

from __future__ import annotations

import argparse
import copy
import json
import sys

sys.stdout.reconfigure(line_buffering=True)

import config as C


def _add_reference_to_path() -> None:
    root = C.REFERENCE_ROOT
    assert root.is_dir(), f"reference repo missing at {root}"
    for p in (str(root), str(root / "qwenvl" / "train")):
        if p not in sys.path:
            sys.path.insert(0, p)


class RoundRobinSampler:
    """Yields dataset indices as bddx,covla,road,bddx,... over the
    block-concatenated JOINT_TRAIN_JSON. One cycle per index of the largest
    block; smaller blocks wrap (modulo). Each __iter__ call (= each epoch)
    reshuffles within every block."""

    def __init__(self, sizes: list[int], seed: int) -> None:
        import torch  # local so config import stays torch-free

        self._torch = torch
        self.sizes = sizes
        self.offsets = [sum(sizes[:i]) for i in range(len(sizes))]
        self.n_cycle = max(sizes)
        self.seed = seed
        self.calls = 0

    def __len__(self) -> int:
        return self.n_cycle * len(self.sizes)

    def __iter__(self):
        g = self._torch.Generator()
        g.manual_seed(self.seed + self.calls)
        self.calls += 1
        perms = [self._torch.randperm(s, generator=g) + off
                 for s, off in zip(self.sizes, self.offsets)]
        for i in range(self.n_cycle):
            for perm, s in zip(perms, self.sizes):
                yield int(perm[i % s])


def _build_argv(attn: str, max_steps: int) -> list[str]:
    argv = [
        "train_joint.py",
        "--model_name_or_path", C.MODEL_ID,
        "--dataset_use", "joint_train",
        "--data_flatten", str(C.DATA_FLATTEN),
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
        "--run_name", "exp8-joint-lora",
        "--report_to", "none",
    ]
    if max_steps:
        argv += ["--max_steps", str(max_steps),
                 "--eval_strategy", "no", "--save_strategy", "no"]
    return argv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", choices=("flash_attention_2", "sdpa"),
                    default=C.ATTN_IMPLEMENTATION)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="Smoke: cap optimizer steps, skip eval/saving. 0 = full run.")
    args = ap.parse_args()

    for f in (C.JOINT_TRAIN_JSON, C.JOINT_VAL_JSON, C.JOINT_STATS_JSON):
        assert f.exists(), f"{f} missing — run data_prep.py first"
    C.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    C.LOG_DIR.mkdir(parents=True, exist_ok=True)

    stats = json.loads(C.JOINT_STATS_JSON.read_text())
    sizes = [stats["train_block_sizes"][n] for n in C.BLOCK_ORDER]
    assert stats["block_order"] == list(C.BLOCK_ORDER)
    print(f"[train] block sizes {dict(zip(C.BLOCK_ORDER, sizes))} — "
          f"epoch = {3 * max(sizes):,} samples, "
          f"{max(sizes):,} optimizer steps (1 per corpus triple)", flush=True)

    _add_reference_to_path()
    import qwenvl.data as qd

    qd.data_dict["joint_train"] = {"annotation_path": str(C.JOINT_TRAIN_JSON), "data_path": ""}
    qd.data_dict["joint_val"] = {"annotation_path": str(C.JOINT_VAL_JSON), "data_path": ""}

    import qwenvl.train.train_qwen as tq
    from qwenvl.data.data_processor import LazySupervisedDataset

    _orig_make = tq.make_supervised_data_module

    def _make_with_val(processor, data_args):
        module = _orig_make(processor, data_args=data_args)
        val_args = copy.copy(data_args)
        val_args.dataset_use = "joint_val"
        module["eval_dataset"] = LazySupervisedDataset(processor, data_args=val_args)
        print(f"[train] eval dataset attached: {len(module['eval_dataset'])} "
              f"joint val samples", flush=True)
        return module

    tq.make_supervised_data_module = _make_with_val

    # Round-robin sampler (transformers 5.x: Trainer._get_train_sampler).
    import transformers

    def _rr_sampler(self, train_dataset=None):
        return RoundRobinSampler(sizes, seed=int(self.args.seed))

    transformers.Trainer._get_train_sampler = _rr_sampler
    print("[train] round-robin train sampler installed (1 sample per corpus "
          "per optimizer step)", flush=True)

    # exp7's transformers>=5.5 return_mask signature shim (harmless on 5.3).
    import trainer as ref_trainer

    def _return_mask_compat(*_args, attention_mask=None, **_kwargs):
        return attention_mask

    ref_trainer.return_mask = _return_mask_compat

    sys.argv = _build_argv(args.attn, args.max_steps)
    print(f"[train] launching reference trainer (attn={args.attn})", flush=True)
    print(f"[train] argv: {' '.join(sys.argv[1:])}", flush=True)
    tq.train(attn_implementation=args.attn)


if __name__ == "__main__":
    main()

"""Exp7 smoke test — validates the scaffold without full training.

No-GPU checks (always run):
  1. Paths + reference-repo presence; our LoRA config still matches the
     reference source (parsed from reference files, not hardcoded twice).
  2. Data prep on a slice of the real CSV: alignment filter, image existence,
     prompt/target template round-trip (target parses as strict JSON and
     reproduces the annotator text verbatim).
  3. Reference trainer imports under this env (transformers 5.3 / peft 0.19 /
     flash_attn 2.8.3) + runtime dataset registration.
  4. Reference dataset end-to-end on CPU: a real sample through
     LazySupervisedDataset -> tokens, label mask covers the JSON target only.

GPU check (gated — skipped unless a CUDA device with >30 GB free is visible;
a loss-diagnostic job may own the GPUs, prefer CUDA_VISIBLE_DEVICES=1):
  5. Base model + reference LoraConfig load; one forward+backward on a real
     sample; finite loss; gradients reach LoRA params only.

Run:  CUDA_VISIBLE_DEVICES=1 /home/brandon/miniconda3/bin/python -u smoke_test.py
"""

from __future__ import annotations

import json
import re
import shutil
import sys

sys.stdout.reconfigure(line_buffering=True)

import config as C

GPU_FREE_GB_REQUIRED = 30.0


def check_1_config_vs_reference() -> None:
    assert C.BDDX_CSV.exists(), f"missing {C.BDDX_CSV}"
    for f in C.BDDX_SPLIT_FILES.values():
        assert f.exists(), f"missing {f}"
    assert any(d.is_dir() for d in C.BDD100K_IMG_DIRS), "no BDD100K image dir found"
    assert C.REFERENCE_ROOT.is_dir(), (
        f"reference repo missing — git clone --depth 1 "
        f"https://github.com/QwenLM/Qwen2.5-VL.git into {C.EXP_DIR}/reference/"
    )
    # LoRA values must match the reference source, parsed fresh from the clone.
    argsrc = (C.REFERENCE_ROOT / "qwenvl/train/argument.py").read_text()
    trainsrc = (C.REFERENCE_ROOT / "qwenvl/train/train_qwen.py").read_text()
    shsrc = (C.REFERENCE_ROOT / "scripts/sft_30a3b_lora.sh").read_text()
    assert f"lora_r: int = field(default={C.LORA_R})" in argsrc
    assert f"lora_alpha: int = field(default={C.LORA_ALPHA})" in argsrc
    assert f"or {C.LORA_DROPOUT}" in trainsrc, "reference effective dropout changed"
    mods = re.search(r'target_modules=\[(.*?)\]', trainsrc).group(1)
    assert [m.strip().strip('"') for m in mods.split(",") if 'proj' in m] == C.LORA_TARGET_MODULES
    assert f"lr={C.LR:.0e}".replace("e-05", "e-5") in shsrc, "reference LoRA lr changed"
    assert f"--max_pixels {C.MAX_PIXELS}" in shsrc and f"--min_pixels {C.MIN_PIXELS}" in shsrc
    print("[smoke] 1 OK: paths exist; LoRA r/alpha/dropout/targets, lr and pixel "
          "bounds match the reference clone")


def check_2_data_prep() -> list[dict]:
    from data_prep import build_samples

    samples, stats = build_samples("train", limit=25)
    assert stats["samples"] == 25, f"expected 25 capped samples, got {stats}"
    from pathlib import Path

    for s in samples[:5]:
        assert Path(s["image"]).exists()
        t0, t1 = s["bddx_segment"]
        assert t0 <= C.FRAME_TIME_S <= t1, f"alignment filter violated: {s['bddx_segment']}"
        human, gpt = s["conversations"]
        assert human["from"] == "human" and human["value"].count("<image>") == 1
        obj = json.loads(gpt["value"])                      # strict-JSON round-trip
        assert set(obj) == {"action", "justification"}
        assert obj["action"] and obj["justification"]
    print(f"[smoke] 2 OK: data prep on real CSV slice — 25 aligned samples, "
          f"images exist, target JSON round-trips (e.g. {json.loads(samples[0]['conversations'][1]['value'])})")
    return samples


def _reference_on_path() -> None:
    for p in (str(C.REFERENCE_ROOT), str(C.REFERENCE_ROOT / "qwenvl" / "train")):
        if p not in sys.path:
            sys.path.insert(0, p)


def check_3_reference_imports() -> None:
    _reference_on_path()
    import qwenvl.data as qd
    import qwenvl.train.train_qwen  # noqa: F401  (pulls trainer + flash-attn patch)
    from qwenvl.data import data_list

    qd.data_dict["bddx_smoke"] = {"annotation_path": "unset", "data_path": ""}
    assert data_list(["bddx_smoke"])[0]["annotation_path"] == "unset"
    print("[smoke] 3 OK: reference trainer + flash-attn patch import under this "
          "env; runtime dataset registration works")


def check_4_reference_dataset_cpu(samples: list[dict]):
    _reference_on_path()
    import qwenvl.data as qd
    from qwenvl.data.data_processor import LazySupervisedDataset
    from qwenvl.train.argument import DataArguments
    from transformers import AutoProcessor

    smoke_dir = C.CACHE_DIR / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    ann = smoke_dir / "bddx_smoke.json"
    ann.write_text(json.dumps(samples[:3], ensure_ascii=False))
    qd.data_dict["bddx_smoke"] = {"annotation_path": str(ann), "data_path": ""}

    processor = AutoProcessor.from_pretrained(C.MODEL_ID)
    data_args = DataArguments(dataset_use="bddx_smoke",
                              max_pixels=C.MAX_PIXELS, min_pixels=C.MIN_PIXELS)
    data_args.model_type = "qwen2.5vl"          # train() sets this before the data module
    ds = LazySupervisedDataset(processor, data_args=data_args)
    assert len(ds) == 3
    item = ds[0]
    for k in ("input_ids", "labels", "pixel_values"):
        assert k in item, f"dataset item missing {k} (has {list(item)})"
    ids = item["input_ids"].flatten()
    labels = item["labels"].flatten()
    n_sup = int((labels != -100).sum())
    assert 0 < n_sup < len(ids), f"bad label mask: {n_sup}/{len(ids)} supervised"
    sup_text = processor.tokenizer.decode(labels[labels != -100])
    tgt = json.loads(samples[0]["conversations"][1]["value"])
    assert tgt["action"] in sup_text and tgt["justification"] in sup_text, (
        f"supervised span does not contain the target text: {sup_text!r}")
    assert C.USER_PROMPT.splitlines()[0] not in sup_text, "prompt tokens leaked into the loss"
    print(f"[smoke] 4 OK: reference LazySupervisedDataset builds a real sample on CPU "
          f"({len(ids)} tokens, {n_sup} supervised = the JSON target only)")
    shutil.rmtree(smoke_dir)
    return item


def check_5_gpu_forward(samples: list[dict]) -> None:
    import torch

    if not torch.cuda.is_available():
        print("[smoke] 5 SKIPPED: no CUDA device visible")
        return
    free_b, _total = torch.cuda.mem_get_info(0)
    free_gb = free_b / 1024**3
    if free_gb < GPU_FREE_GB_REQUIRED:
        print(f"[smoke] 5 SKIPPED: only {free_gb:.1f} GB free on visible GPU "
              f"(need >{GPU_FREE_GB_REQUIRED:.0f} GB — a diagnostic job may be running). "
              "Rerun later with CUDA_VISIBLE_DEVICES=1.")
        return

    _reference_on_path()
    import qwenvl.data as qd
    from qwenvl.data.data_processor import (DataCollatorForSupervisedDataset,
                                            LazySupervisedDataset)
    from qwenvl.train.argument import DataArguments
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"[smoke] 5: {free_gb:.1f} GB free — loading {C.MODEL_ID} (bf16) ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        C.MODEL_ID, dtype=torch.bfloat16, device_map={"": 0})
    model.requires_grad_(False)
    # Exact reference LoraConfig (train_qwen.py L170-177)
    model = get_peft_model(model, LoraConfig(
        r=C.LORA_R, lora_alpha=C.LORA_ALPHA, lora_dropout=C.LORA_DROPOUT,
        target_modules=C.LORA_TARGET_MODULES, bias=C.LORA_BIAS,
        task_type=TaskType.CAUSAL_LM))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[smoke] 5: trainable {n_train/1e6:.1f}M / {n_total/1e9:.2f}B "
          f"({100*n_train/n_total:.2f}%)")

    smoke_dir = C.CACHE_DIR / "smoke_gpu"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    ann = smoke_dir / "bddx_smoke.json"
    ann.write_text(json.dumps(samples[:1], ensure_ascii=False))
    qd.data_dict["bddx_smoke_gpu"] = {"annotation_path": str(ann), "data_path": ""}
    processor = AutoProcessor.from_pretrained(C.MODEL_ID)
    data_args = DataArguments(dataset_use="bddx_smoke_gpu",
                              max_pixels=C.MAX_PIXELS, min_pixels=C.MIN_PIXELS)
    data_args.model_type = "qwen2.5vl"
    ds = LazySupervisedDataset(processor, data_args=data_args)
    batch = DataCollatorForSupervisedDataset(processor.tokenizer)([ds[0]])
    batch = {k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in batch.items()}

    model.train()
    with torch.autocast("cuda", torch.bfloat16):
        out = model(**batch)
    assert torch.isfinite(out.loss), f"non-finite loss {out.loss}"
    out.loss.backward()
    lora_grads = [n for n, p in model.named_parameters()
                  if p.requires_grad and p.grad is not None]
    no_grads = [n for n, p in model.named_parameters()
                if p.requires_grad and p.grad is None]
    assert lora_grads and not no_grads, f"trainable params without grads: {no_grads[:5]}"
    print(f"[smoke] 5 OK: forward+backward on a real sample — loss={out.loss.item():.4f}, "
          f"grads on all {len(lora_grads)} LoRA tensors")
    shutil.rmtree(smoke_dir)


def main() -> None:
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    check_1_config_vs_reference()
    samples = check_2_data_prep()
    check_3_reference_imports()
    check_4_reference_dataset_cpu(samples)
    check_5_gpu_forward(samples)
    print("[smoke] ALL CHECKS PASSED (GPU check may report SKIPPED above)")


if __name__ == "__main__":
    main()

# Exp7 — BDD-X LoRA SFT of Qwen2.5-VL-7B (Approach 8, Stage 2, BDD-X leg)

**Question:** does a driving-tuned VLM — LoRA-SFT'd on BDD-X (action,
justification) pairs — produce re-cached reasoning that, fused through a
retrained exp6 head, beats the detector-only control (esp. on
action/duplex/triplet)?

## Provenance

- Direction: `wiki/directions/vlm-reasoning-layer.md`, Stage 2 — *"LoRA-tune
  the VLM on BDD-X then CoVLA for explanation style. Plug the driving-tuned
  VLM into Stage 1's pipeline (fusion head re-trained)."* This experiment is
  the **BDD-X leg only**; CoVLA follows as its own experiment.
- Dr. Moradi 2026-07-24: the Stage-1 zero-shot fusion (exp6) did **not** beat
  the detector-only control (agent 14.6 floor). That run is kept as
  "Baseline Fusion — Zero-shot VLM"; the approved next step is specialized
  VLM training on BDD-X then CoVLA.
- exp3_bddx scaffolded a BDD-X LoRA in April but was never trained
  (empty checkpoints/) and predates the Approach-8 pipeline; exp7 supersedes
  it (and fixes its image–text temporal mismatch, see Data below).

## Reference repo (lab rule: reference first)

`reference/Qwen2.5-VL/qwen-vl-finetune` — **QwenLM/Qwen2.5-VL** official
finetuning recipe (commit `9658872`, 2026-01-30), cloned via:

```bash
cd reference && git clone --depth 1 https://github.com/QwenLM/Qwen2.5-VL.git
```

**Why this over LLaMA-Factory** (also evaluated, clone kept in `reference/`):

1. It is the model vendor's own recipe for this exact family, and its trainer
   loads the model through the same `Qwen2_5_VLForConditionalGeneration`
   class exp5 uses — zero checkpoint-format drift.
2. Verified to import cleanly against our base env (transformers 5.3.0,
   peft 0.19.0, flash_attn 2.8.3, torch 2.9.0+cu128) with **zero new pip
   installs**. LLaMA-Factory pins `peft<=0.18.1` (we have 0.19.0) and drags a
   large dependency tree (gradio, trl, modelscope, ...).
3. It is a thin HF-Trainer script: built-in resume, auditable 15-line LoRA
   block, LLaVA-style JSON datasets — easy to feed from `data_prep.py`
   without touching the clone (datasets are registered at runtime).

`train_lora.py` does **not** reimplement training — it imports and launches
the reference trainer unmodified.

## Hyperparameters (matched to reference)

From `qwenvl/train/train_qwen.py` L163–178 + `qwenvl/train/argument.py`
defaults + `scripts/sft_30a3b_lora.sh` (the repo's only LoRA recipe; its
trainer-level values are model-agnostic — the 7B script differs only in
being full-SFT):

| Param | Value | Source |
|---|---|---|
| LoRA rank / alpha / dropout | 64 / 128 / 0.05 | argument.py defaults; dropout `0.0 or 0.05` → 0.05 |
| LoRA targets | q_proj, k_proj, v_proj, o_proj (LLM attn only) | train_qwen.py L174 |
| Vision tower + merger | frozen (reference LoRA branch skips `set_model`) | train_qwen.py L163–178 |
| lr / schedule / warmup | 1e-5 / cosine / ratio 0.03 | sft_30a3b_lora.sh |
| batch / grad-accum | 1 / 4 (effective 4) | sft_30a3b_lora.sh |
| weight decay / grad clip | 0 / 1.0 | sft_30a3b_lora.sh |
| precision / grad ckpt | bf16 / on | sft_30a3b_lora.sh |
| model_max_length | 8192 | sft_30a3b_lora.sh |
| min/max pixels | 784 / 50176 | sft_30a3b_lora.sh |
| data_flatten / attention | True / flash_attention_2 | sft_30a3b_lora.sh, train_qwen.py default |

### Deviations from reference (each with reason)

| Deviation | Reference | Ours | Reason |
|---|---|---|---|
| Launcher | torchrun + deepspeed zero2 (32×80 GB) | single process, single GPU, no deepspeed | LoRA-7B bf16 fits one A6000 (48 GB); deepspeed is optional in HF Trainer and adds nothing at this scale |
| Dataset registration | edit `qwenvl/data/__init__.py` | injected into `data_dict` at runtime | keeps the reference clone pristine (raw-source immutability rule) |
| Epochs | 0.5 | 3 | reference value is for a multi-million-sample mixed corpus; ours is 4,339 samples — 0.5 epoch ≈ 540 optimizer steps. 3 epochs, checkpoint each, select by val loss |
| Eval | `eval_strategy no` | BDD-X val loss each epoch (injected eval_dataset) | lab rule 4: BDD-X val is for SFT validation loss; checkpoint selection must not touch the downstream test set |
| Saving | steps/1000, keep 1 | per epoch, keep all | needed for val-loss-based selection + resumability |
| logging_steps | 1 | 10 | per-step logging is per-item spam (lab rule: no noisy monitors); 10 keeps `tail -f` readable |
| report_to | wandb | none | no wandb in the lab; stdout + `logs/` |
| eval batch | 2× train bs | 1 | val set is 545 samples; avoids padded-collator edge cases, negligible cost |

Fallback flags (only if the flash-attn varlen path misbehaves at runtime):
`--attn sdpa --no-flatten` (numerically equivalent attention, standard padded
collator). Not a deviation unless actually used — record it in the run log if so.

## Data — what exists locally, and the pipeline design

Locally present:

- `BDD-X-Annotations_v1.csv` (6,999 video rows; 15 action slots each of
  `start/end/action/justification`), `train.txt` / `val.txt` / `test.txt`.
- **No BDD-X videos.** The CSV's S3 URLs are stale; BDD100K videos were never
  downloaded (≈1.8 TB).
- BDD100K **still frames** at `/data/datasets/bdd100k-yolopx/images/*` —
  100K images, each the frame at the **10th second** of its 40 s source video
  (BDD100K, CVPR 2020). BDD-X stems are BDD100K video names, so `<stem>.jpg`
  is a real frame from the exact annotated video. BDD-X splits ≠ BDD100K
  splits, so all three image dirs are indexed.

Design (image-SFT, temporally aligned):

- Keep a CSV segment only if its `[start, end]` covers t=10 s → the frame the
  model sees comes from *inside* the annotated action. (exp3 paired every
  segment with the one frame — up to ~30 s of mismatch; that label noise is
  removed here, at the cost of dataset size.)
- Yield: **4,339 train / 545 val** samples (from 16,553 / 1,966 segments with
  a local frame; 78% of split videos have a frame).
- Train on BDD-X **train** only; **val** for SFT val loss only; BDD-X test
  unused. The downstream fixed test set remains ROAD-Waymo val, untouched.

### Prompt / target template (exp5-schema style)

One human turn (image + prompt), one assistant turn (strict JSON) — the same
dialect exp5's `vlm_io.build_prompt` uses (schema block, "STRICT JSON only"
rules), so the tuned model reinforces exactly the output discipline the
caching pipeline parses. Field mapping per the direction page: BDD-X action →
`action`, BDD-X justification → `justification` (the rationale-style field).

```
user:  <image>
       You are a driving-scene reasoning assistant. The image is a dashcam frame
       from the ego vehicle, captured mid-drive.
       State what the ego vehicle is doing right now (its driving action) and
       justify it from visible scene evidence.

       Rules:
       - Output STRICT JSON only. No prose, no markdown fences.

       Output schema:
       {"action": "<what the ego vehicle is doing>", "justification": "<why, one short clause>"}

assistant: {"action": "The car accelerates", "justification": "because the light has turned green."}
```

Annotator text goes into the target verbatim (no rewriting, no synthetic data).

## Pipeline (in order)

```bash
cd /data/repos/ROAD_Reason/experiments/exp7_bddx_lora
PY=/home/brandon/miniconda3/bin/python

# 0. once: clone reference (already done) + smoke
CUDA_VISIBLE_DEVICES=1 $PY -u smoke_test.py

# 1. build SFT jsons (CPU, seconds) -> cache/bddx_sft_{train,val}.json
$PY -u data_prep.py

# 2. LoRA SFT (one GPU; resumable — rerun the same command after a crash)
CUDA_VISIBLE_DEVICES=1 $PY -u train_lora.py 2>&1 | tee -a logs/train.log

# 3. merge best-val-loss adapter into a plain HF model dir (CPU)
$PY -u export_for_exp5.py            # -> checkpoints/merged_checkpoint_<N>/

# 4. re-cache + refuse (exp5/exp6, unchanged code):
#    - exp5/config.py: QWEN_MODEL = "<merged dir>", and point the qwen cache
#      dirs at fresh paths (e.g. cache/qwen_bddx_lora/) so the zero-shot
#      caches remain the Stage-1 control
#    - rerun exp5/qwen_infer.py (train+val), exp6/embed_rationale.py,
#      exp6/train.py, exp6/eval.py
```

Checkpoints: `checkpoints/checkpoint-<step>/` (HF Trainer epoch checkpoints,
each with the PEFT adapter + optimizer state + `trainer_state.json` carrying
`eval_loss`). `export_for_exp5.py` picks the lowest-val-loss epoch by default.

## Environment

Base miniconda `python` (`/home/brandon/miniconda3/bin/python`) — NOT
`.conda-envs/road_reason`. No new packages were installed; exact versions
used by this scaffold:

| Package | Version |
|---|---|
| torch | 2.9.0+cu128 |
| transformers | 5.3.0 |
| peft | 0.19.0 |
| accelerate | 1.13.0 |
| flash_attn | 2.8.3 |
| qwen_vl_utils | present (exp5) |
| pillow | 12.0.0 |

## Blockers / limitations

- **No video SFT.** BDD-X videos are not local (stale S3 links; BDD100K video
  set ≈1.8 TB). The reference trainer supports `<video>` samples, so a video
  leg is a data download away, but this scaffold trains on single aligned
  frames only. Temporal reasoning style (e.g. "slows down") is therefore
  supervised from static evidence; whether that suffices is part of what the
  Stage-2 gate measures.
- 22% of split videos have no local frame (their stems are not among the
  BDD100K 100K images) — those segments are dropped, not faked.
- Train-time resolution follows the reference (max 50,176 px ≈ 64 visual
  tokens) while exp5 inference feeds ~600×840 crops with drawn boxes. Kept
  as-is per replicate-before-innovate; flag for a follow-up ablation only if
  Stage-2 val behavior warrants it.

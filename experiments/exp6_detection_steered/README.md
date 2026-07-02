# Exp6 — Detection-Steered Fusion (Approach 8, Stage 1 trained)

**Question:** does fusing Qwen's detection-steered reasoning with the frozen
3D-RetinaNet's logits — through a trained few-layer head — beat the detector
alone?

## Provenance

Dr. Moradi's direction, verbatim:

- 2026-06-02: *"use language embedding to fused with output of 3d retina net
  and then re predict with adding few layer to be trained"*
- 2026-06-11: *"Qwen can be reasoning given detections from retinaNet ... not
  from ground truth ... it is psedu-label"* — the VLM is **steered by
  detections** (RetinaNet's predicted boxes overlaid on the frame), never GT.
- 2026-06-11: *"VLM and retina fusion should be trained on training, not test
  ... test always the same in all experiments"*

This implements **architecture (a)** of his 2026-06-11 note (new trainable
layers on top of the fused representation). Architecture (b) — fuse inside
RetinaNet and train its remaining layers — is the sibling variant, not built
here.

## Architecture

```
                 cached, frozen                            trained (~0.9M)
┌─────────────────────────────────────────────┐   ┌────────────────────────┐
│ 3D-RetinaNet (17.76%) ─► top-K boxes ───────┼─┐ │                        │
│                       └► flat logits [184] ─┼─┼─┼─► det_proj ──┐         │
│                                             │ │ │              │         │
│ boxes overlaid on frame                     │ │ │              ▼         │
│   └► Qwen2.5-VL-7B ─► JSON per box          │ │ │   concat ─► LN ─ MLP   │
│         ├─ agent/actions/locations/risk ────┼─┼─┼─► struct_proj │        │
│         └─ rationale ─► SigLIP text [1152] ─┼─┼─┼─► rat_proj ─┘ │        │
└─────────────────────────────────────────────┘ │ │              ▼         │
                                                └─┼──────────► + delta ────┼─► fused logits [184]
                                                  └────────────────────────┘
```

- **Unit of fusion:** one detector box on one frame (matches the Qwen cache
  granularity: exp5 classified the top-40 boxes per frame).
- **Language embedding** = vectorized structured fields (54-d) ⊕ frozen
  SigLIP-SO400M text embedding of the rationale sentence (1152-d + has-flag).
  SigLIP is the encoder already used in exp4's smoke path — no new dependency.
- **Floor guarantee:** the delta layer is zero-initialized, so the model's
  output at init *is* the detector's logits. Training starts at the control.
- **Targets:** GT flat 184-multi-hot assigned by IoU≥0.5 greedy-argmax match
  of detector boxes to GT boxes; unmatched boxes get all-zeros (background).
  Focal-on-all + flat per-class alphas (exp2f/exp4 recipe, imported from exp4).

## The three-way comparison (same top-K rows, same evaluator)

| Run | What it shows |
|-----|---------------|
| `eval.py --detector-only` | RetinaNet logits over top-K boxes — the control/floor |
| exp5 `results_nms_fixed.json` | zero-shot Qwen hard labels over the same boxes (agent 5.60%) |
| `eval.py --ckpt ...` | detection-steered fusion (this experiment) |

None of these is the 17.76% all-anchor number — that baseline scores every
anchor, these score the top-K post-NMS set. The Stage-1 gate is **fused >
detector-only on identical rows** (and per-head lift on duplex/triplet where
language should matter most).

## Pipeline

```bash
# 1. Detections + logits + GT (GPU; val ≈ same cost as exp5's dump)
CUDA_VISIBLE_DEVICES=0 python -u dump_detections.py --split val
CUDA_VISIBLE_DEVICES=0 python -u dump_detections.py --split train   # stride 16

# 2. Qwen JSON for the TRAIN split (val is already cached NMS-fixed in exp5).
#    THE expensive step — shard it like the val run was:
cd ../exp5_qwen_reasoning
CUDA_VISIBLE_DEVICES=0 python -u qwen_infer.py --split train \
    --detections ../exp6_detection_steered/cache/detections_train.pkl \
    --num-shards 2 --shard 0   # + --shard 1 on GPU 1

# 3. Rationale embeddings (GPU, minutes)
cd ../exp6_detection_steered
CUDA_VISIBLE_DEVICES=0 python -u embed_rationale.py --split val
CUDA_VISIBLE_DEVICES=0 python -u embed_rationale.py --split train

# 4. Train fusion head (fast — cached vectors, no video decoding)
python -u train.py

# 5. Score (control first, then fused)
python -u eval.py --detector-only --out results_detector_only.json
python -u eval.py --ckpt checkpoints/fusion_ep020.pth --out results_fused.json
```

`smoke_test.py` runs with no caches (model + layout checks) and upgrades to a
real end-to-end micro-step once the val caches exist.

## Notes

- **Python env:** run with the base miniconda `python` (torch 2.9 + easydict +
  sentencepiece), the same interpreter exp5 used — NOT the repo's
  `.conda-envs/road_reason` env, which lacks the baseline's deps.

- exp6's dumped boxes are bit-identical to exp5's NMS-fixed dump (same frozen
  weights, same conf/NMS/top-K path) — that is what licenses reusing exp5's
  val Qwen cache. `dataset.py` asserts the pkl carries logits.
- Train/test discipline: fusion trains on `train` (RetinaNet supplying its
  boxes is train-trained — Moradi's approved fallback); `val` is the fixed
  test split for every experiment in the series.
- No early stopping on the test split: fixed epochs, checkpoint every epoch,
  `eval.py` scores whichever checkpoint — the epoch curve goes in the report.
- Deferred to sibling experiments (per Moradi, run one variable at a time):
  prompt-side ROAD-R negative constraints, loss-side T-norm penalty
  (`tnorm_loss.py`), architecture (b), Stage-2 LoRA on BDD-X/CoVLA.

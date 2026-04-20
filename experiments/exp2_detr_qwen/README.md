# Exp2: DETR-Style Detection on Qwen ROAD-Waymo Features

This experiment replaces `exp1b`'s FCOS-style dense token detector with a DETR-style set-prediction decoder on the same Qwen2.5-VL visual features.

## Goal

- Keep the strong VLM semantics we saw in `exp1` and the internal `exp1b` eval.
- Replace the weak per-token localization path with learnable queries + Hungarian matching.
- Produce official-style frame-mAP numbers that are more comparable to the 3D-RetinaNet baseline.

## What Is Reused

- Qwen ViT + LoRA path: `exp1b_road_r/model.py`
- ROAD-Waymo dataset and preprocessing: `exp1_road_r/dataset.py`, `exp1_road_r/train.py`
- t-norm constraint: `tnorm_loss.py`
- Baseline-compatible frame evaluator pattern: `exp1b_road_r/eval_baseline_compat.py`

## What Is New

- `model.py`: projection + clip-level DETR decoder + tube queries
- `matcher.py`: Hungarian matching
- `losses.py`: set-style classification/box/GIoU/t-norm loss
- `eval_baseline_compat.py`: DETR-style export to baseline frame-mAP

## Important Current Assumption

The existing dataset class does not expose annotation keys across frames, so GT tubes are currently grouped with a greedy IoU-based heuristic in `losses.py`.

That means:

- frame-level training/eval scaffolding is real and usable
- tube grouping is a best-effort approximation for now
- if we later expose true annotation keys from the JSON, this experiment gets stronger immediately

## Suggested First Checks

1. `python -u experiments/exp2_detr_qwen/train.py`
2. Confirm first-batch loss is finite and matched query count is sensible.
3. Run `python -u experiments/exp2_detr_qwen/eval.py --ckpt .../best.pt`
4. Run `python -u experiments/exp2_detr_qwen/eval_baseline_compat.py --ckpt .../best.pt --mode frame`


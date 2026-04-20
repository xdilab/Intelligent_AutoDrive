#!/bin/bash
set -euo pipefail

cd /data/repos/ROAD_Reason

echo "=== Exp2: DETR-style detection on Qwen ROAD-Waymo features ==="
echo "[1/3] Training..."
python -u experiments/exp2_detr_qwen/train.py 2>&1 | tee experiments/exp2_detr_qwen/logs/train.log

echo "[2/3] Baseline-compatible frame mAP..."
python -u experiments/exp2_detr_qwen/eval_baseline_compat.py \
  --ckpt experiments/exp2_detr_qwen/checkpoints/best.pt \
  --mode frame \
  2>&1 | tee experiments/exp2_detr_qwen/logs/eval_fmap.log

echo "[3/3] Approximate tube eval..."
python -u experiments/exp2_detr_qwen/eval_baseline_compat.py \
  --ckpt experiments/exp2_detr_qwen/checkpoints/best.pt \
  --mode video \
  2>&1 | tee experiments/exp2_detr_qwen/logs/eval_vmap.log


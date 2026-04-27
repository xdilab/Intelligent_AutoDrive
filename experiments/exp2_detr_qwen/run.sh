#!/bin/bash
set -euo pipefail

cd /data/repos/ROAD_Reason
PYTHON_BIN="/data/repos/ROAD_Reason/.conda-envs/road_reason/bin/python"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

echo "=== Exp2: DETR-style detection on Qwen ROAD-Waymo features ==="
echo "[1/3] Training..."
"${PYTHON_BIN}" -u experiments/exp2_detr_qwen/train.py 2>&1 | tee experiments/exp2_detr_qwen/logs/train.log

echo "[2/3] Baseline-compatible frame mAP..."
"${PYTHON_BIN}" -u experiments/exp2_detr_qwen/eval_baseline_compat.py \
  --ckpt experiments/exp2_detr_qwen/checkpoints/best.pt \
  --mode frame \
  2>&1 | tee experiments/exp2_detr_qwen/logs/eval_fmap.log

echo "[3/3] Approximate tube eval..."
"${PYTHON_BIN}" -u experiments/exp2_detr_qwen/eval_baseline_compat.py \
  --ckpt experiments/exp2_detr_qwen/checkpoints/best.pt \
  --mode video \
  2>&1 | tee experiments/exp2_detr_qwen/logs/eval_vmap.log

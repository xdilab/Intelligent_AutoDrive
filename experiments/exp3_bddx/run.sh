#!/bin/bash
set -euo pipefail

cd /data/repos/ROAD_Reason
PYTHON="/data/repos/ROAD_Reason/.conda-envs/road_reason/bin/python"

echo "=== Exp3: BDD-X captioning fine-tune on Qwen2.5-VL ==="

mkdir -p experiments/exp3_bddx/logs experiments/exp3_bddx/checkpoints

echo "[0/2] Installing eval dependencies ..."
"$PYTHON" -m pip install sacrebleu -q

echo "[1/2] Training (${MAX_EPOCHS:-3} epochs) ..."
"$PYTHON" -u experiments/exp3_bddx/train.py \
    2>&1 | tee experiments/exp3_bddx/logs/train.log

echo "[2/2] Evaluating on test split ..."
"$PYTHON" -u experiments/exp3_bddx/eval.py \
    --ckpt experiments/exp3_bddx/checkpoints/best.pt \
    --split test \
    2>&1 | tee experiments/exp3_bddx/logs/eval_test.log

echo "Done. Results in experiments/exp3_bddx/logs/"

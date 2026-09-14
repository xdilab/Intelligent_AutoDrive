#!/bin/bash
#SBATCH --job-name=adapt-cache
#SBATCH --partition=gpu-hp
#SBATCH --qos=ncat_h200_hp
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/work/bbyrd1/adapted-road-20260911/results/logs/cache-%A-%a.log
#SBATCH --gres=gpu:h200:1
#SBATCH --array=0-11%4
set -euo pipefail
export COMPARISON_ROOT=/work/bbyrd1/adapted-road-20260911
export ADAPTED_CHECKPOINT=/work/bbyrd1/bddx-pilot-20260911/run-seed0/epoch-2.pt
export HF_HOME=/work/bbyrd1/road_crop/hf_cache
export HF_HUB_OFFLINE=1
export ROADCROP_FRAMES=$COMPARISON_ROOT/frames
export ROADCROP_JSON=$COMPARISON_ROOT/road_waymo_trainval_v1.1.json
export ROADCROP_DUMPS=/work/bbyrd1/road_crop/dumps
export ROADCROP_OUT=$COMPARISON_ROOT/cache
python=/work/bbyrd1/road_crop/py311/bin/python
if [ "$SLURM_ARRAY_TASK_ID" -lt 8 ]; then
  split=train; n=8; i=$SLURM_ARRAY_TASK_ID
else
  split=val; n=4; i=$((SLURM_ARRAY_TASK_ID-8))
fi
"$python" -u "$COMPARISON_ROOT/code/cache-paired.py" --split "$split" --nshard "$n" --ishard "$i"

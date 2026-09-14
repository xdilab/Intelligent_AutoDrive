#!/bin/bash
#SBATCH --job-name=adapt-train
#SBATCH --partition=gpu-hp
#SBATCH --qos=ncat_h200_hp
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/work/bbyrd1/adapted-road-20260911/results/logs/train-%A-%a.log
#SBATCH --gres=gpu:h200:1
#SBATCH --array=0-1%2
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
conditions=(original adapted)
condition=${conditions[$SLURM_ARRAY_TASK_ID]}
"$python" -u "$COMPARISON_ROOT/code/train-repeats.py" --data "$COMPARISON_ROOT/data/$condition" --out "$COMPARISON_ROOT/runs/$condition" --seeds 0 1 2

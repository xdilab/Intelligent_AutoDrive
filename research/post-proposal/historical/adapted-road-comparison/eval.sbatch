#!/bin/bash
#SBATCH --job-name=adapt-eval
#SBATCH --partition=gpu-hp
#SBATCH --qos=ncat_h200_hp
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/work/bbyrd1/adapted-road-20260911/results/logs/eval-%A-%a.log
#SBATCH --array=0-5%2
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
condition=${conditions[$((SLURM_ARRAY_TASK_ID/3))]}
seed=$((SLURM_ARRAY_TASK_ID%3))
"$python" -u "$COMPARISON_ROOT/code/evaluate-study.py" --data "$COMPARISON_ROOT/data/$condition" --runs "$COMPARISON_ROOT/runs/$condition" --out "$COMPARISON_ROOT/results/metrics/$condition" --manifest "$COMPARISON_ROOT/code/shared-frames.json" --workers 8 --only "seed${seed}-stage5" "seed${seed}-stage6"

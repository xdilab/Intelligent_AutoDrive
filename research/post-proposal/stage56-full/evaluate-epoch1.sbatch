#!/bin/bash
#SBATCH --job-name=full56-epoch1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/work/bbyrd1/stage56-full-20260914/results/%x-%A-%a.log
#SBATCH --partition=gpu-hp
#SBATCH --qos=ncat_h200_hp
#SBATCH --gres=gpu:h200:1
#SBATCH --time=7-00:00:00
#SBATCH --array=0-5%2
set -euo pipefail
export ROAD_FRAME_ROOT=/work/bbyrd1/stage56-full-20260914/frames
export HF_HOME=/work/bbyrd1/road_crop/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONPATH=/work/bbyrd1/bddx-pilot-20260911/packages
export PATH=/work/bbyrd1/road_crop/py311/bin:$PATH
python /work/bbyrd1/stage56-full-20260914/code/guard.py
stages=(stage5 stage6)
conditions=(frozen classification contrastive)
python -u /work/bbyrd1/stage56-full-20260914/code/evaluate-epoch1.py --stage "${stages[$((SLURM_ARRAY_TASK_ID / 3))]}" --condition "${conditions[$((SLURM_ARRAY_TASK_ID % 3))]}"

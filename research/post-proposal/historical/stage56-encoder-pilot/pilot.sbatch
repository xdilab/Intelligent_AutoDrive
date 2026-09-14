#!/bin/bash
#SBATCH --job-name=stage56-encoder
#SBATCH --partition=gpu-hp
#SBATCH --qos=ncat_h200_hp
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --output=/work/bbyrd1/stage56-encoder-pilot-20260914/results/%x-%A-%a.log
#SBATCH --array=0-5%2
set -euo pipefail
export HF_HOME=/work/bbyrd1/road_crop/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONPATH=/work/bbyrd1/bddx-pilot-20260911/packages
export PATH=/work/bbyrd1/road_crop/py311/bin:$PATH
python /work/bbyrd1/stage56-encoder-pilot-20260914/code/guard.py
stages=(stage5 stage6)
conditions=(frozen classification contrastive)
python -u /work/bbyrd1/stage56-encoder-pilot-20260914/code/experiment.py --root /work/bbyrd1/stage56-encoder-pilot-20260914 --stage "${stages[$((SLURM_ARRAY_TASK_ID / 3))]}" --condition "${conditions[$((SLURM_ARRAY_TASK_ID % 3))]}"

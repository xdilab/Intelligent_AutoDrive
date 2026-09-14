#!/bin/bash
#SBATCH --job-name=road-contrast-pilot
#SBATCH --partition=gpu-hp
#SBATCH --qos=ncat_h200_hp
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/work/bbyrd1/road-contrastive-pilot-20260914/results/pilot-%A-%a.log
#SBATCH --gres=gpu:h200:1
#SBATCH --array=0-2%2
set -euo pipefail
export HF_HOME=/work/bbyrd1/road_crop/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONPATH=/work/bbyrd1/bddx-pilot-20260911/packages
export PATH=/work/bbyrd1/road_crop/py311/bin:$PATH
conditions=(frozen classification contrastive)
python -u /work/bbyrd1/road-contrastive-pilot-20260914/code/experiment.py --root /work/bbyrd1/road-contrastive-pilot-20260914 --condition "${conditions[$SLURM_ARRAY_TASK_ID]}"

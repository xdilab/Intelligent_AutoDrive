#!/bin/bash
#SBATCH --job-name=road-contrast-prepare
#SBATCH --partition=gpu-hp
#SBATCH --qos=ncat_h200_hp
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/work/bbyrd1/road-contrastive-pilot-20260914/results/prepare-%A-%a.log
set -euo pipefail
export HF_HOME=/work/bbyrd1/road_crop/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONPATH=/work/bbyrd1/bddx-pilot-20260911/packages
export PATH=/work/bbyrd1/road_crop/py311/bin:$PATH
python -u /work/bbyrd1/road-contrastive-pilot-20260914/code/prepare.py

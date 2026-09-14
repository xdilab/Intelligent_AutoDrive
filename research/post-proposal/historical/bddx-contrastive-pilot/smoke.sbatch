#!/bin/bash
#SBATCH --job-name=bddx-tower-smoke
#SBATCH --partition=gpu-hp
#SBATCH --qos=ncat_h200_hp
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/work/bbyrd1/bddx-pilot-20260911/logs/smoke-%j.log
set -euo pipefail
export HF_HOME=/work/bbyrd1/road_crop/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONPATH=/work/bbyrd1/bddx-pilot-20260911/packages
/work/bbyrd1/road_crop/py311/bin/python -u /work/bbyrd1/bddx-pilot-20260911/code/smoke-model.py

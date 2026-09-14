#!/bin/bash
#SBATCH --job-name=full56-prepare
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/work/bbyrd1/stage56-full-20260914/results/%x-%A-%a.log
#SBATCH --time=03:00:00
set -euo pipefail
export HF_HOME=/work/bbyrd1/road_crop/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONPATH=/work/bbyrd1/bddx-pilot-20260911/packages
export PATH=/work/bbyrd1/road_crop/py311/bin:$PATH
python -u /work/bbyrd1/stage56-full-20260914/code/prepare.py

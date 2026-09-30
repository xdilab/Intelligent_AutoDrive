#!/bin/bash
#SBATCH --job-name=full56-1b-assets
#SBATCH --partition=common
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/work/bbyrd1/contextual-roi-1b-20260930/results/%x-%j.log
set -euo pipefail
export PYTHONPATH=/work/bbyrd1/bddx-pilot-20260911/packages
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_DISABLE_PROGRESS_BARS=1
/work/bbyrd1/road_crop/py311/bin/python -u /work/bbyrd1/contextual-roi-1b-20260930/code/prepare_assets.py --root /work/bbyrd1/contextual-roi-1b-20260930

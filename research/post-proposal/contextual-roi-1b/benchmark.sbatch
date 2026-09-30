#!/bin/bash
#SBATCH --job-name=full56-1b-benchmark
#SBATCH --partition=gpu-hp
#SBATCH --qos=ncat_h200_hp
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/work/bbyrd1/contextual-roi-1b-20260930/results/%x-%j.log
set -euo pipefail
export HF_HOME=/work/bbyrd1/road_crop/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONPATH=/work/bbyrd1/bddx-pilot-20260911/packages
export OMP_NUM_THREADS=8
/work/bbyrd1/road_crop/py311/bin/python -u /work/bbyrd1/contextual-roi-1b-20260930/code/benchmark_vision.py --root /work/bbyrd1/contextual-roi-1b-20260930

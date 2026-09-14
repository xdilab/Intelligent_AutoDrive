#!/bin/bash
#SBATCH --job-name=language-trace
#SBATCH --partition=gpu-hp
#SBATCH --qos=ncat_h200_hp
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/work/bbyrd1/language-fusion-diagnostics-20260914/results/trace-%j.log
set -euo pipefail
/work/bbyrd1/road_crop/py311/bin/python -u /work/bbyrd1/language-fusion-diagnostics-20260914/code/trace.py --study /work/bbyrd1/proposal-study-20260910 --out /work/bbyrd1/language-fusion-diagnostics-20260914/results

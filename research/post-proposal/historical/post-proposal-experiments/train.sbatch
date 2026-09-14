#!/bin/bash
#SBATCH --job-name=proposal-seeds
#SBATCH --partition=gpu-hp
#SBATCH --qos=ncat_h200_hp
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/work/bbyrd1/proposal-study-20260910/results/train-%j.log
set -euo pipefail
root=/work/bbyrd1/proposal-study-20260910
/work/bbyrd1/road_crop/py311/bin/python -u "$root/code/train-repeats.py" --data "$root/data" --out "$root/runs" --seeds 0 1 2

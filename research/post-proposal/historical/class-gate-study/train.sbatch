#!/bin/bash
#SBATCH --job-name=gate-train
#SBATCH --partition=gpu-hp
#SBATCH --qos=ncat_h200_hp
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=0-2%2
#SBATCH --output=/work/bbyrd1/class-gate-study-20260914/results/train-%A-%a.log
set -euo pipefail
/work/bbyrd1/road_crop/py311/bin/python -u /work/bbyrd1/class-gate-study-20260914/code/train.py --root /work/bbyrd1/class-gate-study-20260914 --seed "$SLURM_ARRAY_TASK_ID"

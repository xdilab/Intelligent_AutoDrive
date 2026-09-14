#!/bin/bash
#SBATCH --job-name=gate-eval
#SBATCH --partition=gpu-hp
#SBATCH --qos=ncat_h200_hp
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%2
#SBATCH --output=/work/bbyrd1/class-gate-study-20260914/results/eval-%A-%a.log
set -euo pipefail
/work/bbyrd1/road_crop/py311/bin/python -u /work/bbyrd1/class-gate-study-20260914/code/evaluate.py --root /work/bbyrd1/class-gate-study-20260914 --seed "$SLURM_ARRAY_TASK_ID" --workers 8

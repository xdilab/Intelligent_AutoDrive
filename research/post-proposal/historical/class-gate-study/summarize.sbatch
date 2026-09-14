#!/bin/bash
#SBATCH --job-name=gate-summary
#SBATCH --partition=gpu-hp
#SBATCH --qos=ncat_h200_hp
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/work/bbyrd1/class-gate-study-20260914/results/summary-%j.log
set -euo pipefail
/work/bbyrd1/road_crop/py311/bin/python -u /work/bbyrd1/class-gate-study-20260914/code/summarize.py --root /work/bbyrd1/class-gate-study-20260914

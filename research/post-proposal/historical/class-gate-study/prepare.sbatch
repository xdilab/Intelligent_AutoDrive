#!/bin/bash
#SBATCH --job-name=gate-prepare
#SBATCH --partition=gpu-hp
#SBATCH --qos=ncat_h200_hp
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/work/bbyrd1/class-gate-study-20260914/results/prepare-%j.log
set -euo pipefail
/work/bbyrd1/road_crop/py311/bin/python -u /work/bbyrd1/class-gate-study-20260914/code/prepare.py --root /work/bbyrd1/class-gate-study-20260914

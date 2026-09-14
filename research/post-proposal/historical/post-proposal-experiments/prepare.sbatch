#!/bin/bash
#SBATCH --job-name=proposal-verify
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/work/bbyrd1/proposal-study-20260910/results/prepare-%j.log
set -euo pipefail
root=/work/bbyrd1/proposal-study-20260910
/work/bbyrd1/road_crop/py311/bin/python -u "$root/code/merge-verified.py" "$root"

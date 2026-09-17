#!/bin/bash
#SBATCH --job-name=full56-epoch1-ready
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=7-00:00:00
#SBATCH --output=/work/bbyrd1/stage56-full-20260914/results/%x-%j.log
set -euo pipefail
python3 -u /work/bbyrd1/stage56-full-20260914/code/wait-epoch1.py

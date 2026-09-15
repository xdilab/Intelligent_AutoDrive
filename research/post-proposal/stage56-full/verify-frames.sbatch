#!/bin/bash
#SBATCH --job-name=full56-frame-recovery
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:30:00
#SBATCH --output=/work/bbyrd1/stage56-full-20260914/results/frame-recovery-%j.log
set -euo pipefail
python3 /work/bbyrd1/stage56-full-20260914/code/verify-frames.py

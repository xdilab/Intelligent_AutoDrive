#!/bin/bash
#SBATCH --job-name=gate-audit
#SBATCH --partition=gpu-hp
#SBATCH --qos=ncat_h200_hp
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-2%2
#SBATCH --output=/work/bbyrd1/gate-objective-audit-20260914/results/audit-%A-%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
/work/bbyrd1/road_crop/py311/bin/python -u /work/bbyrd1/gate-objective-audit-20260914/code/audit.py --root /work/bbyrd1/gate-objective-audit-20260914 --seed "$SLURM_ARRAY_TASK_ID"

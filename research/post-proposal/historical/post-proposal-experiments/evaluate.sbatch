#!/bin/bash
#SBATCH --job-name=proposal-eval
#SBATCH --partition=gpu-hp
#SBATCH --qos=ncat_h200_hp
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-3%2
#SBATCH --output=/work/bbyrd1/proposal-study-20260910/results/eval-%A-%a.log
set -euo pipefail
root=/work/bbyrd1/proposal-study-20260910
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  names=(stage0 stage1 stage2 stage3)
else
  seed=$((SLURM_ARRAY_TASK_ID-1))
  names=("seed${seed}-head-flat" "seed${seed}-head-phrase" "seed${seed}-head-shuffled" "seed${seed}-stage5" "seed${seed}-stage6" "seed${seed}-fusion-flat-evidence" "seed${seed}-fusion-shuffled")
fi
/work/bbyrd1/road_crop/py311/bin/python -u "$root/code/evaluate-study.py" --data "$root/data" --runs "$root/runs" --out "$root/results/metrics" --manifest "$root/code/shared-frames.json" --workers 8 --only "${names[@]}"

#!/bin/bash
set -euo pipefail
study=/work/bbyrd1/contextual-road-transfer-20261005
py=/work/bbyrd1/road_crop/py311/bin/python
export HF_HOME=/work/bbyrd1/road_crop/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONPATH=/work/bbyrd1/bddx-pilot-20260911/packages
export OMP_NUM_THREADS=8
export ROAD_FRAME_ROOT=/work/bbyrd1/contextual-roi-1b-20260930/frames
cd "$study/code"
phase="$1"
arm="${2:-}"
root="$study/$arm"
case "$phase" in
 prepare) "$py" -u prepare_full.py --root "$study" ;;
 benchmark) "$py" -u benchmark_vision.py --root "$study" ;;
 cache)
  arms=(frozen classification contrastive)
  root="$study/${arms[$((SLURM_ARRAY_TASK_ID / 8))]}"
  "$py" -u cache_full.py --root "$root" --shard "$((SLURM_ARRAY_TASK_ID % 8))" --shards 8 ;;
 compact) "$py" -u compact_full.py --root "$root" ;;
 preflight) "$py" -u preflight_full.py --root "$root" ;;
 train)
  seed=$((SLURM_ARRAY_TASK_ID / 2)); weight=0
  if ((SLURM_ARRAY_TASK_ID % 2)); then weight=.001; fi
  "$py" -u train_cached.py --root "$root" --fusion attention --contrastive "$weight" --seed "$seed" ;;
 blend) "$py" -u prepare_blend.py --root "$root" --seed "$SLURM_ARRAY_TASK_ID" ;;
 evaluate)
  names=(classification contrastive-all184 global)
  run="attention-${names[$((SLURM_ARRAY_TASK_ID % 3))]}-dcb-seed$((SLURM_ARRAY_TASK_ID / 3))"
  "$py" -u evaluate_cached.py --root "$root" --run "$run" ;;
 summarize) "$py" -u summarize_full.py --root "$root" ;;
 *) exit 2 ;;
esac

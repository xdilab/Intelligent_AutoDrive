#!/bin/bash
set -euo pipefail
root=/work/bbyrd1/contextual-roi-1b-20260930
py=/work/bbyrd1/road_crop/py311/bin/python
export HF_HOME=/work/bbyrd1/road_crop/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONPATH=/work/bbyrd1/bddx-pilot-20260911/packages
export OMP_NUM_THREADS=8
export ROAD_FRAME_ROOT="$root/frames"
cd "$root/code"
case "$1" in
 prepare) "$py" -u prepare_full.py --root "$root" ;;
 cache) "$py" -u cache_full.py --root "$root" --shard "$SLURM_ARRAY_TASK_ID" --shards 8 ;;
 compact) "$py" -u compact_full.py --root "$root" ;;
 summarize) "$py" -u summarize_full.py --root "$root" ;;
 preflight) "$py" -u preflight_full.py --root "$root" ;;
 train)
   seed=$((SLURM_ARRAY_TASK_ID / 2)); weight=0
   if (( SLURM_ARRAY_TASK_ID % 2 )); then weight=.001; fi
   "$py" -u train_cached.py --root "$root" --fusion attention --contrastive "$weight" --seed "$seed"
   ;;
 blend) "$py" -u prepare_blend.py --root "$root" --seed "$SLURM_ARRAY_TASK_ID" ;;
 evaluate)
   seed=$((SLURM_ARRAY_TASK_ID / 3)); names=(classification contrastive-all184 global)
   run="attention-${names[$((SLURM_ARRAY_TASK_ID % 3))]}-dcb-seed${seed}"
   "$py" -u evaluate_cached.py --root "$root" --run "$run"
   ;;
 *) exit 2 ;;
esac

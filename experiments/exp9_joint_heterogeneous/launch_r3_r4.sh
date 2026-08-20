#!/bin/bash
# Queue R3+R4 behind the running R2 evals: wait for GPUs, smoke the warmup
# path, pick best lambda from the completed R2 sweep (duplex+triplet, the gate
# heads), then launch R3 (GPU0) and R4 (GPU1) with head warmup.
set -u
E=/data/repos/ROAD_Reason/experiments/exp9_joint_heterogeneous
PY=/home/brandon/miniconda3/bin/python

# 1. wait for the eval chains (PIDs passed as args) to release the GPUs
for pid in "$@"; do
  while kill -0 "$pid" 2>/dev/null; do sleep 60; done
done
echo "[launch] evals done $(date '+%F %T')"

# 2. smoke the warmup code path: 2 cycles, warmup=1 exercises both branches
cd "$E" || exit 1
CUDA_VISIBLE_DEVICES=0 $PY -u train.py --corpora road,bddx,covla --lam 1.0 \
  --max-cycles 2 --head-warmup 1 --tag warmsmoke > logs/train_warmsmoke.log 2>&1
if [ $? -ne 0 ] || ! grep -q "warmup complete" logs/train_warmsmoke.log; then
  echo "[launch] WARMUP SMOKE FAILED — aborting R3/R4. See logs/train_warmsmoke.log"
  exit 1
fi
echo "[launch] warmup smoke passed"

# 3. pick lambda: argmax of duplex+triplet over the R2 sweep results
LAM=$($PY - << 'EOF'
import json
best, best_lam = -1.0, None
for tag, lam in (("r2_lam0.1", 0.1), ("r2_lam1", 1.0), ("r2_lam10", 10.0)):
    try:
        s = json.load(open(f"results_{tag}.json"))["summary"]
    except FileNotFoundError:
        continue
    score = s["duplex"] + s["triplet"]
    if score > best:
        best, best_lam = score, lam
print(best_lam)
EOF
)
if [ -z "$LAM" ] || [ "$LAM" = "None" ]; then
  echo "[launch] NO R2 RESULTS FOUND — aborting"; exit 1
fi
echo "[launch] chosen lambda for R4: $LAM"

# 4. launch R3 and R4 with warmup
CUDA_VISIBLE_DEVICES=0 nohup $PY -u train.py --corpora road,bddx,covla \
  --lam 0 --tag r3 > logs/train_r3.log 2>&1 &
R3=$!
CUDA_VISIBLE_DEVICES=1 nohup $PY -u train.py --corpora road,bddx,covla \
  --lam "$LAM" --tag r4 > logs/train_r4.log 2>&1 &
R4=$!
echo "[launch] R3 pid $R3 (gpu0, lam=0) | R4 pid $R4 (gpu1, lam=$LAM) — both head_warmup=1000"
wait $R3 $R4
d3=$(grep -c '\[train\] finished\.' logs/train_r3.log)
d4=$(grep -c '\[train\] finished\.' logs/train_r4.log)
echo "[launch] R3+R4 ENDED: r3 clean=$d3 r4 clean=$d4 — run evals on r3/r4 ep checkpoints"

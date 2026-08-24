#!/bin/bash
# Waits for cache chain ($1), trains lam0 + lam10 heads, evals both in parallel.
E=/data/repos/ROAD_Reason/experiments/exp11_yolo
while kill -0 "$1" 2>/dev/null; do sleep 300; done
if ! grep -q "roi caches done" $E/roi_cache.log; then
  echo "[chain] caches incomplete - aborting" >> $E/roialign.log; exit 1
fi
cd $E
python3 -u train_head.py --lam 0  >> $E/roialign.log 2>&1
python3 -u train_head.py --lam 10 >> $E/roialign.log 2>&1
python3 -u eval_head.py --head $E/head_roialign_lam0.pt  --out $E/results_roialign_lam0.json  >> $E/roialign.log 2>&1 &
python3 -u eval_head.py --head $E/head_roialign_lam10.pt --out $E/results_roialign_lam10.json >> $E/roialign.log 2>&1 &
wait
echo "[chain] roialign hybrid done $(date)" >> $E/roialign.log

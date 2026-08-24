#!/bin/bash
E=/data/repos/ROAD_Reason/experiments/exp11_yolo
cd $E
python3 -u train_head.py --lam 0.1 --bs 16384 >> $E/roialign.log 2>&1
python3 -u train_head.py --lam 1   --bs 16384 >> $E/roialign.log 2>&1
python3 -u eval_head.py --head $E/head_roialign_lam0.1.pt --gate --out $E/results_roialign_lam0.1_gated.json >> $E/roialign.log 2>&1 &
python3 -u eval_head.py --head $E/head_roialign_lam1.pt   --gate --out $E/results_roialign_lam1_gated.json   >> $E/roialign.log 2>&1 &
wait
echo "[chain] LAMBDA SWEEP done $(date)" >> $E/roialign.log

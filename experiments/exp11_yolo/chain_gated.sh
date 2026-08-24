#!/bin/bash
E=/data/repos/ROAD_Reason/experiments/exp11_yolo
cd $E
python3 -u eval_head.py --head $E/head_roialign_lam0.pt --gate --out $E/results_roialign_lam0_gated.json >> $E/roialign.log 2>&1 &
until [ -f $E/head_roialign_lam10.pt ]; do sleep 60; done
python3 -u eval_head.py --head $E/head_roialign_lam10.pt --gate --out $E/results_roialign_lam10_gated.json >> $E/roialign.log 2>&1 &
wait
echo "[chain] GATED results done $(date)" >> $E/roialign.log

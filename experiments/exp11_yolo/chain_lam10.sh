#!/bin/bash
E=/data/repos/ROAD_Reason/experiments/exp11_yolo
cd $E
python3 -u train_head.py --lam 10 --bs 16384 >> $E/roialign.log 2>&1
python3 -u eval_head.py --head $E/head_roialign_lam10.pt --out $E/results_roialign_lam10.json >> $E/roialign.log 2>&1
echo "[chain] lam10 RESULTS done $(date)" >> $E/roialign.log

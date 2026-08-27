#!/bin/bash
# Final exp11 chain: wait for train dump -> junk cache -> retrain -> 3 evals + 26x scoring
E=/data/repos/ROAD_Reason/experiments/exp11_yolo
L=$E/final_chain.log
until [ -f $E/dets_v8x_best_train_fullcand.pkl ]; do
  pgrep -f eval_yolo_agent_train >/dev/null || { echo "[chain] dump died without output" >> $L; exit 1; }
  sleep 300
done
echo "[chain] dump landed $(date)" >> $L
cd $E
CUDA_VISIBLE_DEVICES=0 python3 -u cache_roi_feats.py --split train --junk >> $L 2>&1 || exit 1
echo "[chain] junk cache done $(date)" >> $L
python3 -u train_head.py --lam 0 --bs 16384 --junk-cache $E/roi_feats_i3d_train_junk.pkl --out-tag _junkneg >> $L 2>&1 || exit 1
python3 -u eval_head.py --head $E/head_roialign_lam0_junkneg.pt --gate --out $E/results_roialign_junkneg_gated.json >> $L 2>&1 &
python3 -u eval_head.py --head $E/head_roialign_lam0_junkneg.pt --out $E/results_roialign_junkneg_ungated.json >> $L 2>&1 &
CUDA_VISIBLE_DEVICES=0 python3 -u eval_yolo_agent.py --ckpt $E/runs/yolo26x_agent_1280_eccv/weights/best.pt --out $E/results_yolo26x_best_agent_fmap.json >> $L 2>&1
wait
echo "[chain] ALL FINAL RESULTS done $(date)" >> $L

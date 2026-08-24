#!/bin/bash
E=/data/repos/ROAD_Reason/experiments/exp11_yolo
cd $E
python3 -u eval_yolo_agent.py --ckpt $E/runs/v8x_agent_1280_eccv/weights/best.pt \
  --dump $E/dets_v8x_best_val_fullcand.pkl --out $E/results_v8x_best_val_redump.json \
  >> $E/followups.log 2>&1
python3 -u eval_yolo_agent.py --ckpt $E/runs/v8x_agent_1280_eccv/weights/last.pt \
  --out $E/results_v8x_last_ep50_agent_fmap.json >> $E/followups.log 2>&1
echo "[followups] done $(date)" >> $E/followups.log

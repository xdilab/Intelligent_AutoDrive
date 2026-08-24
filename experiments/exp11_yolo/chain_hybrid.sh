#!/bin/bash
E=/data/repos/ROAD_Reason/experiments/exp11_yolo
while kill -0 "$1" 2>/dev/null; do sleep 120; done
if [ -f $E/dets_i3d_val_fullcand.pkl ]; then
  python3 -u $E/eval_hybrid_score_transfer.py >> $E/hybrid.log 2>&1
  echo "[chain] hybrid eval done $(date)" >> $E/hybrid.log
else
  echo "[chain] I3D dump missing - dump crashed, not running hybrid" >> $E/hybrid.log
fi

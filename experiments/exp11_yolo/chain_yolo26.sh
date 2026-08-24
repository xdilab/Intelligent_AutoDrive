#!/bin/bash
# Auto-launch YOLO26x when the v8x run (PID $1) exits cleanly.
E=/data/repos/ROAD_Reason/experiments/exp11_yolo
while kill -0 "$1" 2>/dev/null; do sleep 300; done
sleep 60   # let DDP workers drain and GPU memory release
if grep -qa "50 epochs completed" $E/train_v8x_agent_1280_eccv.log; then
  echo "[chain] v8x completed cleanly $(date) — launching YOLO26x" >> $E/chain.log
  cd $E && nohup $E/venv26/bin/python3 -u $E/train_v8x.py \
    --model $E/yolo26x.pt --name yolo26x_agent_1280_eccv \
    > $E/train_yolo26x_agent_1280_eccv.log 2>&1 &
  echo "[chain] yolo26x PID $!" >> $E/chain.log
else
  echo "[chain] v8x exited WITHOUT completing — NOT launching 26x. Investigate." >> $E/chain.log
fi

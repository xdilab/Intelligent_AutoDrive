#!/bin/bash
E=/data/repos/ROAD_Reason/experiments/exp11_yolo
cd $E
python3 -u eval_yolo_agent_train.py >> $E/dump_train.log 2>&1

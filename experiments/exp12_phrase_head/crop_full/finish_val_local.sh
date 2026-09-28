#!/bin/bash
# Local insurance: finish val shards 2,3 from pulled partials on the A6000.
set -e
CF=/data/repos/ROAD_Reason/experiments/exp12_phrase_head/crop_full
cd $CF
export ROADCROP_FRAMES=/data/datasets/ROAD_plusplus/rgb-images
export ROADCROP_JSON=/data/datasets/ROAD_plusplus/road_waymo_trainval_v1.1.json
export ROADCROP_DUMPS=/data/repos/ROAD_Reason/experiments/exp11_yolo
export ROADCROP_OUT=$CF
python3 -u cache_crop_feats_h200.py --split val --nshard 4 --ishard 2
python3 -u cache_crop_feats_h200.py --split val --nshard 4 --ishard 3
echo "[local] VAL FINISH DONE $(date)"

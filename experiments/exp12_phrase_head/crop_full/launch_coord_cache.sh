#!/bin/bash
cd /data/repos/ROAD_Reason/experiments/exp12_phrase_head/crop_full
CUDA_VISIBLE_DEVICES=0 setsid nohup python3 -u cache_crop_feats_h200.py --split train --shard --nshard 2 --ishard 0 > coord_cache_s0.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=1 setsid nohup python3 -u cache_crop_feats_h200.py --split train --shard --nshard 2 --ishard 1 > coord_cache_s1.log 2>&1 < /dev/null &
echo launched

#!/bin/bash
E=/data/repos/ROAD_Reason/experiments/exp11_yolo
cd $E
CUDA_VISIBLE_DEVICES=0 python3 -u cache_roi_feats.py --split val >> $E/roi_cache.log 2>&1
CUDA_VISIBLE_DEVICES=0 python3 -u cache_roi_feats.py --split train >> $E/roi_cache.log 2>&1
echo "[chain] roi caches done $(date)" >> $E/roi_cache.log

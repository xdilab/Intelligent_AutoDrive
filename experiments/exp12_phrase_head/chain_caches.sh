#!/bin/bash
E12=/data/repos/ROAD_Reason/experiments/exp12_phrase_head
cd $E12
CUDA_VISIBLE_DEVICES=0 python3 -u cache_clip_feats.py --split val >> $E12/cache.log 2>&1
CUDA_VISIBLE_DEVICES=0 python3 -u cache_clip_feats.py --split train >> $E12/cache.log 2>&1
echo "[chain] exp12 caches done $(date)" >> $E12/cache.log

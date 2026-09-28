#!/bin/bash
# Heads + comp MLPs on the merged full crop train cache. Launch detached:
#   setsid nohup bash crop_full/chain_train_crop_full.sh > crop_full/chain_train.log 2>&1 < /dev/null &
set -e
E12=/data/repos/ROAD_Reason/experiments/exp12_phrase_head
CF=$E12/crop_full
cd $E12
test -f $CF/crop_feats_train.pkl
python3 -u train_clip_head.py --head flat   --train-cache $CF/crop_feats_train.pkl --tag _crop_full
python3 -u train_clip_head.py --head phrase --train-cache $CF/crop_feats_train.pkl --tag _crop_full
python3 -u $CF/train_comp_mlp_crop.py --prims flat
python3 -u $CF/train_comp_mlp_crop.py --prims phrase
echo "[chain] TRAIN STAGE DONE $(date)"

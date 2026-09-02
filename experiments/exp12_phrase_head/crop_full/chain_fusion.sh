#!/bin/bash
# Exp14 fusion: train the 1208-d fusion comp MLP, then eval the fused record row.
#   setsid nohup bash crop_full/chain_fusion.sh > crop_full/chain_fusion.log 2>&1 < /dev/null &
set -e
E12=/data/repos/ROAD_Reason/experiments/exp12_phrase_head
CF=$E12/crop_full
cd $E12
test -f $CF/crop_feats_train.pkl
test -f $CF/crop_feats_val.pkl
python3 -u $CF/train_comp_mlp_fusion.py
python3 -u eval_comb.py \
  --ckpt $E12/clip_head_flat_crop_full.pt \
  --phrase-ckpt $E12/clip_head_phrase_crop_full.pt \
  --comp-mlp $CF/comp_mlp_fusion_crop_full.pt \
  --feat-cache $CF/crop_feats_val.pkl --key-style stem \
  --out $CF/results_crop_full_fusion_mlp.json
echo "[chain] FUSION DONE $(date)"

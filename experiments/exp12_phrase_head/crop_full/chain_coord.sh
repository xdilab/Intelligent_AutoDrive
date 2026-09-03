#!/bin/bash
# Coordinate-append probe: train base+coord heads on the re-cached 2/12 subset, eval both on full val.
#   setsid nohup bash crop_full/chain_coord.sh > crop_full/chain_coord.log 2>&1 < /dev/null &
set -e
E12=/data/repos/ROAD_Reason/experiments/exp12_phrase_head
CF=$E12/crop_full
cd $E12
test -f $CF/crop_feats_train_probe.shard0of2.pkl
test -f $CF/crop_feats_train_probe.shard1of2.pkl
python3 -u $CF/train_head_coord.py
python3 -u eval_comb.py --ckpt $E12/clip_head_flat_probe_base.pt \
  --feat-cache $CF/crop_feats_val.pkl --key-style stem \
  --out $CF/results_crop_probe_coordbase.json
python3 -u eval_comb.py --ckpt $E12/clip_head_flat_probe_coord.pt --coords \
  --feat-cache $CF/crop_feats_val.pkl --key-style stem \
  --out $CF/results_crop_probe_coord.json
echo "[chain] COORD DONE $(date)"

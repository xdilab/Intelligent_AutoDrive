#!/bin/bash
# All evals on the merged full crop val cache. Launch detached after val merge
# (and after chain_train for the last four evals; the first two need only val):
#   setsid nohup bash crop_full/chain_eval_crop_full.sh > crop_full/chain_eval.log 2>&1 < /dev/null &
set -e
E12=/data/repos/ROAD_Reason/experiments/exp12_phrase_head
CF=$E12/crop_full
cd $E12
test -f $CF/crop_feats_val.pkl
# Milestone 1: probe-trained heads on the full val cache (protocol-comparable immediately)
python3 -u eval_comb.py --ckpt $E12/clip_head_flat_crop.pt   --feat-cache $CF/crop_feats_val.pkl --key-style stem --out $CF/results_crop_probeheads_flat.json
python3 -u eval_comb.py --ckpt $E12/clip_head_phrase_crop.pt --feat-cache $CF/crop_feats_val.pkl --key-style stem --out $CF/results_crop_probeheads_phrase.json
# Milestone 2: full-trained heads and both stacks
python3 -u eval_comb.py --ckpt $E12/clip_head_flat_crop_full.pt   --feat-cache $CF/crop_feats_val.pkl --key-style stem --out $CF/results_crop_full_flat.json
python3 -u eval_comb.py --ckpt $E12/clip_head_phrase_crop_full.pt --feat-cache $CF/crop_feats_val.pkl --key-style stem --out $CF/results_crop_full_phrase.json
python3 -u eval_comb.py --ckpt $E12/clip_head_flat_crop_full.pt   --feat-cache $CF/crop_feats_val.pkl --key-style stem --comp-mlp $CF/comp_mlp_flat_crop_full.pt   --out $CF/results_crop_full_flat_mlp.json
python3 -u eval_comb.py --ckpt $E12/clip_head_phrase_crop_full.pt --feat-cache $CF/crop_feats_val.pkl --key-style stem --comp-mlp $CF/comp_mlp_phrase_crop_full.pt --out $CF/results_crop_full_phrase_mlp.json
echo "[chain] EVAL STAGE DONE $(date)"

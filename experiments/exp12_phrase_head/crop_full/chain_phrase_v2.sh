#!/bin/bash
set -e
E12=/data/repos/ROAD_Reason/experiments/exp12_phrase_head
CF=$E12/crop_full
cd $E12
export PHRASES=$E12/phrases_v2.json
export EMBEDS_OUT=$E12/phrase_embeds_v2.pt
export EMBEDS=$E12/phrase_embeds_v2.pt
python3 -u embed_phrases.py
python3 -u train_clip_head.py --head phrase --train-cache $CF/crop_feats_train.pkl --tag _crop_full_v2
python3 -u eval_comb.py --ckpt $E12/clip_head_phrase_crop_full_v2.pt --feat-cache $CF/crop_feats_val.pkl --key-style stem --out $CF/results_crop_full_phrase_v2.json
echo "[chain] PHRASE V2 DONE $(date)"

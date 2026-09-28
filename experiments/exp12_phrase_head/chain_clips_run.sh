#!/bin/bash
E12=/data/repos/ROAD_Reason/experiments/exp12_phrase_head
L=$E12/clips_run.log
cd $E12
CUDA_VISIBLE_DEVICES=0 python3 -u cache_clip_feats.py --split val --clips >> $L 2>&1 || exit 1
CUDA_VISIBLE_DEVICES=0 python3 -u cache_clip_feats.py --split train --clips >> $L 2>&1 || exit 1
python3 -u train_clip_head.py --head flat   --train-cache $E12/clip_feats_train_clips.pkl --tag _clips >> $L 2>&1 || exit 1
python3 -u train_clip_head.py --head phrase --train-cache $E12/clip_feats_train_clips.pkl --tag _clips >> $L 2>&1 || exit 1
setsid python3 -u eval_clip_head.py --ckpt $E12/clip_head_flat_clips.pt   --feat-cache $E12/clip_feats_val_clips.pkl --out $E12/results_c5_flat_clips_gated.json   >> $L 2>&1 &
setsid python3 -u eval_clip_head.py --ckpt $E12/clip_head_phrase_clips.pt --feat-cache $E12/clip_feats_val_clips.pkl --out $E12/results_c6_phrase_clips_gated.json >> $L 2>&1 &
wait
echo "[chain] TRUE-CLIP RUN done $(date)" >> $L

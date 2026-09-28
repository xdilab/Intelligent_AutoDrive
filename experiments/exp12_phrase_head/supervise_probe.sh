#!/bin/bash
# Crop-RoI probe: waits for both probe caches, trains crop heads, runs
# 4 evals (crop flat/phrase natively on the subset; featuremap C5/C6
# comparators restricted to the identical frames).
E12=/data/repos/ROAD_Reason/experiments/exp12_phrase_head
L=$E12/probe_run.log
cd $E12
until [ -f $E12/crop_feats_val_probe.pkl ] && [ -f $E12/crop_feats_train_probe.pkl ]; do sleep 600; done
python3 -u train_clip_head.py --head flat   --train-cache $E12/crop_feats_train_probe.pkl --tag _crop >> $L 2>&1 || exit 1
python3 -u train_clip_head.py --head phrase --train-cache $E12/crop_feats_train_probe.pkl --tag _crop >> $L 2>&1 || exit 1
setsid python3 -u eval_clip_head.py --ckpt $E12/clip_head_flat_crop.pt   --feat-cache $E12/crop_feats_val_probe.pkl --out $E12/results_probe_crop_flat.json   >> $L 2>&1 &
setsid python3 -u eval_clip_head.py --ckpt $E12/clip_head_phrase_crop.pt --feat-cache $E12/crop_feats_val_probe.pkl --out $E12/results_probe_crop_phrase.json >> $L 2>&1 &
setsid python3 -u eval_clip_head.py --ckpt $E12/clip_head_flat_clips.pt   --feat-cache $E12/clip_feats_val_clips.pkl --restrict $E12/crop_feats_val_probe.pkl --out $E12/results_probe_featmap_flat.json   >> $L 2>&1 &
setsid python3 -u eval_clip_head.py --ckpt $E12/clip_head_phrase_clips.pt --feat-cache $E12/clip_feats_val_clips.pkl --restrict $E12/crop_feats_val_probe.pkl --out $E12/results_probe_featmap_phrase.json >> $L 2>&1 &
wait
echo "[chain] CROP PROBE done $(date)" >> $L

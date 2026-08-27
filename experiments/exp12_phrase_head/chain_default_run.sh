#!/bin/bash
E12=/data/repos/ROAD_Reason/experiments/exp12_phrase_head
L=$E12/default_run.log
until grep -q "exp12 caches done" $E12/cache.log 2>/dev/null; do sleep 600; done
cd $E12
python3 -u train_clip_head.py --head flat   >> $L 2>&1 || exit 1
python3 -u train_clip_head.py --head phrase >> $L 2>&1 || exit 1
python3 -u eval_clip_head.py --ckpt $E12/clip_head_flat.pt   --out $E12/results_c1_flat_gated.json   >> $L 2>&1 &
python3 -u eval_clip_head.py --ckpt $E12/clip_head_phrase.pt --out $E12/results_c2_phrase_gated.json >> $L 2>&1 &
wait
echo "[chain] EXP12 DEFAULT RUN done $(date)" >> $L

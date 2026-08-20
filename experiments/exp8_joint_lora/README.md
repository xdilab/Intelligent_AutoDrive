# Exp8 — Joint LoRA SFT: ROAD-Waymo + BDD-X + CoVLA (Approach 8, Stage 2, joint)

**Question:** does a VLM jointly tuned on the three corpora — interleaved so
every optimizer step updates on one sample from each — beat both the zero-shot
baseline fusion and the detector-only control when plugged into the exp5→exp6
pipeline?

## Provenance

- Dr. Moradi (relayed 2026-07-27): Stage-2 training is **joint**, not
  sequential — "one instance of the dataset from each set would update the
  weights", alternating ROAD-Waymo / BDD-X / CoVLA.
- Consistent with the long-approved corpus order `ROAD-R → BDD-X → CoVLA →
  joint` (direction page, "Re-validated") — this goes straight to the joint
  phase.
- Supersedes exp7 (BDD-X-only leg). exp7's adapter, merged checkpoint, and
  partial re-cache (~200 frames in exp5 `cache/qwen_bddx_lora/`) stay on disk
  as a resumable ablation.

## The three corpora (all local, nothing downloaded)

| Block | Train | SFT-val | Sample | Target (verbatim GT/annotator text) |
|---|---|---|---|---|
| bddx | 4,339 | 545 | exp7's temporally-aligned jsons, reused byte-identical | `{action, justification}` |
| covla | ~2,250 | ~250 | CoVLA-mini frame (stride 12 = 0.6 s; last 5 of 50 scenes held out) | `{caption: plain_caption, risk}` |
| road | ~9,596 | 500 | native frame + frozen RetinaNet top-40 boxes drawn (exp5 deployment rendering, same `vlm_io.draw_numbered_boxes`) | per-box `{agent, actions, locations}` via IoU≥0.5 greedy-argmax GT assignment (exp6 rule); unmatched → `agent: null` |

ROAD frame subsets by sorted-key index mod 12: train = shards {0,1} (the same
9,596-frame subset the Stage-1 fusion pipeline caches), SFT-val = shard 11 —
disjoint from train, and the ROAD-Waymo **val split stays untouched** (it is
the downstream fixed test set; lab rule).

## Interleaving (the joint strategy)

`RoundRobinSampler` over the block-concatenated dataset + `GRAD_ACCUM=3`:
every optimizer step's accumulation window is exactly (bddx, covla, road).
Epoch = one pass over the largest block (road); smaller blocks cycle. Within
each block the order reshuffles per epoch.

## Deviations from the exp7/reference recipe (reasons)

| Deviation | exp7 | exp8 | Reason |
|---|---|---|---|
| grad accum | 4 | 3 | one sample per corpus per optimizer step (the joint strategy) |
| sampler | HF random | round-robin | same |
| data_flatten / attn | flash-attn varlen (fallback sdpa used in practice) | off / sdpa | flatten packs samples across corpora into one sequence, breaking one-per-corpus step semantics; sdpa+padded is exp7's proven path |
| max_pixels | 50,176 (64 tok) | 501,760 (640 tok) | per-box supervision needs legible box ids; deployment feeds ~600×840 ≈ 504k px — training now matches inference resolution (exp7 README flagged the old mismatch) |
| ROAD prompt | — | deployment prompt minus risk/rationale, plus a null-agent rule | risk/rationale have no ROAD GT (no synthetic labels); a *distinct* prompt prevents the model from learning to drop those fields under the deployment prompt |

Everything else (LoRA r64/α128/dropout 0.05 on LLM attn only, frozen vision
tower, lr 1e-5 cosine, bf16, grad ckpt, 3 epochs, per-epoch eval/save,
checkpoint selection by joint SFT-val loss) is inherited from exp7/reference.

## Pipeline

```bash
cd /data/repos/ROAD_Reason/experiments/exp8_joint_lora
PY=/home/brandon/miniconda3/bin/python

$PY -u data_prep.py                                   # 1. build corpora (CPU)
CUDA_VISIBLE_DEVICES=0 $PY -u train_joint.py --max-steps 2   # 2. smoke
CUDA_VISIBLE_DEVICES=1 $PY -u train_joint.py 2>&1 | tee -a logs/train.log  # 3. train
$PY -u export_merged.py                               # 4. merge best-val ckpt
# 5. re-cache + refuse: exp5 config QWEN_MODEL -> merged dir, fresh cache dirs
#    (cache/qwen_joint_lora/), rerun qwen_infer train shards 0,1 + val, then
#    exp6 embed_rationale / train / eval — same protocol as the exp7 plan.
```

`export_merged.py` = exp7's `export_for_exp5.py` pointed at this experiment
(picks lowest joint-val-loss epoch; never selects on ROAD-Waymo val).

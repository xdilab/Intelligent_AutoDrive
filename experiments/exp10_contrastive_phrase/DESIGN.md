# Exp10 — Contrastive Phrase Head on Frozen Text-Aligned Features (DESIGN)

**Status:** design + Tier-0 feature caching started 2026-08-19. Head code not
yet scaffolded. Provenance: Dr. Moradi 2026-08-18 ("contrastive matching
between RoI features and text embeddings of the composed triplet phrases lets
rare triplets borrow statistical strength from semantically related common
ones"); motivated by the exp9 attribution grid (three-way language negative —
the one untested mechanism is text as label-space geometry).

## 1. Question

Does classifying detector boxes by **proximity to phrase embeddings in a
pre-aligned video–text space** beat the flat-184 independent-sigmoid head on
the relational/long-tail heads (duplex/triplet), on the same boxes, features,
and protocol?

## 2. Cells (attribution discipline)

| cell | encoder | head | isolates |
|---|---|---|---|
| R1 (exists, exp9) | Qwen ViT (drawn frames) | flat-184 | reference |
| C1 | InternVideo2 (clean frames) | flat-184 | encoder swap |
| C2 | InternVideo2 (clean frames) | phrase head | **the experiment** (vs C1) |

Caveat: R1→C1 confounds encoder with drawn→clean input. If C1 diverges
sharply from R1, run R1′ (Qwen + clean) to deconfound — cheap (~3.4 h).

## 3. Head design — RECOMMENDED (A): phrases as classifier weights

Compute cosine similarity between the projected RoI feature and each of the
**183 class phrases** (10 agent + 22 action + 16 loc + 49 duplex + 86
triplet), scale by learnable temperature + bias → treat as **logits** in the
same 184-dim layout (agentness stays a scalar head). Then reuse the exp9 loss
machinery **verbatim**: focal-on-all with flat alphas + corrected Gödel t-norm.

Why (A): (i) multi-label native — a box can carry several actions, which
single-positive InfoNCE cannot express; (ii) background = all-zeros target
rows, exactly the exp2f negative-supervision recipe — **this dissolves the
"background embedding vs threshold" open question** from the 2026-08-19 email
(flag to Moradi for confirmation); (iii) every number stays comparable to the
exp9 grid because only the classifier's weight *source* changes (learned free
slots → frozen phrase embeddings + learned projection).

Alternatives if Moradi prefers: (B) per-head softmax InfoNCE with a learned
background embedding; (C) SigLIP-style pairwise sigmoid. Both are small
variations on the same cached features.

Trainable surface (Tier 0): projection MLP (1024 → 512), temperature, bias,
scalar agentness head. Everything else frozen.

## 4. Phrases

`phrases.json` (checked in, human-reviewed): natural-language surface for the
dataset vocabulary — e.g. Ped→"a pedestrian", MovTow→"moving toward the ego
vehicle", VehLane→"in the ego vehicle's lane"; compositions by template
("{agent} {action} {location}"). Vocabulary is the dataset's own (Moradi
2026-06-11: "if it come from dataset I'm fine"); only the phrasing is ours —
send the file for his review with the design.

## 5. Encoders — pipeline now, flagship after

| | InternVideo2_CLIP_S (now) | InternVideo2-CLIP-1B (target) |
|---|---|---|
| availability | self-contained HF repo, loads today (custom-code meta-init workaround: instantiate class directly, load safetensors manually) | raw `1B_clip.pth` — needs the cloned OpenGVLab/InternVideo repo (integration task, in progress) |
| role | validate cache → head → eval end-to-end; a legitimate small-encoder cell | the experiment's encoder (paper's clip variant) |
| token map | 8×16×16×1024 post-blocks; shared space 512-d | larger dims, same extraction pattern |

Both consume: clean native frame → resize 224×224 → replicate ×8 (static
clip). **Known risks, named:** (i) resolution — 224² → 16×16 token grid; a
distant pedestrian box can span <1 token (the exp2 resolution lesson; the
RoI pool's min-1-token rule applies; measured by C1-vs-R1 agent delta before
any conclusion is drawn about the head); (ii) static-replication wastes the
encoder's temporal modeling — the natural follow-up cell C4 feeds the true
8-frame clip around the keyframe, the lineage's first genuine temporal-context
cell (targets the action-head deficit directly).

## 6. Caching (running)

`cache_features.py --split {train,val}`: exp6 detections pkl → clean frame →
CLIP_S tower → post-blocks token map, mean over T → RoI-pool per top-40 box →
`[n,1024]` fp16 per frame → one `.pt` per split (~0.8 GB each; train subset
9,596 + val 9,504 frames). Phrase embeddings cached at head-training time
(seconds). Head training on cached vectors = minutes (exp6 pattern); eval
stays forward-only.

## 7. Eval & gates

Same top-40 protocol, f-mAP + corrected-set violations. Gates:
**C2 > C1 on duplex/triplet** (the head effect — Moradi's hypothesis);
C1 vs R1 reported as the encoder effect; everything vs the detector-only
control (agent 14.62 / duplex 10.33 / triplet 7.48).

## 8. Open items for Moradi

1. Confirm formulation (A) — it makes his "background box" question moot.
2. Review `phrases.json` wording.
3. 1B now vs CLIP_S results first (we proceed CLIP_S-first regardless; 1B
   integration continues in parallel).

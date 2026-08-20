# Exp9 — Joint Round-Robin Training with Per-Corpus Losses (DESIGN, pre-code)

**Status:** scaffolded + smoke-tested 2026-08-17 (all checks green: corrected
t-norm sanity, data paths, LoRA placement ViT 15.7M / LLM 40.4M / head 0.66M,
gradient isolation ROAD→{head, ViT-LoRA} only, 2 full alternation cycles,
checkpoint save/load, eval pipeline on val frames). Runs R1–R4 awaiting
Moradi's answers on the §9 flags (each is a config/CLI switch, no code change).
NOTE (deviation from §3 as written): the merged visual tokens are 3584-dim in
transformers 5.x (the merger projects into LLM width), so the head is
Linear(3584→184); and LoRA deliberately covers BOTH ViT and LLM attention so
the ViT LoRA is the shared trainable surface between legs — without it the
R1→R3 comparison would be vacuous (LLM-only LoRA is untouched by the ROAD leg).
**Provenance:** Moradi/Brandon discussion (joint training, per-dataset loss, round-robin);
Moradi 2026-06-02 direction (loss-side t-norm as separately-attributable experiment);
supersedes exp8's homogeneous-loss interleaving for the ROAD leg.

---

## 1. Question

Does joint training with **heterogeneous per-corpus losses** — where the ROAD leg
trains structured sigmoid heads under a t-norm constraint penalty instead of
emitting JSON text — produce a model whose ROAD-Waymo compositional predictions
(duplex/triplet) beat both (a) the same architecture trained on ROAD only, and
(b) the same joint run with the t-norm off?

This isolates two attributable effects Moradi asked to be separable:
- **language co-training effect** — ROAD-only vs joint, at fixed λ
- **constraint effect** — λ=0 vs λ>0, at fixed data mix

## 2. What exp8 did vs what exp9 does

| | exp8 (done) | exp9 (this) |
|---|---|---|
| Data order | round-robin bddx→covla→road | same round-robin |
| ROAD target | structured labels serialized to JSON **text** | **184-dim sigmoid heads** (no text) |
| Loss | single next-token CE for all corpora | **per-corpus**: ROAD = focal + λ·t-norm; BDD-X/CoVLA = next-token CE |
| Update rule | 1 summed update per corpus triple (grad-accum 3) | **strict alternation**: one optimizer step per corpus sample |
| Constraints | prompt-side only (baked into SFT text) | **loss-side Gödel t-norm** on head probabilities |

## 3. Model

- **Backbone:** Qwen2.5-VL-7B + LoRA (exp7/exp8 LoRA config as starting point), shared across all legs.
- **BDD-X / CoVLA legs:** standard LM head, next-token CE (reference trainer path, unchanged from exp8).
- **ROAD leg:** no text generation. Frame + detector boxes → ViT feature map →
  `ROIAveragePool` per box → classification head → flat **184-dim sigmoid**
  `[agentness|agent 10|action 22|loc 16|duplex 49|triplet 86]`.
  Reuses exp1's `ROIAveragePool`/`ClassificationHeads` mechanism and exp2f's
  flat-head recipe (single Linear, focal-on-all with flat alphas).

## 4. ROAD leg data (decided: detector boxes)

- Boxes from the frozen 3D-RetinaNet (`model_000025.pth`) — pseudo-labels per
  Moradi 2026-06-11 ("not from ground truth … it is psedu-label").
- Source: `exp6_detection_steered/cache/detections_train.pkl` (boxes + logits + GT,
  train split, train-trained detector per the approved fallback).
- GT assignment at IoU ≥ 0.5 (exp6 `dataset.py` machinery). **Unmatched detector
  boxes get all-zeros targets** — explicit negative supervision (exp2f lesson;
  the dominant bug of the exp2 series was its absence).
- Val: fixed ROAD-Waymo val split, never used for model selection.

## 5. Losses

- **ROAD:** `L_road = FocalOnAll(184-dim, flat alphas from exp4/losses.py) + λ · L_tnorm`
  - t-norm: **Gödel** (best in ROAD-R Table 7; matched in exp1b). Penalty over
    the head's sigmoid probabilities using the valid-composition sets.
  - λ sweep: {0, 0.1, 1.0, 10.0}. λ=0 is a mandatory control, not an afterthought.
  - Known regime risk: t-norm is inert when predictions are conservative
    (exp1: violations 0.02%, L_tnorm ≈ 1000× smaller than L_cls). The live
    target is triplets (exp2f: 80.09% violation rate). Report **violation rate
    per head** alongside f-mAP in every run so bind-vs-inert is visible.
- **BDD-X / CoVLA:** next-token CE, unchanged.

## 6. Training loop (decided: strict alternation)

```
for cycle in range(N):
    x_b = next(bddx);  loss = CE(x_b);            loss.backward(); step(); zero_grad()
    x_c = next(covla); loss = CE(x_c);            loss.backward(); step(); zero_grad()   # if CoVLA in
    x_r = next(road);  loss = focal + λ·tnorm;    loss.backward(); step(); zero_grad()
```

- Three optimizer steps per cycle, each from one corpus's own loss. No gradient
  mixing across corpora (unlike exp8's accumulated triple).
- Per-corpus reshuffle each epoch; smaller corpora wrap (reuse exp8's
  `RoundRobinSampler` block logic, sampler unchanged — only the step semantics differ).
- Stability (Moradi: "balance the auxiliary loss or joint training gets
  unstable"): log per-corpus loss curves separately, line-buffered, intra-epoch
  progress prints. Watch for the ROAD head collapsing to majority class
  (encoder-collapse diagnostics: >95% one-class predictions, per-class F1=0 on
  >90% of classes) — the Mov/Ovtak imbalance (1.84M vs 88) makes this the
  named failure mode.
- Checkpoint every epoch; keep by **joint val loss per leg** (exp8 lesson:
  epoch 1 was best; epochs 2–3 overfit).

## 7. Runs (the attribution grid)

| run | data | λ | answers |
|---|---|---|---|
| R1 | ROAD only | 0 | head-on-Qwen baseline with detector boxes (replaces exp1's oracle-box numbers) |
| R2 | ROAD only | best λ | pure loss-side t-norm effect — **Moradi's requested experiment** |
| R3 | joint round-robin | 0 | pure language-co-training effect |
| R4 | joint round-robin | best λ | the combined hypothesis |

λ selection happens inside R2 via the sweep; R4 reuses it. R1→R2 and R3→R4
isolate the constraint; R1→R3 and R2→R4 isolate co-training.

## 8. Evaluation

- **ROAD structured head:** baseline evaluator, f-mAP @ IoU=0.5 per head, over
  detector top-K rows — directly comparable to the exp5/exp6/exp8 control
  family (detector-only 14.62 agent / 12.16 action / 11.76 loc / 10.33 duplex /
  7.48 triplet). Plus per-head constraint-violation rate.
- **Language legs:** BDD-X captioning metrics vs exp7/exp8 numbers (sanity that
  alternation didn't destroy the language capability).
- **Gates:** primary — R2 or R4 > detector-only control on **duplex and
  triplet** (the compositional heads where constraints have live signal).
  Secondary — R4 > R2 (co-training adds something beyond constraints).

## 9. Open items for Dr. Moradi

1. **CoVLA in round one?** ("if we use that") — 2-corpus alternation is a
   cleaner first read; CoVLA adds the risk vocabulary but also a third loss to
   balance.
2. **Constraint sourcing — RESOLVED (2026-08-17), with a defect finding.** The
   JSON's `duplex_childs` (39) / `triplet_childs` (68) arrays are mis-indexed
   against the JSON's own label lists: only 20/39 and 6/68 decode to valid
   compositions. Both existing `tnorm_loss.py` implementations (ROAD_Reason
   root and the baseline's `modules/tnorm_loss.py`) build the penalty set as
   the complement of those arrays, so **all prior loss-side t-norm runs
   penalized 29/49 valid duplexes and 80/86 valid triplets** — exp1's
   "constraints inert" result and the baseline Gödel comparison are both
   tainted; exp2f's 80.09% triplet violation rate needs rechecking against
   the corrected set before being quoted. Exp9 uses
   `constraints_verified.json` (49 duplex + 86 triplet index tuples derived
   from the `duplex_labels`/`triplet_labels` strings — unambiguous). The
   ROAD-R 243-rule file remains a possible *extension*, not the base set.
3. **Per-leg learning rates:** single LR for all steps, or lower LR on the LM
   legs to protect language capability while the fresh ROAD head trains hot?
   (Fresh head + pretrained backbone at one LR is a known tension.)
4. **ROAD leg image input:** frame with boxes *drawn* (exp5-style, matches the
   cached-prompt lineage) vs clean frame + box coordinates to RoI pooling only
   (exp1-style, cleaner separation). Design assumes exp1-style RoI.

## 10. Reuse map

| Piece | From |
|---|---|
| RoundRobinSampler (block logic) | exp8 `train_joint.py` |
| ROIAveragePool + ClassificationHeads | exp1 `model.py` |
| Flat 184-dim focal + alphas | exp2f recipe via exp4 `losses.py` |
| Gödel t-norm | exp1 `tnorm_loss.py` lineage |
| Detector boxes + GT assignment | exp6 `cache/detections_train.pkl`, `dataset.py` |
| BDD-X/CoVLA SFT data + LM path | exp8 cache + reference trainer |
| Eval harness | exp6 `eval.py` protocol family |

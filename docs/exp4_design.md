# Exp4 Design Spec — Reasoning + Boxes for ROAD-Waymo

**Author:** Brandon Byrd · **Advisor:** Dr. Moradi · **Date:** 2026-06-02

The final output of this work is **(a) per-agent bounding boxes** on ROAD-Waymo
clips and **(b) natural-language reasoning** about agent intent / scene
constraints. The architectural choice driving the rest of the thesis is **where
the boxes come from**. Two viable designs:

---

## Option A — Modular: RetinaNet detector + Qwen reasoner

```
Clip (8 frames, 600×840) ──► 3D-RetinaNet (FROZEN, epoch 25)
                                 │
                                 ├──► boxes [K, T, 4]
                                 ├──► agent / action / loc / duplex / triplet logits
                                 │
                                 └──► RoI-pooled tube features [K, T, 256]
                                                      │
                                                      ▼
                            MoonViT (FROZEN) ──► tube-level semantic features
                                                      │
                                                      ▼
                                            Q-Former (TRAIN, ~2.5M params)
                                                      │
                                                      ▼
                                            Qwen2.5-3B (LoRA-tuned)
                                            "Pedestrian is crossing because…"
```

- Detection is decoupled from language. The 17.76% agent f-mAP baseline is
  guaranteed (it is the floor).
- Qwen sees per-tube visual tokens and emits **reasoning text only**, not boxes.
- Constraint reasoning (ROAD-R duplex/triplet logical relations) is enforced on
  the detection head with T-norm losses, exactly as in the baseline.

**Strengths**
- Detection floor is locked at 17.76% — we cannot regress on the proven number.
- Trainable surface is tiny (~2.5M Q-Former + LoRA adapters), feasible on one A6000.
- Each component is replaceable: swap MoonViT in/out, swap Qwen3 → Qwen2.5 → etc.
- Honest evaluation: we report mAP on the standard ROAD-Waymo metric directly.

**Weaknesses**
- Two losses, two evaluation regimes (detection metric + language metric).
- Reasoning is "downstream" of detection — wrong boxes → wrong reasoning.
- No path to open-vocabulary detection later (RetinaNet is fixed to ROAD-R's
  10/22/16/49/86 class structure).

---

## Option B — Unified: Qwen as decoder (LocateAnything style)

```
Clip (8 frames, native resolution) ──► MoonViT (FROZEN)
                                            │
                                            ▼
                                    Visual tokens (~1–4K)
                                            │
                                            ▼
                                  Qwen2.5-3B (LoRA-tuned)
                                            │
              ┌─────────────────────────────┼─────────────────────────────┐
              ▼                             ▼                             ▼
   <box> x1, y1, x2, y2 </box>  agent label + duplex + triplet     reasoning text
        (parallel decoded,        (language tokens)              (free-form CoT)
         per Wang et al. 2026)
```

Wang et al., *LocateAnything* (arXiv 2605.27365, NVIDIA 2026): MoonViT + Qwen2.5-3B
with **parallel box decoding** (PBD) achieves **+3.8% F1 on LVIS** over the
prior generative-grounding SOTA at **2.5× inference throughput**, generalizes
zero-shot across driving + GUI + document domains.

**Strengths**
- One model, one loss (next-token), one decoder. Output is genuinely joint.
- Open-vocabulary by construction — can label novel agents (deer, debris) at test
  time with no retraining.
- Reasoning and grounding share the same hidden state — pedagogically and
  scientifically the cleaner statement of the thesis.
- PBD removes the inference-speed objection to generative detection.

**Weaknesses**
- Our exp1b empirically *failed* this paradigm on ROAD-Waymo: Qwen features +
  FCOS detector hit 3.20% f-mAP vs 17.76% baseline. PBD might fix this, but it
  is unverified on ROAD-Waymo.
- Training a Qwen detector requires re-tokenizing the ROAD-R 184-dim label
  vocabulary as language. Non-trivial.
- Risk that the multi-label structure (49 duplexes × 86 triplets co-occurring
  per agent) does not serialize cleanly into autoregressive tokens.
- Disk: Qwen2.5-3B + MoonViT weights ≈ 7 GB; current `/data` has 18 GB free.

---

## Side-by-side

| Dimension | Option A (modular) | Option B (unified) |
|---|---|---|
| Detection floor | **17.76% f-mAP guaranteed** | unverified on ROAD-Waymo |
| Reasoning | bolt-on, downstream of detection | native, joint with boxes |
| Trainable params | ~2.5M + LoRA | LoRA only on Qwen (~50M typical) |
| Loss design | detection (focal) + language (CE) + T-norm | next-token CE only |
| Eval metric | ROAD-Waymo f-mAP directly | needs box-extraction pipeline before f-mAP |
| Open-vocab | no | yes |
| Reference impl | none (novel composition) | LocateAnything (arXiv 2605.27365) |
| Compute | feasible on 1× A6000 | needs careful sequence packing |
| Empirical risk | low (RetinaNet is locked) | high (exp1b suggests Qwen-grounding is hard on ROAD-Waymo) |
| Thesis story | "augmenting a strong detector with reasoning" | "unifying detection and reasoning in one decoder" |

---

## Recommendation

**Option A as the primary path, with Option B as an explicit ablation.** Three
reasons:

1. **Replication-before-innovation.** We already burned the exp2 series chasing
   an architectural restructuring that did not match the baseline. Locking the
   17.76% RetinaNet floor and *adding* MoonViT-based reasoning is the
   incremental, defensible move.
2. **Empirical evidence we have:** exp1b (Qwen features + FCOS) collapsed to
   3.20% on ROAD-Waymo. LocateAnything's +3.8% on LVIS is encouraging but does
   not transfer guarantees to ROAD-Waymo's small-object, multi-label,
   constrained label space.
3. **Reasoning is the contribution.** The novelty is the *combination* of
   detection + neuro-symbolic reasoning + language. Option A makes the reasoning
   explicit and ablatable. Option B conflates them.

Run Option B as the final ablation: take Option A's working pipeline, swap
RetinaNet→Qwen-PBD on the same dataset, report the delta. That contrast is
exactly what a thesis reader wants to see.

---

## Open questions for Dr. Moradi

1. Is "open-vocabulary detection of novel agents at test time" a hard
   requirement for the thesis, or a nice-to-have? If hard, Option B becomes
   primary.
2. Reasoning output: free-form CoT, or structured-template (e.g. `<reason
   action="crossing" because="...">`)? Affects whether we need a language
   metric beyond perplexity.
3. Should we cite LocateAnything as motivation for Option A's MoonViT choice,
   or reserve it for the Option B ablation framing?
4. Timeline: I have a working Option A skeleton (frozen RetinaNet + frozen
   MoonViT + Q-Former + flat head) that passes a forward-pass smoke test today.
   First training-ready milestone is the dataloader + train loop. Acceptable
   to spend 1–2 weeks on Option A before scoping Option B?

---

## References

- 3D-RetinaNet baseline (this repo): 17.76% agent f-mAP, ROAD-Waymo val (epoch 25 ckpt).
- Wang et al., *LocateAnything: Fast and High-Quality Vision-Language Grounding
  with Parallel Box Decoding*, arXiv 2605.27365 (NVIDIA, 2026).
- Kimi Team, *Kimi-VL / MoonViT*, arXiv 2504.07491 (2025); MoonViT-3D in
  arXiv 2602.02276 (2026). Native-resolution vision encoder.
- Fu et al., *Frozen-DETR*, NeurIPS 2024. Encoder-level CLIP fusion (the
  approach the exp2 series tried, identified as architecturally limited).
- Exp1b — Qwen2.5-VL + FCOS: 3.20% f-mAP on ROAD-Waymo val (replication-style).
- Exp2 series (this repo): exp2f best at 5.51% agent f-mAP — diagnosed as
  CLIP-encoder-fusion spatial mismatch + frozen-encoder adaptation gap.

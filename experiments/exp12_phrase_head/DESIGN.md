# Exp12 — Contrastive Phrase Head on YOLO Rows (DESIGN)

**The proposed thesis contribution** (Moradi's contrastive idea, on the exp11
platform). Timebox: hard go/no-go mid-September (defense end of October).

## Question
Do per-composition text embeddings in a video-text pre-aligned space beat the
stacked composition MLP (duplex 15.52 / triplet 9.73) on identical
full-candidate YOLO rows — with the long tail (Amber, Ovtak, EmVeh, rare
triplets) as the signature evidence?

## Cells (attribution discipline, all on exp11's val rows)
| cell | features | classifier | isolates |
|---|---|---|---|
| exp11 record | I3D P3 (256-d) | linear head + comp MLP | reference (sweep) |
| C1 | InternVideo2_CLIP_S (1024-d) | same head stack retrained | encoder swap |
| C2 | CLIP_S + phrase head | cosine(RoI proj, phrase emb) ×τ+bias → 184 layout | **the experiment** |

C2 formulation (A) from exp10 DESIGN: phrase embeddings as classifier
weights; focal-on-all verbatim; background = all-zero rows; conf gating as in
exp11. Every composition gets its OWN phrase (exp11's ladder: compositions
need their own parameters — here the parameters are semantic).

## Protocol notes
- Training rows: GT + 8 random negatives + ≤16 YOLO junk negatives per frame —
  protocol-equal to exp11's head training (fresh RNG; row-identity not
  required since the comparison is at eval on identical val rows).
- Phrases: `phrases.json`, generated from dataset label names via readable
  templates — REQUIRES Brandon + Moradi review before C2 results are quoted
  (vocabulary is the dataset's own; phrasing is ours).
- Encoder input: clean native frame → 224² → ×8 static clip (exp10 loader,
  meta-init workaround). Known risks carried over: 16×16 token grid vs
  distant pedestrians (C1-vs-record isolates it before C2 is judged).
- Text tower: CLIP_S encode_text, verified present in the checkpoint.

## Gate
C2 > stacked-MLP sweep on duplex/triplet, identical rows; per-class tail
gains reported regardless. C1 reported as the encoder effect either way.

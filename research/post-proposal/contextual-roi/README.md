# Contextual residual RoI fusion

Approved design direction, September17,2026. New architecture, not a renamed Stage5/6 result. Initial implementation only; no full study launched.

## Inputs and cache contract

Use the same frozen InternVideo2-CLIP-S revision as the matched study. Actor crop features remain1024-D. New full-frame clip inference yields a keyframe16x16 spatial map; extract actor context with un-padded7x7 RoIAlign+mean, and retain4x4 scene tokens shared by all actors. Clip time window is the existing eight-frame manifest. This is one temporal clip forward, not independent single-frame inference. Full-frame preprocessing resizes the whole rectangular image to224x224 (aspect distortion explicitly recorded), avoiding the center-crop/box-coordinate mismatch. Small actors may be poorly resolved by this branch; preserve high-detail crop features as the residual path.

Cache uses exact manifest boxes, frame hashes and pinned model revision. It does not resample boxes or silently mix GT and detector rows. Row ordering and box arrays must match before joining an old crop cache. Existing cache mode/dim alone is insufficient evidence of alignment. No encoder gradients; trainable normalized box embedding remains downstream.

## Architecture

Residual video adapter → crop projection. Context-region and scene projections + eight box geometry values. Visual fusion compares (1) concatenation MLP and (2) actor-query attention over16 scene tokens. The visual RoI is aligned to the fixed original triplet phrase prototypes using multi-positive contrastive loss. All applicable triplets are positives; background rows are excluded. No ground-truth label enters forward().

Residual text adapter over the complete184-phrase bank → language attention conditioned on visual RoI → residual joint RoI feature. Final MLP receives joint RoI, adapted visual skip and184 phrase compatibility scores, producing184 logits. Original text features are preserved by the residual text adapter; its complete bank is aggregated through attention/compatibility, not concatenated as a huge constant per sample.

Important refinement relative to the hand sketch: contrastive supervision acts on the **visual view** of the RoI representation, before phrase-conditioned fusion. The joint view goes to classification. This prevents the contrastive prediction from directly copying its target text input. This two-view distinction must be shown explicitly in a future diagram.

## Planned comparison

- MLP fusion, classification only.
- Same MLP fusion + contrastive auxiliary.
- Actor-query attention, classification only.
- Same attention + contrastive auxiliary.

Same cache/rows/splits/seeds, initializations and training budget. Retain420 expert/90 development/90 reserved split; verify manifests before launch. Select on development only. Classification is the current class-weighted focal gamma2 over184 labels; initial auxiliary weight0.001 and temperature0.07 inherit the existing study, not claimed optimal. Report all six detector groups and triplet common/tail/deep-tail; also all development heads rather than only triplet.

AE/VAE/WAE, joint encoder token changes and encoder unfreezing are deferred. Current NCShare jobs are untouched. No claims of novelty or improved AP yet.

## Usage

Head/geometry tests pass in the road_reason environment. Full encoder extraction requires the existing base environment (`/home/brandon/miniconda3/bin/python`, torch2.9.0+cu128, timm1.0.25, transformers5.3.0); road_reason lacks timm. No packages were changed. Tests:

    python -m unittest discover -p 'test_*.py'

Cache smoke or full extraction (omit limit only after validating full manifest/resources):

    python cache_context.py --manifest /path/to/train.jsonl --frames /data/datasets/ROAD_plusplus/rgb-images --output /path/to/context-cache --limit 1

`model.py` implements both heads and the combined objective. `cache_context.py` provides resumable atomic per-frame caches. Full cache joining, training orchestration and detector evaluation integration remain to be completed before a study launch. Smoke inputs may use an archived pilot manifest solely to verify extraction, not to define training scope.

References and original sketches: /data/repos/wiki/artifacts/advisor-architecture-sketch-2026-09-17/.

Validation: one real eight-frame clip (`train_00136_00001`, eight actors) extracted successfully on local GPU0; fingerprinted cache resume succeeded. Details: wiki/artifacts/advisor-architecture-sketch-2026-09-17/implementation-checks.json. This is an extraction smoke check, not a training pilot or result.

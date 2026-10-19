# Shared architecture and evaluation contract

[Architecture index](README.md)

## Ordered outputs

| Slice | Meaning | Count |
|---|---|---:|
|0:1|Agentness|1|
|1:11|Agent classes|10|
|11:33|Actions|22|
|33:49|Locations|16|
|49:98|Duplex compositions|49|
|98:184|Triplet compositions|86|

“49 primitives” includes agentness,10agents,22actions,16locations. There are183semantic labels plus agentness. Valid compositions come from label strings/verified constraints, never raw legacy `duplex_childs` or `triplet_childs` arrays.

## Features and spatial context

A padded eight-frame crop sequence produces **one vector**, not eight independent crop features. Original crop width is 1024; scaled visual width 1408. Contextual full-frame clips produce a 16×16 keyframe token map. Unpadded candidate coordinates define 7×7 RoIAlign followed by spatial mean. Pooling the full map to 4×4 supplies 16 scene tokens. Position is an 8-value vector (xyxy,center,width,height), embedded after the encoder. It is neither a text prompt nor position pixels injected into the frozen backbone.

The older Stage 2 uses I3D P3 stride 8/256 channels and 1×1 RoIAlign; its pooling is a different implementation. Stages 4–6 use padded image crops and no contextual RoIAlign branch.

## Contextual forward and gradient paths

Projected crop c, RoI r, scene summary s and geometry g are concatenated. The standard visual skip adds c only around fusion MLP. Visual RoI → language attention to adapted phrases → joint RoI. Classifier receives joint RoI+c+phrase similarity scores. No CompMLP in standard standalone or blend experts.

Scene attention: Q=c+g; K=V=entire-frame tokens. Language attention: Q=visual RoI; K=V=adapted phrases. Both are trained when present. Corrected all 184 contrastive loss aligns **pre-language visual RoI** with **adapted text**, not joint RoI or classifier probabilities. Encoders remain frozen in cached-head studies. The old 86 auxiliary instead uses frozen phrase targets.

## Training, selection and detector evaluation

GT positives and background negatives train semantic heads; evaluation candidates come from YOLO. This is not GT-box detector evaluation. Original head studies have their own training recipes; controlled contextual studies use 420 expert videos,90 development videos and 90 reserved gate videos, with 45 reserved selector videos for scalar blending. Follow each exact protocol instead of treating those counts as universal.

At detector assembly, YOLO supplies agentness and agent classes; semantic action/location/composition probabilities are weighted with candidate confidence as implemented in each evaluator. In the contextual blend, primitive probabilities are retained from the classification expert before assembly and only 135 composition probabilities blend. Agent channels are overridden by YOLO, so their unchanged AP is not a learned-head gain.

Headline comparisons use the locked 36,717 validation-frame detector protocol atIoU0.5. Tail 47,deep 28,common 39 subsets are training-frequency-derived triplet groups. Never relabel development crop AP as detector AP, local reproduction as a published metric, or means as best seeds. Official validation has been examined repeatedly during development; it is not an untouched test set.

## Interpretation

Classification-only still uses language attention and phrase scores. “No contrastive” does not mean “no language.” DCB is a training loss; global AP-selected blending is a fixed inference scalar, not a learned sample-dependent gate. Scaling changes both encoder checkpoints and input/numerical paths. Negative ablations and historical corrections remain evidence, not extra successful stages.

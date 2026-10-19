# CSEB: Scaled Contextual Scene-Attention Expert Blend + DCB

[Architecture index](../README.md) · [Shared tensor and evaluation contract](../CONTRACT.md)

Headline configuration: frozen scaled encoders, two contextual DCB experts, development-AP-selected composition blend.

![CSEB: Scaled Contextual Scene-Attention Expert Blend + DCB architecture](../assets/cseb-scaled.png)

Established thesis diagram, copied without geometry or label changes. [Editable draw.io](../assets/cseb-scaled.drawio) · [Source paths and SHA-256](../assets/provenance.json).

## Forward path

Frozen InternVideo 2-1B visual encoder yields 1408-D crop/RoI/scene features. Matching frozen text system yields 184×768 phrase bank → trainable 768→512 input projection → residual text adapter. Context head stays 512-D. Classification and all 184 contrastive experts each output 184 logits; primitive 49 retained from classification, composition 135 probability-blended.

## Training and inference

Both encoder checkpoints change, not just the video tower. Official visual base+CLIP projection and InternVL text assets are required. Head-only training uses cached features; all 184 contrastive trains visual/context projections and adapted text, never the frozen encoders. Use exact protocol, tokenizer and asset hashes.

## Implementation

- [model.py](../../../research/post-proposal/contextual-roi-1b/model.py) — `research/post-proposal/contextual-roi-1b/model.py`
- [protocol.json](../../../research/post-proposal/contextual-roi-1b/protocol.json) — `research/post-proposal/contextual-roi-1b/protocol.json`
- [prepare_blend.py](../../../research/post-proposal/contextual-roi-1b/prepare_blend.py) — `research/post-proposal/contextual-roi-1b/prepare_blend.py`
- [evaluate_cached.py](../../../research/post-proposal/contextual-roi-1b/evaluate_cached.py) — `research/post-proposal/contextual-roi-1b/evaluate_cached.py`

## Variants and interpretation

The shortened “InternVideo 2-1B Text Encoder” display label names the released system, not a 1B text parameter count. Matching text weights originate from InternVL. Scaling also changes numerical extraction path; do not attribute every gain solely to parameter count. [End-to-end video demo](../../../research/post-proposal/demo/README.md) runs fresh YOLO and visual encoding on raw input video. Its annotated output is recorded playback, not a real-time claim.

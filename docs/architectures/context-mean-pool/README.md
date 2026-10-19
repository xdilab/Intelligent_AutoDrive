# Contextual + Mean Pool + Focal

[Architecture index](../README.md) · [Shared tensor and evaluation contract](../CONTRACT.md)

Standalone contextual head with mean-pooled scene summary.

![Contextual + Mean Pool + Focal architecture](../assets/context-mean.png)

Established thesis diagram, copied without geometry or label changes. [Editable draw.io](../assets/context-mean.drawio) · [Source paths and SHA-256](../assets/provenance.json).

## Forward path

Inputs: crop vector, actor RoI feature, entire-frame scene tokens, and normalized candidate-box position. Project to 512-D; fuse the four inputs; crop-only residual + LayerNorm creates the visual RoI. Language cross-attention to adapted phrases creates the joint RoI. Concatenate joint RoI (512), projected crop (512), and phrase similarities (184) → MLP(1208,512,184). No Composition MLP in this readout. Scene summary is the mean over projected scene tokens, not an average over crop/RoI/position.

## Training and inference

Both pretrained encoders and cached inputs are frozen. Crop/text adapters, projections, geometry MLP, fusion MLP, language attention and classifier train. Classification covers all 184 channels; “classification only” means no contrastive auxiliary, not no language. See the shared contract for final YOLO score assembly.

## Implementation

- [model.py](../../../research/post-proposal/contextual-roi/model.py) — `research/post-proposal/contextual-roi/model.py`
- [cache_context.py](../../../research/post-proposal/contextual-roi/cache_context.py) — `research/post-proposal/contextual-roi/cache_context.py`
- [protocol.json](../../../research/post-proposal/contextual-roi/protocol.json) — `research/post-proposal/contextual-roi/protocol.json`

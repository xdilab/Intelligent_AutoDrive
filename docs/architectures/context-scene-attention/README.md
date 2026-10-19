# Contextual + Scene Attention + Focal

[Architecture index](../README.md) · [Shared tensor and evaluation contract](../CONTRACT.md)

Standalone contextual head with actor-conditioned scene attention.

![Contextual + Scene Attention + Focal architecture](../assets/context-attention.png)

Established thesis diagram, copied without geometry or label changes. [Editable draw.io](../assets/context-attention.drawio) · [Source paths and SHA-256](../assets/provenance.json).

## Forward path

Inputs: crop vector, actor RoI feature, entire-frame scene tokens, and normalized candidate-box position. Project to 512-D; fuse the four inputs; crop-only residual + LayerNorm creates the visual RoI. Language cross-attention to adapted phrases creates the joint RoI. Concatenate joint RoI (512), projected crop (512), and phrase similarities (184) → MLP(1208,512,184). No Composition MLP in this readout. Scene attention: Q=projected crop+box-position embedding; K=V=projected entire-frame tokens. Language attention is separate: Q=visual RoI; K=V=adapted label phrases.

## Training and inference

Both pretrained encoders and cached inputs are frozen. Crop/text adapters, projections, geometry MLP, fusion MLP, language attention and classifier train. Classification covers all 184 channels; “classification only” means no contrastive auxiliary, not no language. See the shared contract for final YOLO score assembly. Scene attention is trained as well.

## Implementation

- [model.py](../../../research/post-proposal/contextual-roi/model.py) — `research/post-proposal/contextual-roi/model.py`
- [protocol.json](../../../research/post-proposal/contextual-roi/protocol.json) — `research/post-proposal/contextual-roi/protocol.json`

## Variants and interpretation

The baseline visual skip adds projected crop around the visual fusion MLP. There is no mean-scene residual around attention; that is a separate experiment.

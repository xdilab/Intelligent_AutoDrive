# Contextual + Scene Attention + Focal + All 184 Contrastive

[Architecture index](../README.md) · [Shared tensor and evaluation contract](../CONTRACT.md)

Scene attention with the corrected all-label adapted-text auxiliary.

![Contextual + Scene Attention + Focal + All184 Contrastive architecture](../assets/context-attention-all184.png)

Established thesis diagram, copied without geometry or label changes. [Editable draw.io](../assets/context-attention-all184.drawio) · [Source paths and SHA-256](../assets/provenance.json).

## Forward path

Inputs: crop vector, actor RoI feature, entire-frame scene tokens, and normalized candidate-box position. Project to 512-D; fuse the four inputs; crop-only residual + LayerNorm creates the visual RoI. Language cross-attention to adapted phrases creates the joint RoI. Concatenate joint RoI (512), projected crop (512), and phrase similarities (184) → MLP(1208,512,184). No Composition MLP in this readout. Scene attention Q=crop+position, K/V=entire-frame tokens.

## Training and inference

Both pretrained encoders and cached inputs are frozen. Crop/text adapters, projections, geometry MLP, fusion MLP, language attention and classifier train. Classification covers all 184 channels; “classification only” means no contrastive auxiliary, not no language. See the shared contract for final YOLO score assembly. Both attention modules train. All 184 contrastive supervision connects pre-language visual RoI to adapted phrases (λ0.001, temperature 0.07), not final scores.

## Implementation

- [model.py](../../../research/post-proposal/contextual-roi-all184/model.py) — `research/post-proposal/contextual-roi-all184/model.py`
- [protocol.json](../../../research/post-proposal/contextual-roi-all184/protocol.json) — `research/post-proposal/contextual-roi-all184/protocol.json`

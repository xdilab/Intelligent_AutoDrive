# Contextual + Composition MLP

[Architecture index](../README.md) · [Shared tensor and evaluation contract](../CONTRACT.md)

Readout ablation: replace the flat 184 classifier with primitive and composition branches.

![Contextual + Composition MLP architecture](../assets/context-all184.png)

Shared contextual parent figure; its final flat184 classifier is replaced by the two readout branches specified below. No approved distinct composition-readout export was found; this parent illustration is not an exact variant diagram. [Editable draw.io](../assets/context-all184.drawio) · [Source paths and SHA-256](../assets/provenance.json).

## Forward path

Inputs: crop vector, actor RoI feature, entire-frame scene tokens, and normalized candidate-box position. Project to 512-D; fuse the four inputs; crop-only residual + LayerNorm creates the visual RoI. Language cross-attention to adapted phrases creates the joint RoI. Let f=[joint RoI512,crop 512,phrase scores 184]. Flat MLP(1208,512,49) gives primitive logits. CompMLP takes [sigmoid(primitives),f] (1257) →512→135; output concatenates primitive and composition logits.

## Training and inference

Train contextual backbone and both readouts jointly. Focal-only and focal+corrected all 184 conditions form the matched comparison. This is not Stage 5 with frozen heads.

## Implementation

- [model.py](../../../research/post-proposal/contextual-roi-comp/model.py) — `research/post-proposal/contextual-roi-comp/model.py`
- [protocol.json](../../../research/post-proposal/contextual-roi-comp/protocol.json) — `research/post-proposal/contextual-roi-comp/protocol.json`

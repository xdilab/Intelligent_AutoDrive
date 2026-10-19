# Contextual + Scene Attention + Expert Blend + Focal

[Architecture index](../README.md) · [Shared tensor and evaluation contract](../CONTRACT.md)

Combine two separately trained contextual experts with a development-selected probability blend.

![Contextual + Scene Attention + Expert Blend + Focal architecture](../assets/context-blend.png)

Established thesis diagram, copied without geometry or label changes. [Editable draw.io](../assets/context-blend.drawio) · [Source paths and SHA-256](../assets/provenance.json).

## Forward path

Classification expert: scene+language attention, focal 184. Contrastive expert: same architecture, focal 184+0.001 all 184 contrastive. Keep primitive 49 from classification expert; blend composition 135 as (1−w)p_cls+w p_con. Both experts use sigmoid probabilities.

## Training and inference

Freeze experts after development checkpoint selection. Select one global scalar per seed from {0,0.125,…,1} on 45 reserved development videos using triplet crop AP; do not tune on final validation. The blend is neither a per-frame neural gate nor feature averaging.

## Implementation

- [train_gate.py](../../../research/post-proposal/contextual-gate/train_gate.py) — `research/post-proposal/contextual-gate/train_gate.py`
- [common.py](../../../research/post-proposal/contextual-gate/common.py) — `research/post-proposal/contextual-gate/common.py`
- [protocol.json](../../../research/post-proposal/contextual-gate/protocol.json) — `research/post-proposal/contextual-gate/protocol.json`

## Variants and interpretation

Both experts train all labels. “Common expert” and “Tail expert” are misleading permanent names: which subset improves depends on objective and run.

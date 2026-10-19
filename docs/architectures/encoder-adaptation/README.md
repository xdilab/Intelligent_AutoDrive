# Encoder Adaptation Conditions (Stage 5/6 and Contextual)

[Architecture index](../README.md) · [Shared tensor and evaluation contract](../CONTRACT.md)

BDD-X/ROAD encoder adaptation changes representation conditions without inventing another stage.

![Encoder Adaptation Conditions (Stage 5/6 and Contextual) architecture](../assets/stage6.png)

Original Stage 6 reference; adaptation training graph and trainable encoder subset are defined in each linked protocol. [Editable draw.io](../assets/stage6.drawio) · [Source paths and SHA-256](../assets/provenance.json).

## Forward path

Preserve the selected downstream architecture (e.g. Stage 5 or Stage 6). Adapt the specified pretrained encoder components using the study objective, then extract the matching features and evaluate the matching downstream head. Contextual ROAD transfer consumes the resulting adapted representations.

## Training and inference

Unlike frozen-cache head training, encoder adaptation needs gradients through encoder computation. Focal-only and focal+contrastive are distinct arms. Report detector evaluation separately from development crop AP and retrieval.

## Implementation

- [README.md](../../../research/post-proposal/stage56-full/README.md) — `research/post-proposal/stage56-full/README.md`
- [protocol.json](../../../research/post-proposal/stage56-full/protocol.json) — `research/post-proposal/stage56-full/protocol.json`
- [model.py](../../../research/post-proposal/contextual-road-transfer/model.py) — `research/post-proposal/contextual-road-transfer/model.py`
- [protocol.json](../../../research/post-proposal/contextual-road-transfer/protocol.json) — `research/post-proposal/contextual-road-transfer/protocol.json`
- [smoke-model.py](../../../research/post-proposal/historical/bddx-contrastive-pilot/smoke-model.py) — `research/post-proposal/historical/bddx-contrastive-pilot/smoke-model.py`

## Variants and interpretation

This page groups encoder conditions only. Baseline Stage 6 diagram shown for architecture reference; frozen icons describe its original inference/head-training condition, not adaptation-time gradients.

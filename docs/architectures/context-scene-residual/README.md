# Contextual + Scene-Attention Mean Residual (ablation)

[Architecture index](../README.md) · [Shared tensor and evaluation contract](../CONTRACT.md)

Parameter-matched scene residual around attention, separate from the full-context MLP skip.

![Contextual + Scene-Attention Mean Residual (ablation) architecture](../assets/context-attention.png)

Parent scene-attention figure. The extra mean-token residual described below is absent from this inherited image; no approved residual-specific figure was found. [Editable draw.io](../assets/context-attention.drawio) · [Source paths and SHA-256](../assets/provenance.json).

## Forward path

Replace scene summary by attention(c+g,tokens,tokens)+mean(tokens). Existing crop-only visual MLP residual and all text/readout paths stay unchanged.

## Training and inference

DCB classification and all 184 contrastive experts plus global AP-selected blend. No new parameters; fixed unit-weight addition intentionally changes initial predictions/magnitude.

## Implementation

- Local study source: `research/post-proposal/contextual-roi-scene-skip/model.py`
- Local study source: `research/post-proposal/contextual-roi-scene-skip/protocol.json`

# Thesis model architecture catalog

Current documentation snapshot: October19,2026. Start with [scaled CSEB](cseb-scaled/README.md), the headline frozen-encoder expert-blend configuration. Stage numbers identify thesis evolution; experiment-folder and workbook-row numbers do not.

Every entry embeds a repository-local figure and links implementation. Figure copies retain established draw.io ancestry, with source paths and SHA-256 in [assets/provenance.json](assets/provenance.json). Later skip/readout/gate ablations explicitly show a **parent figure with the changed path described underneath** when no approved dedicated diagram exists; they must not be presented as exact variant exports.

[Shared tensor/evaluation contract](CONTRACT.md) · [Research study index](../../research/post-proposal/STUDY-INDEX.md) · [Historical exploration](HISTORY.md)

| Architecture / condition | Purpose |
|---|---|
| [Stage 0: 3D-RetinaNet (I3D)](stage-0/README.md) | Published detector baseline and local replication; retain their separate result identities. |
| [Stage 1: Score Transfer](stage-1/README.md) | Replace localization with YOLO while transferring existing I3D semantic scores. |
| [Stage 2: RoI Linear Scoring Head](stage-2/README.md) | Learn semantic scores at the YOLO box coordinates instead of copying an overlapping detection. |
| [Stage 3: RoI Linear Scoring Head + Composition MLP](stage-3/README.md) | Add a learned composition readout to the I3D RoI classifier. |
| [Stage 4: Semantic Crop Classifier](stage-4/README.md) | Score label phrases against a learned projection of an eight-frame crop feature. |
| [Stage 5: Semantic Crop Classifier + Comp MLP](stage-5/README.md) | Visual composition model: crop features and flat primitive scores; no explicit phrase-score input. |
| [Stage 6: Semantic Crop Classifier + Phrase Fusion + Comp MLP](stage-6/README.md) | Extend Stage 5 with explicit phrase-composition evidence. |
| [Contextual + Mean Pool + Focal](context-mean-pool/README.md) | Standalone contextual head with mean-pooled scene summary. |
| [Contextual + Scene Attention + Focal](context-scene-attention/README.md) | Standalone contextual head with actor-conditioned scene attention. |
| [Contextual + Mean Pool + Focal + Triplet Contrastive (historical)](context-triplet-contrastive/README.md) | Historical auxiliary-loss design retained to explain the correction. |
| [Contextual + Mean Pool + Focal + All184 Contrastive](context-all184-contrastive/README.md) | Corrected contrastive objective trains both visual and adapted-text branches. |
| [Contextual + Scene Attention + Focal + All184 Contrastive](context-scene-attention-contrastive/README.md) | Scene attention with the corrected all-label adapted-text auxiliary. |
| [Contextual + Composition MLP](context-composition/README.md) | Readout ablation: replace the flat184 classifier with primitive and composition branches. |
| [Stage 7: Contextual Refinement of Frozen Stage 5 Heads](stage-7-refinement/README.md) | Historical exploratory bridge from contextual features into Stage 5 readout. |
| [Contextual + Scene Attention + Expert Blend + Focal](context-expert-blend/README.md) | Combine two separately trained contextual experts with a development-selected probability blend. |
| [Contextual + Scene Attention + Expert Blend + DCB](context-dcb/README.md) | Same two-expert inference topology; difficulty-calibrated BCE replaces focal classification. |
| [CSEB: Scaled Contextual Scene-Attention Expert Blend + DCB](cseb-scaled/README.md) | Headline configuration: frozen scaled encoders, two contextual DCB experts, development-AP-selected composition blend. |
| [Contextual + Full-Context Concatenative Skip (ablation)](context-full-context-skip/README.md) | A distinct skip ablation, not the crop-only residual in the headline model. |
| [Contextual + Scene-Attention Mean Residual (ablation)](context-scene-residual/README.md) | Parameter-matched scene residual around attention, separate from the full-context MLP skip. |
| [Contextual Expert Gates (supporting alternatives)](context-learned-gates/README.md) | Supporting classwise and sample-conditioned routing alternatives to the global AP-selected blend. |
| [Encoder Adaptation Conditions (Stage 5/6 and Contextual)](encoder-adaptation/README.md) | BDD-X/ROAD encoder adaptation changes representation conditions without inventing another stage. |
| [Language Controls and No-Composition Baselines](language-controls/README.md) | Controls test phrase dependence while preserving the appropriate parent architecture and protocol. |

## How to reproduce a result

Read the linked protocol, then source model, cache extractor, trainer, selector and evaluator. Preserve label order, split manifests, seed, candidate rows, encoder revision and checkpoint selection. These README files do not certify historical absolute paths or claim that every environment can launch unchanged. Data, large encoder weights, caches and run checkpoints remain external; see the study index. Do not compare development crop AP or retrieval with detector frame-mAP.

New documentation does not restart or retrain any experiment. Earlier README launch/status paragraphs are dated historical receipts, not live scheduler status.

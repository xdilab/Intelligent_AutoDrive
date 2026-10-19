# Post-proposal research code

Canonical home for new experiment code, per Brandon's September 14, 2026 instruction. The research wiki remains the home for notes, result artifacts and diagrams.

`historical/` preserves all 145+ executable/source files and available protocol configurations from wiki artifacts at consolidation time, including experiments, analysis, dataset exploration and presentation/animation generators. `code-inventory.json` records every source path and SHA-256. These are exact historical copies: scripts may expect data/results and companion assets at their original wiki or cluster paths. The wiki copies remain archived so old reports and launch provenance do not break; do not treat these archives as a second development branch.

`stage56-full/` is the active full-dataset Stage 5/6 encoder adaptation study. Develop and launch new code from this repository. Large data, checkpoints and collected experiment outputs stay outside Git. Existing exp12 working changes were not altered.

## September 28 synchronization

See [STUDY-INDEX.md](STUDY-INDEX.md) for current study directories, historical revisions, compact result snapshots and external asset locations. `sync-manifest-20260928.json` verifies the newly consolidated files. Active contextual, gate, DCB, language-control and composition studies are alongside `stage56-full/`; its earlier description as the only active study is historical.

## Architecture documentation

Browse the [model-by-model README catalog with diagrams](../../docs/architectures/README.md), including the [headline scaled CSEB](../../docs/architectures/cseb-scaled/README.md). The catalog separates architectures from adaptation/loss conditions and preserves historical/corrected variants.

## Prototype

[End-to-end CSEB video inference](demo/README.md) uses the exact frozen visual encoder and trained experts, with fresh detection and feature extraction for each input video. Large runtime assets remain outside Git.

## Causal student pilot (October 19, 2026)

The [CSEB student pilot](cseb-student/README.md) contains a shared frozen frame encoder, causal scene-memory head, supervised/distilled controls, checkpoint verification and fresh inference. See its supported-class proposal metrics and paired latency before considering a larger study; it is not a replacement headline detector.

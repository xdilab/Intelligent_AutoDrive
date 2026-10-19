# Headline thesis results

Snapshot collected October 19, 2026. These files preserve the completed scaled CSEB study results.

- [Full summary](scaled-cseb/full-summary.json): all three seeds, means and sample standard deviations for classification, contrastive and blended configurations.
- [Raw detector results](scaled-cseb/runs/): nine per-run JSON files, including every class AP and checkpoint/evaluation signatures.
- [Training and evaluation protocol](../../contextual-roi-1b/protocol.json).
- [Demo runtime verification](demo-verification.json): end-to-end processing timings and score checks for the 120-frame prototype.
- [Collection provenance](provenance.json): original evidence paths and SHA-256 values.

Detector values are frame-mAP at IoU0.5, in percent, on the locked36717 ROAD-Waymo validation frames. Tail47,deep28 andcommon39 use training-frequency triplet groups. Each raw detector file hash was checked against the original summary; means and sample standard deviations were recomputed from seeds0,1,2.

Demo timing describes the separate qualitative video run with confidence0.25. The detector study uses confidence0.001. The demo verification file contains runtime and score consistency checks, not dataset AP.

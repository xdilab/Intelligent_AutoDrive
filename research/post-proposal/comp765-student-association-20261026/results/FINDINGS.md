# Explicit cross-actor associations on the causal student

Completed exploratory ROAD-Waymo COMP 765 pilot, started 2026-10-26. The original frozen-ResNet distilled student and its scene GRU remain unchanged. This tests incremental usefulness of an explicit association correction; it does not identify what the GRU internally represents. No new student training was performed.

The mean official triplet detector mAP increased by +0.107 percentage points. The paired95%t interval is [+0.046, +0.168]. With only3seeds and a reused validation benchmark, this is preliminary evidence, not a definitive generalization claim.

All values below are percent AP, mean ± sample standard deviation across seeds0,1,2. IoU0.5;36,717scoredframes,6,163,968fixed candidates,198videos;85/86triplet classes have evaluation GT. All86classes remain in the mean.

| Condition | Triplet | Duplex | Tail47 | Deep28 | Common39 |
|---|---:|---:|---:|---:|---:|
| baseline | 6.586 ± 0.339 | 11.721 ± 0.244 | 3.351 ± 0.517 | 2.646 ± 0.533 | 10.485 ± 0.180 |
| association | 6.693 ± 0.337 | 11.792 ± 0.229 | 3.374 ± 0.517 | 2.676 ± 0.538 | 10.692 ± 0.217 |
| frequency | 6.613 ± 0.337 | 11.772 ± 0.230 | 3.389 ± 0.518 | 2.706 ± 0.558 | 10.499 ± 0.182 |

Association minus the context-free frequency control is +0.080 triplet points (95%paired interval [-0.034, +0.193]). This control uses the same rule effects with training-marginal antecedent frequencies, rather than the current scene. It is one calibration control, not an exhaustive test of calibration methods.

Tail47 changes by +0.023 points. Across the47tail classes, 8 improve, 2 decline and 37 tie in mean AP. Complete duplex and triplet changes are in per-class.csv; individual seed endpoints are in per-seed.csv.

## Frozen experiment

- Same original distilled checkpoints, seeds0,1,2, selected by historical development supervised BCE. Encoder: frozen ImageNetV2 ResNet50 through layer3, shared448x672 input; actor RoI features/geometry/agent embedding plus256-dimensional scene GRU. No teacher is used at inference.
- Rules mined only from ground-truth actors in the same64training-video32frame segments. Each target-agent/video contributes equal weight. Minimum5joint actor-frames and3distinct training videos; shrinkage toward target-agent marginal, bounded log-odds effect. Labels are mapped through master label strings; no child arrays.
- Other actors only, fixed spatial neighborhood and overlap exclusion. Identity/parent-child event implications removed. At inference, context candidates have detector confidence≥0.25, class-agnostic NMS0.5 and cap30; output candidates remain unchanged. Single pass uses original scores, never corrected context feedback.
- Only duplex/triplet logits change; target-agent compatibility enforced. Primitive scores are bit-identical. Final detector confidence is applied once; contextual confidence only weights evidence.
-2,537rules cover48/86triplet and36/49duplex classes; tail coverage=10/47, deep=1/28, common=38/39. Sparse coverage limits conclusions about rare events.
- One common strength per condition selected from[0,0.25,0.5,1] by mean development triplet proposal AP on12videos. Both selected1. Development AP and detector AP are different metrics and are not compared directly. No strength/rule changes after official evaluation.
- Full causal prefixes process38,663current-image steps, including unscored frames; state resets only at video boundaries. All current image hashes match the original evaluation. The unchanged official evaluator/GT/candidates/tail manifest are reused. Primitive-group metrics for corrected conditions are inherited only after per-frame exact score equality checks.

## Interpretation and limits

The comparison measures the practical effect of these fixed association rules on an already contextual student. Any gain can combine scene context with score calibration; even an advantage over the frequency control does not establish a unique learned mechanism. A weak or negative result would not prove that the GRU already learned all useful associations. No GRU ablation was added.

The final198-video benchmark has prior research use and exposure in detector checkpoint selection. Student/semantic-teacher training and selection are disjoint from these videos, but this is not an untouched full-system test. Student development videos came from the teacher's historical development pool. Rules were not tuned on final AP. The3existing checkpoints are paired evaluation replicates, not3new training runs.

Ground-truth presence during mining differs from uncertain detector/student evidence at inference. Nearby actors can share a scene without interacting. Observed associations do not establish causation, intention or future events. Short training segments, dependent actor-frames and limited rare-class support constrain generalization. No statistical non-significance is interpreted as equivalence. There is no deployment-safety claim.

## Reproduction and artifacts

Code: /data/repos/ROAD_Reason/research/post-proposal/comp765-student-association-20261026

Run root: /home/brandon/.local/share/road-reason/comp765-student-association-20261026

Protocol: protocol.json. Frozen code/source/rule/selection hashes: run-root freeze.json. Original model hashes/backups and dependency details: provenance.json and preflight.json. All rules: rules.csv and rules.json. Split manifest: split-manifest.json. Per-seed detector files: runs/<condition>-seed<seed>/detector-results.json. Raw-video predictions and latency records: runtime/<video>/.

No thesis, defense slides, original checkpoints or working inference bundle were changed. No training curves are generated because this is an inference-only pilot. Analysis plots summarize every class, paired endpoints and incremental correction cost.

## Runtime scope

Teacher-free adapter ran593sampled frames across3complete videos at10Hz sampling, with fresh state per video and no future decoding. Model weights match the evaluated seed0checkpoint exactly; the same correction module is used in evaluation and the adapter. Asset access was restricted to the original standalone bundle's model files; no teacher/features were opened. CPU/GPU retained-memory bounds passed.

Median correction time: 0.939ms; p95: 0.982ms. Median serial total: 57.95ms; p95: 68.92ms. Total includes MP4 decoding, production detector, student, correction and compressed JSON export; excludes camera/display/transport. This file-processing measurement is not a claim of end-to-end real-time latency. Runtime uses production candidate filtering; detector AP uses the fixed full-candidate manifest.

Independent raw-video replay passed all 593 frames: maximum absolute score discrepancy 1.79e-7 versus tolerance 1e-6; candidate boxes, IDs, confidence and primitive scores matched exactly. The replay uses historical baseline predictions only as audit references. Independent arithmetic reconstruction passed 30 condition/contrast summaries.

# Full-video mining and two-event rules: ten-seed study

Completed COMP 765 extension, started October 26, 2026. All ten original distilled student checkpoints (seeds 0–9) were evaluated under seven fixed conditions. No student retraining, architecture change, rule retuning or checkpoint promotion occurred. The original three-seed block remains immutable.

The user requested ten seeds after the first three outcomes were visible. The seven additional seeds and unchanged protocol were fixed before their corrected-condition evaluations, and all were completed regardless of outcomes. Combined intervals are exploratory; this is not a retrospective confirmatory preregistration or a new independent video test.

## Detector results

Official detector AP at IoU 0.5, percent, mean ± sample standard deviation across ten paired checkpoints:

| Condition | Triplet | Duplex | Tail47 | Deep28 | Common39 |
|---|---:|---:|---:|---:|---:|
| baseline | 6.436 ± 0.391 | 11.656 ± 0.407 | 3.252 ± 0.475 | 2.427 ± 0.610 | 10.273 ± 0.346 |
| short_single | 6.550 ± 0.374 | 11.732 ± 0.397 | 3.274 ± 0.471 | 2.443 ± 0.610 | 10.497 ± 0.317 |
| short_frequency | 6.459 ± 0.386 | 11.706 ± 0.378 | 3.282 ± 0.468 | 2.450 ± 0.621 | 10.288 ± 0.346 |
| full_single | 6.544 ± 0.358 | 11.745 ± 0.386 | 3.229 ± 0.450 | 2.421 ± 0.611 | 10.539 ± 0.309 |
| full_frequency | 6.464 ± 0.386 | 11.682 ± 0.390 | 3.297 ± 0.466 | 2.458 ± 0.626 | 10.281 ± 0.347 |
| full_pair | 6.560 ± 0.351 | 11.752 ± 0.381 | 3.233 ± 0.441 | 2.433 ± 0.615 | 10.569 ± 0.307 |
| full_pair_frequency | 6.468 ± 0.386 | 11.705 ± 0.383 | 3.303 ± 0.467 | 2.467 ± 0.628 | 10.282 ± 0.348 |

Triplet changes in percentage points, with nominal paired-t intervals (df 2, 6 and 9 respectively):

| Contrast | Original 3: mean [95% interval] | Added 7: mean [95% interval] | All 10: mean [95% interval] |
|---|---:|---:|---:|
| full_single-minus-short_single | -0.010 [-0.126, +0.106] | -0.003 [-0.033, +0.026] | -0.005 [-0.030, +0.019] |
| full_pair-minus-full_single | +0.024 [-0.012, +0.060] | +0.012 [-0.003, +0.027] | +0.015 [+0.004, +0.027] |
| full_pair-minus-baseline | +0.120 [-0.030, +0.271] | +0.125 [+0.062, +0.189] | +0.124 [+0.079, +0.169] |
| full_single-minus-full_frequency | +0.059 [-0.068, +0.186] | +0.089 [+0.046, +0.132] | +0.080 [+0.046, +0.114] |
| full_pair-minus-full_pair_frequency | +0.074 [-0.085, +0.233] | +0.100 [+0.046, +0.153] | +0.092 [+0.051, +0.133] |
| full_pair_frequency-minus-full_frequency | +0.009 [-0.019, +0.036] | +0.001 [-0.001, +0.003] | +0.003 [-0.001, +0.008] |
| short_single-minus-short_frequency | +0.080 [-0.034, +0.193] | +0.095 [+0.073, +0.117] | +0.090 [+0.069, +0.112] |

Additional mining frames: mean triplet change **-0.005 points**, 95% paired interval [-0.030, +0.019]. Tail47 changes by -0.044 points; 11 tail classes improve, 15 decline and 21 tie in mean AP.

The incremental two-event rules: mean triplet change **+0.015 points**, 95% paired interval [+0.004, +0.027]. Tail47 changes by +0.003 points; 11 tail classes improve, 11 decline and 25 tie in mean AP.

The complete contextual correction: mean triplet change **+0.124 points**, 95% paired interval [+0.079, +0.169]. Tail47 changes by -0.019 points; 15 tail classes improve, 11 decline and 21 tie in mean AP.

The combined full-video/pair condition differs from the original short-rule method by +0.010 triplet points. This descriptive difference is not a substitute for the separately declared data-scale and rule-complexity contrasts above.

Single-event context versus its frequency control: +0.080 triplet points, interval [+0.046, +0.114]. Two-event context versus its frequency control: +0.092 triplet points, interval [+0.051, +0.133]. 

These controls use training-marginal evidence with the same fixed rule effects. A baseline gain can combine scene information with calibration; the controls help assess that distinction without proving a unique mechanism. No non-significant interval is interpreted as equivalence. The intervals describe variability across these training seeds conditional on one reused benchmark, not uncertainty over new videos. Multiple endpoints/contrasts are exploratory and unadjusted.

## Fixed study

- Architecture: frozen ImageNet ResNet50 through layer3, shared 448×672 visual features, actor RoI/geometry/agent features and a 256-dimensional scene GRU. Exactly 9,651,864 semantic parameters; detector parameters counted separately. The teacher is absent at inference, and only present/past frames update state.
- Same seven conditions: baseline, short_single, short_frequency, full_single, full_frequency, full_pair, full_pair_frequency. All use the same checkpoint within each seed. Strength remains 1 from the original pilot; no new AP tuning.
- Full mining uses all 12,270 annotated frames in the same 64 training videos (11,948 nonempty target frames; 269,600 target actor-frames), versus 2,036 annotated frames and 45,766 actor-frames in the original segments. This is extra ground-truth supervision, not new video identities or student retraining.
- Frozen rules: 3,970 full single-event rules plus 1,532 residual pair rules over 395 distinct antecedent pairs. Two event labels may occur on the same or different eligible neighboring actors. Identity/parent-child implications are excluded. Minimum support is 5 actor-frames and 3 videos, with video/agent balancing, shrinkage and a fixed cap of 16 pair rules per consequent. All retained and discarded qualified pair candidates remain available in pair-candidates.csv.
- Pair effects add only when their shrunk target conditional lies above both single conditionals or below both. Inference uses min(original event evidences), one pass, no feedback, bounded combined log-odds change, and unchanged primitive outputs. Final detector confidence is applied once.
- Same 198 official validation videos, 36,717 scored frames, 38,663 causal prefix steps and 6,163,968 fixed candidates. 85/86 triplets have evaluation GT; all 86 remain in the mean. Tail47/Deep28/Common39 use the authoritative training-frequency manifest. No mis-indexed child arrays are used.
- Original 21 result files were reused after their complete score/integrity audit. Seven additional baseline arrays were regenerated and matched their original whole-file SHA256 hashes before reusing baseline AP. All 42 additional corrected-condition detector evaluations were computed. Complete per-seed and all 135 event-class outcomes are in per-seed.csv and per-class.csv.

## Runtime and reproducibility

The audited, prespecified seed-0 raw-video adapter is unchanged. Its prior 593-frame, three-video runtime/replay record is reused by hash; no new speed improvement is claimed from adding seeds. It accepts raw MP4 without the teacher, cached features or future frames. Candidates and primitive outputs matched the verified baseline exactly, and an independent NumPy full-pair replay agreed within the recorded 1e-6 tolerance. Video resets, finiteness and retained-memory checks passed.

Measured full-pair correction median 1.225 ms, p95 1.258 ms. Serial raw-video total median 57.760 ms, p95 69.121 ms, including decoding, detector, student, correction and compressed JSON export. This file-processing measurement excludes camera/display/transport and is not a claim of end-to-end online latency. Production candidate filtering differs from the fixed full-candidate detector AP protocol.

Exact commands and paths are in README.md. protocol.json defines the extension; freeze.json pins code, rules, splits, original result sources and all ten checkpoints. preflight.json/provenance.json record input equivalence, environment and source revisions. Each batch preserves score hashes, baseline replay, all per-class results, image identities and verified cleanup receipts. The final audit independently checks all aggregate arithmetic, per-class rows, frozen assets and checkpoint backups.

## Limits

The official validation set has prior research use and detector checkpoint-selection exposure. Student/semantic-teacher training/selection are disjoint from it, but it is not an untouched full-system test. More seeds improve measurement of training randomness; they do not remove dataset reuse or establish generalization to new video populations. All seeds share one fixed mined rule bank; these intervals do not measure uncertainty from resampling mining data or constructing different rules.

Dependent actor-frame transactions and uncertain predicted context differ from ground-truth mining. Nearby actors need not interact. Associations do not establish causes, intentions, future events or the GRU's internal representations. The pair rule design and cap are one fixed practical implementation, not an exhaustive search. No deployment-safety claim is made. No thesis, slides, original checkpoints, production bundle or remote branch were modified.

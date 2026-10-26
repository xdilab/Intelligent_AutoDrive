# Full-video mining and two-event association rules

Completed exploratory COMP765 extension, 2026-10-26. This is an inference-time extension of the existing teacher–student study, separate from the thesis. The unchanged frozen ResNet50 through layer3, shared448×672 visual encoder, actor RoI/geometry/agent features and256-dimensional scene GRU use the original distilled seeds0,1,2. No new student training was performed.

The substantive interventions are additional training-only mining frames and two-event conjunction rules. The original pilot and its historical artifacts remain immutable. The seed-only ten-seed draft was not launched and is superseded by this methodological extension.

| Condition | Triplet | Duplex | Tail47 | Deep28 | Common39 |
|---|---:|---:|---:|---:|---:|
| baseline | 6.586 ± 0.339 | 11.721 ± 0.244 | 3.351 ± 0.517 | 2.646 ± 0.533 | 10.485 ± 0.180 |
| short_single | 6.693 ± 0.337 | 11.792 ± 0.229 | 3.374 ± 0.517 | 2.676 ± 0.538 | 10.692 ± 0.217 |
| short_frequency | 6.613 ± 0.337 | 11.772 ± 0.230 | 3.389 ± 0.518 | 2.706 ± 0.558 | 10.499 ± 0.182 |
| full_single | 6.683 ± 0.293 | 11.793 ± 0.221 | 3.329 ± 0.461 | 2.660 ± 0.537 | 10.725 ± 0.209 |
| full_frequency | 6.624 ± 0.329 | 11.751 ± 0.223 | 3.411 ± 0.508 | 2.725 ± 0.562 | 10.496 ± 0.184 |
| full_pair | 6.706 ± 0.279 | 11.806 ± 0.217 | 3.347 ± 0.446 | 2.681 ± 0.541 | 10.755 ± 0.221 |
| full_pair_frequency | 6.632 ± 0.327 | 11.777 ± 0.225 | 3.426 ± 0.508 | 2.744 ± 0.566 | 10.496 ± 0.185 |

| Contrast | Triplet change (pp) | 95% paired-t interval |
|---|---:|---:|
| full_single-minus-short_single | -0.010 | [-0.126, +0.106] |
| full_pair-minus-full_single | +0.024 | [-0.012, +0.060] |
| full_pair-minus-baseline | +0.120 | [-0.030, +0.271] |
| full_single-minus-full_frequency | +0.059 | [-0.068, +0.186] |
| full_pair-minus-full_pair_frequency | +0.074 | [-0.085, +0.233] |
| full_pair_frequency-minus-full_frequency | +0.009 | [-0.019, +0.036] |
| short_single-minus-short_frequency | +0.080 | [-0.034, +0.193] |

All metrics are official detector AP@IoU0.5 in percent. Mean ± sample standard deviation across3 paired checkpoints. Evaluation:198 videos,36,717 scoredframes,38,663 causal prefix steps,6,163,968 fixed candidates,85/86 triplets supported; all86 remain in the mean. Full per-seed and all135 event-class results are in per-seed.csv and per-class.csv.

The confidence intervals have only2 degrees of freedom. They summarize checkpoint-seed variability on this reused benchmark, not uncertainty from a new video sample; multiple contrasts are exploratory and unadjusted. Non-significance is not equivalence.

## Controlled design

- Data scale: full_single−short_single holds architecture, checkpoints, inference/evaluator, rule criteria and strength1 fixed; it changes the training annotation coverage.
- Rule complexity: full_pair−full_single adds only the predeclared residual two-event rule contribution. It retains the same expanded mining data and student predictions.
- Frequency controls replace scene-specific evidence with training-marginal event/conjunction presence conditioned on the target agent, using the same effects and denominators. These are matched controls, not exhaustive calibration baselines.
- Baseline and short controls were freshly regenerated. Their entire score arrays matched the immutable pilot SHA256 hashes exactly before reusing the verified pilot AP files. Twelve new detector metric runs evaluated the four full-data conditions across3 seeds.
- Strength1 was inherited from the frozen pilot; no new development selection or final-AP tuning occurred. Development proposal AP is not used as detector AP.

## Mining and pair semantics

All annotated frames of the same64 training videos were inspected: 12,270 annotatedframes and269,600 valid target actor-frames, versus2,036 annotatedframes and45,766 actor-frames in the short segments. Empty-target frames do not create transactions. All original short-segment weights and marginals were reproduced exactly (maximum absolute error0.0). This supplies additional ground-truth supervision; it does not expand the video identities or retrain the student.

Single rules use identical video/target-agent balancing, support≥5 actor-frames in≥3 videos,5 video-equivalent shrinkage, parent/child exclusions and bounded log-odds effects. Full mining yields3,970 single rules. The frozen pair procedure considered8,959 nonhierarchical event pairs, found17,089 effect/support-qualified pair→target rules and retained1,532 rules across395 distinct antecedent pairs. At most16 rules per consequent are retained by training-only video support, effect magnitude and class-ID tie-breaks. pair-candidates.csv includes discarded candidates as well as retained ones; single-rules.csv lists all full single rules.

A pair means two different observed event labels among eligible other actors; both may occur on the same neighboring actor. Identity and hierarchical duplex/triplet implications are excluded between antecedents and consequent. Pair effects are nonzero only when the shrunk pair conditional is above both single conditionals or below both. The residual log-odds effect measures the increment beyond the closer extreme single conditional, then clips to[-1,1]. This is one specified practical rule design, not a universal test of every association-mining method.

At inference, context is drawn once from original current-frame scores, confidence≥0.25, class-agnostic NMS0.5, at most30 neighbors, normalized center distance≤0.5, IoU<0.3 and intersection/minarea<0.5. Per-event evidence is the maximum compatible neighbor score. Pair evidence is min(evidenceA,evidenceB); this agrees with Boolean conjunction but is a soft inference heuristic. The weighted pair contribution is added to the full single contribution and clipped[-1,1]. No corrected scores feed back into context. Primitive outputs remain bit-identical, final detector confidence is applied once, and GRU state resets only at video boundaries.

Rule coverage:

- short_single: 36/49 duplex, 48/86 triplet, 10/47 tail, 1/28 deep and38/39 common consequents.
- full_single: 42/49 duplex, 64/86 triplet, 26/47 tail, 9/28 deep and38/39 common consequents.
- pair_increment: 40/49 duplex, 60/86 triplet, 22/47 tail, 7/28 deep and38/39 common consequents.

## Interpretation

Expanded mining decreased mean triplet mAP by-0.010 percentage points, with paired seed differences +0.018, +0.016, -0.064. Adding two-event rules increased mean triplet mAP by+0.024 percentage points, with paired seed differences +0.029, +0.034, +0.007. The full combined correction increased mean triplet mAP by+0.120 percentage points, with paired seed differences +0.126, +0.178, +0.057. 

For full_single-minus-short_single, tail-class mean AP improves for11 classes, declines for15 and ties for21. For full_pair-minus-full_single, tail-class mean AP improves for12 classes, declines for10 and ties for25. For full_pair-minus-baseline, tail-class mean AP improves for13 classes, declines for13 and ties for21. 

The combined contextual correction differs from its context-free frequency control by+0.074 triplet points. A gain relative to baseline can mix contextual information and calibration; the frequency comparison is necessary to assess this distinction. No claim is made that this identifies the GRU's internal representations.

## Limitations and reproducibility

The198-video official benchmark has prior research use and historical detector checkpoint-selection exposure. Student/semantic-teacher training/selection are disjoint from these videos, but this is not an untouched full-system test. Student development videos overlap historical teacher development. No new independent-test claim is made.

Ground-truth mining and uncertain predicted context differ. Overlapping actor-frame transactions are dependent; video support is descriptive, not a statistical independence guarantee. Spatial proximity does not imply interaction, causation, intention or future events. Three reused checkpoints are paired evaluation replicates, not three new training runs. No deployment-safety claim or teacher-free retraining claim is made. No unplanned architecture, fine-tuning or loss experiment was added.

Protocol: protocol.json. Frozen source/rule/input/checkpoint hashes: run-root freeze.json. Environment and original source revision: provenance.json. Exact split manifest, asset-audit.json, preflight.json, pilot-replay.json, replay-audit.json and final-audit.json preserve validation. No thesis, defense slides, original checkpoints or working inference bundle were changed. There are no new training curves because this is a fixed-model postprocessing study.

## Raw-video runtime

The full-pair adapter processed593 frames across3 complete videos,10 Hz sampling, from raw MP4 with no teacher/features/future frames. The packaged model tensor hash matches the evaluated seed0 checkpoint, and an independent NumPy replay checked all output scores. Memory bounds and video-boundary resets passed.

Full-pair correction median1.225 ms, p951.258 ms. Serial total median57.760 ms, p9569.121 ms; file-processing throughput15.52 frames/s. The total includes decoding, production detector, student, correction and compressed JSON export, excludes camera/display/transport, and is not a claim of online end-to-end latency. Per-component timing, model loading, sample sizes/resolution/candidate counts and GPU memory are in runtime-summary.json and per-video report.json. Detector AP uses the fixed full-candidate manifest; production runtime uses confidence0.25/NMS0.7/max300.

The unchanged semantic student has 9,651,864 parameters (8,543,296 encoder and 1,108,568 head). The detector contributes another 68,162,238 parameters and is included in raw-video total latency, but excluded from the semantic parameter count. The rule bank adds 5,502 nonzero fixed effects stored in 71,550 dense coefficient slots; these are nontrainable constants. The readable full rule manifest occupies 1,040,434 bytes.

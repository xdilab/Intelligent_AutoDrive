# Fixed-split seed extension — COMP 765

Completed seven additional matched pairs (seeds 3–9), extending the original three to ten. The user requested the extension after seeing the original interval; the seven seeds and stopping rule were frozen before any new training/outcomes. All fourteen new runs use the same 64 training videos, 12 selection videos, 32-frame sequences, frozen ResNet50-layer3 + scene-GRU student, teacher targets, 30 epochs, optimizer, loss weights, initialization/order pairing, checkpoint selection, candidate/frame manifests and official evaluator. Only the seed changes. No wording, architecture, hyperparameter, data or checkpoint tuning used new AP.

## Triplet detector AP@0.5 (%)

| Cohort | Pairs | Supervised mean ± SD | Distilled mean ± SD | Paired change | 95% paired-t interval |
|---|---:|---:|---:|---:|---:|
| original_0_2 | 3 | 5.600 ± 0.689 | 6.586 ± 0.339 | +0.986 | [-0.672, +2.644] |
| extension_3_9 | 7 | 5.740 ± 0.195 | 6.372 ± 0.419 | +0.631 | [+0.162, +1.101] |
| combined_0_9 | 10 | 5.698 ± 0.368 | 6.436 ± 0.391 | +0.738 | [+0.346, +1.130] |

The combined nominal interval excludes zero, but this result-informed extension is exploratory and does not create independent test evidence. The interval measures seed variation conditional on this fixed dataset; it does not quantify generalization to new road videos. The original three-seed result remains unchanged. Original, new-only and combined cohorts are reported separately; the combined cohort overlaps the other two. All seeds are included regardless of outcome. There was no significance-dependent stopping and no confirmatory p-value claim.

## Secondary endpoints

| Endpoint | Supervised (10) | Distilled (10) | Paired change | 95% interval |
|---|---:|---:|---:|---:|
| tail47 | 2.604 ± 0.319 | 3.252 ± 0.475 | +0.649 | [+0.267, +1.030] |
| deep28 | 1.634 ± 0.280 | 2.427 ± 0.610 | +0.793 | [+0.309, +1.277] |
| common39 | 9.427 ± 0.475 | 10.273 ± 0.346 | +0.845 | [+0.390, +1.300] |
| action | 11.807 ± 0.428 | 13.560 ± 0.385 | +1.753 | [+1.415, +2.090] |
| loc | 14.483 ± 0.474 | 15.862 ± 0.505 | +1.379 | [+0.943, +1.815] |
| duplex | 10.346 ± 0.576 | 11.656 ± 0.407 | +1.310 | [+0.770, +1.849] |

Tail47 mean-class changes over ten seeds: 39 improve, 7 decline, 1 tie. Every class and seed is in [per-class.csv](per-class.csv) and [per-seed.csv](per-seed.csv); unsupported classes remain in the fixed 86-class mean. Frequency subsets remain training-derived. All evaluations use 36,717 scored frames, 198 videos and 6,163,968 candidates, with 38,663 causal state steps from video starts.

## Limits and cross-validation

The fixed teacher's semantic training/selection excludes these official validation videos, but the shared YOLO detector used all 198 for historical validation/checkpoint selection. Thus this remains a reused benchmark, not an independent system test. More seeds do not repair that exposure or expand training-video coverage. The earlier teacher numerical-parity caveat remains applicable. The original selected seed-0 inference bundle and runtime measurements remain unchanged; no best-of-ten export or new runtime claim is made.

Cross-validation was not launched. The user clarified that five folds are not required. The optional design memo in the code directory explains video-level grouping, inner checkpoint selection, teacher/detector exposure, and why fold scores must not be treated as independent training seeds.

## Records

Original study: `/home/brandon/.local/share/road-reason/comp765-cseb-student-20261021`. Extension: `/home/brandon/.local/share/road-reason/comp765-cseb-student-seeds-20261022`. Code/protocol: `/data/repos/ROAD_Reason/research/post-proposal/comp765-cseb-student-seeds-20261022`. [Paired-seed plot](paired-seeds.png), [interval plot](paired-intervals.png), [selection curves](selection-curves.png), [training curves](training-curves.png), and [reproducibility](reproducibility.json). The full original architecture/runtime report remains under the original study; this extension changes only the number of paired seeds. Student t critical values use df2/6/9, consistent with [NIST](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3672.htm).

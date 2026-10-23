# EfficientNet-B0 student — COMP 765 architecture follow-up

Completed six runs, three matched seeds0–2, started October23,2026. This is a separate architecture follow-up after the ResNet study, not a replacement of historical results or a thesis experiment. No final-AP tuning, adaptive seed selection, backbone fine-tuning or cross-validation.

## Controlled experiment

Full frozen ImageNet EfficientNet-B0 features at448×672, stride32,1280channels;3×3 RoIAlign mean, scene mean,256-unit GRU, geometry8, agent embedding16, MLP256 to184. Frozen encoder 4,007,548, trainable head 1,240,664, semantic total 5,248,212; shared YOLO adds 68,162,238. ResNet used stride16/1024channels and9,651,864 semantic parameters. The architecture comparison therefore changes pretrained features, spatial resolution, channels and input projection count together. Only the distillation loss differs within each B0 pair; there is no identical trainable initialization across different encoder dimensions.

The exact64train/12selection videos, all2,432 image steps, boxes/agents/confidence/labels/teacher targets and30epoch budget are preserved. New visual features come only from the same current images. All76 caches passed field-by-field supervision checks. Paired initialization, video order, AdamW0.001/0.01, clipping5, full32frame BPTT and minimum supervised development BCE selection match. Teacher targets use independent Bernoulli probabilities; supervised BCE versus supervised BCE+weight1 teacher BCE, outputs1–183. Slot0 excluded from loss; detector confidence applied exactly once. Unannotated frames update state with no loss. Teacher has current−3..current+4 at10Hz as privileged training supervision; student has no future input.

## Official detector AP@0.5 (%)

| Endpoint | B0 supervised mean ± SD | B0 distilled mean ± SD | Paired gain | 95% paired-t interval |
|---|---:|---:|---:|---:|
| triplet | 4.285 ± 0.246 | 4.638 ± 0.073 | +0.353 | [-0.366, +1.072] |
| tail47 | 1.633 ± 0.158 | 1.760 ± 0.039 | +0.128 | [-0.218, +0.473] |
| deep28 | 0.738 ± 0.084 | 1.079 ± 0.223 | +0.342 | [-0.141, +0.825] |
| common39 | 7.482 ± 0.518 | 8.107 ± 0.115 | +0.625 | [-0.886, +2.136] |
| action | 10.101 ± 0.160 | 10.732 ± 0.088 | +0.631 | [+0.105, +1.158] |
| loc | 12.021 ± 0.306 | 12.803 ± 0.035 | +0.781 | [+0.094, +1.468] |
| duplex | 8.422 ± 0.483 | 8.984 ± 0.166 | +0.562 | [-0.483, +1.607] |

The triplet interval includes zero; the sign is inconclusive under this interval. Intervals use paired t(df2), describe seed variation conditional on the fixed benchmark, and do not establish equivalence when zero is included. Tail47: 33 mean-class improvements, 13 declines, 1 ties. All outcomes and all classes are in [per-seed.csv](per-seed.csv) and [per-class.csv](per-class.csv).

Matched historical ResNet seeds0–2: supervised 5.600 ± 0.689, distilled 6.586 ± 0.339. B0 minus ResNet distilled mean is -1.948 points. These are the same three seeds and verified identical split/order/candidate/evaluator inputs; the completed ten-seed ResNet estimate is separate context, not substituted for this matched three-seed comparison. Architecture contrasts are exploratory. Historical fixed CSEB teacher triplet AP is 16.059; B0 distilled remains 11.421 points below it. Teacher has broader training coverage and future context; the original cross-environment teacher numerical-parity caveat remains.

All evaluations use36,717scored frames/198videos/6,163,968fixed candidates,38,663causal state steps from video starts,85/86supported triplets with all86 retained. Tail47/Deep28/Common39 retain fixed training-frequency membership. All current-image hashes match the original official evaluation inputs. This is detector AP, not historical pilot proposal AP.

## Measured efficiency and inference

Fresh processes on RTX A6000 GPU0, same raw MP4, same YOLO policy(imgsz1280/conf0.25/IoU0.7/max300), exact candidate parity;4warmup+32measured frames with CUDA synchronization at stage boundaries. No concurrent experiment on GPU0 during timing. Source resolution/candidate counts, load/decode/preprocessing/detector/encoder/head/CPU+JSONexport times, memory and parameters are in [runtime-summary.json](runtime-summary.json) and [stage CSV](runtime-stages-seconds.csv).

B0 median total 47.65ms, p95 50.08ms, throughput 20.92frames/s. ResNet fresh median 45.59ms, p95 51.37ms, throughput 21.57frames/s. Encoder+RoI median: B0 7.16ms versus ResNet 4.45ms. These are small, instrumented serial file-processing samples with uncontrolled filesystem-cache state, not sensor-to-decision deployment latency or a guaranteed real-time rate. Official AP uses lower-threshold historical full candidates, so timing and AP are different operating points.

The prespecified B0 distilled seed0 bundle processes 593 frames across3 complete training videos with zero reset at boundaries, finite outputs, stable post-warmup CPU/GPU allocation and forbidden external model/cache reads. Packaged and evaluated checkpoint replay is exact within1e-7; teacher and teacher caches are absent. Raw video is the external input. Existing realtime bundle is unchanged. No model is promoted based on final AP.

## Exposure, limits and records

All198official videos were used in shared YOLO historical validation/checkpoint selection; semantic teacher/student train/selection excludes them, but the benchmark is reused and not an independent full-system test. The selection pool belongs to teacher historical development. Three seeds,64training videos, frozen features and32frame training windows limit conclusions. More seeds do not repair data exposure. No deployment-safety claim.

Protocol/code: `/data/repos/ROAD_Reason/research/post-proposal/comp765-cseb-student-efficientnet-20261023`. Full checkpoints/cache/logs: `/home/brandon/.local/share/road-reason/comp765-cseb-student-efficientnet-20261023`. Bundle: `/home/brandon/.local/share/road-reason/comp765-cseb-student-efficientnet-20261023/bundle-seed0`. [Training curves](training-curves.png), [accuracy versus processing cost](accuracy-versus-cost.png), [accuracy versus parameter count](accuracy-versus-parameters.png), [reproducibility](reproducibility.json). Use the frozen protocol and existing environment. No thesis/slides changed or remote push performed.

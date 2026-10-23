# ResNet layer3 accuracy follow-up — COMP 765

Completed twelve runs: four conditions × paired seeds0–2. This post-defense follow-up asks whether partial ResNet fine-tuning improves the existing causal student. Historical studies/checkpoints and the working demo are preserved. The protocol was fixed before these new detector results.

## Architecture and controls

Same ResNet50 ImageNetV2 throughlayer3 at448×672, stride16/1024channels;3×3 RoIAlign mean, scene mean,256-unit GRU, geometry8, agentembedding16 and MLP256→184. Semantic parameters 9,651,864; head 1,108,568; layer3 7,098,368. Frozen arms train only the head; tuned arms additionally train layer3. All BN running statistics remain fixed, including during training; layer3 BN affine parameters train in tuned arms. Headlr1e-3, layer3lr1e-5, AdamW decay.01, gradientclip5. No architecture or data-size change. The deployed network has identical shape and parameter count in all arms.

All four arms start the same ImageNetV2 encoder and per-seed random head; no warmstart. Same64training/12selection videos,2,432current-image steps,32frame full-BPTT windows,30epochs and minimum supervised selection BCE (first tie). Original labels/teacher targets/candidates are read-only and hash-verified. Frozen prefix throughlayer2 is cached losslessly as bfloat16; layer3 uses32-frame batches with fixedBN and pooledhalf→float features. Fresh frozen controls use this same batched feature path. Frame-batching numerical effects are recorded in preflight; historical frozen AP is contextual only.

Supervised independent Bernoulli BCE versus BCE+weight1teacherBCE on outputs1–183; no softmax/temperature. Slot0 is masked from both losses. Detector confidence is applied exactly once; official agent scores come from YOLO. Unannotated frames update state without loss. Teacher targets use current−3..current+4 at10Hz, privileged training supervision; student inference uses only current/past frames. Teacher targets exist only for the intended training inputs.

## Official detector AP@0.5 (%)

| Condition | Triplet mean ± SD | Tail47 | Deep28 | Common39 |
|---|---:|---:|---:|---:|
| frozen-supervised | 5.586 ± 0.679 | 2.617 | 1.560 | 9.164 |
| frozen-distilled | 6.541 ± 0.288 | 3.278 | 2.571 | 10.474 |
| tuned-supervised | 5.729 ± 0.160 | 2.462 | 1.430 | 9.666 |
| tuned-distilled | 6.806 ± 0.032 | 3.201 | 2.152 | 11.150 |

| Triplet contrast | Mean paired delta | 95% paired-t interval |
|---|---:|---:|
| primary_tuning_distilled | +0.264 | [-0.467, +0.996] |
| tuning_supervised | +0.143 | [-1.423, +1.709] |
| kd_frozen | +0.955 | [-0.629, +2.539] |
| kd_tuned | +1.077 | [+0.626, +1.527] |
| interaction | +0.121 | [-1.107, +1.350] |

Primary: tuning within the distilled student gives +0.264 percentage points. Its95% paired-t(df2) interval includes zero; this interval leaves the sign inconclusive. The interval measures training-seed variation conditional on this fixed benchmark; it does not establish unseen-video generalization or equivalence. Secondary contrasts and endpoints are exploratory, with no multiplicity correction. Tail47 mean-class changes for the primary contrast: 22 improve, 24 decline, 1 tie. All184class outputs and all per-seed values are in the CSVs, including negative changes.

Historical verified CSEB teacher triplet mAP 16.059; tuned-distilled mean remains 9.253 points below it. Teacher uses broader training and future context; this is a descriptive gap, not a matched architecture-only comparison. Its prior cross-environment numerical caveat is retained.

Evaluation:198official videos,36,717scoredframes,6,163,968fixedcandidates;38,663causalsteps fromvideo starts, zero state at boundaries and continuous state through gaps.85/86triplets supported; all86 retained. Fixed training-derivedTail47/Deep28/Common39, no validation-frequency redefinition. Same metric/core, candidate and image hashes as original. These are detector metrics, not tiny-pilot proposal AP.

## Runtime and packaged model

Prespecified tuned-distilled seed0 versus fresh frozen-distilled seed0 on GPU0 RTX A6000, same rawMP4 and exact YOLO candidate parity. Fourwarmup+32measured frames, CUDA synchronization at stage boundaries; liveconf.25/imgsz1280/IoU.7/max300 differs from full-candidate AP. Tuned median 44.90ms, p95 46.65ms, 22.24frames/s; frozen median 45.02ms, p95 46.57ms, 22.22frames/s. Same architecture, so small timing differences must not be interpreted as a structural efficiency gain. Full loading/decode/preprocessing/detector/encoder/head/export, memory and parameter measurements are provided. Serial file-processing throughput is not sensor-to-decision latency or a deployment guarantee.

The tuned bundle contains the selected layer3 weights merged into the portable ResNet, its selected head, detector and standalone source. It runs from rawvideo with external model/cache access blocked; teacher and teacher caches are absent. Evaluator-to-bundle parity is checked within1e-7. Three complete training videos (593frames) pass causal indexing, finite outputs and bounded post-warmup memory tests. The old live bundle is unchanged; no automatic promotion based on AP.

## Limitations and reproducibility

Official validation was previously used by thesis research and sharedYOLO checkpoint selection. Semantic teacher/student training and selection excludes these198videos, but this is not an independent full-system test. Selection videos belong to teacher historical development. Three seeds,64training videos,32frame training windows and one fixed fine-tuning schedule limit conclusions; no post-result hyperparameter search or selective seed reporting was performed. CUDA RoIAlign backward can be nondeterministic; resume restores optimizer/all RNG and checks numerical tolerance, with observed error in preflight. No deployment-safety claim.

Code/protocol: `/data/repos/ROAD_Reason/research/post-proposal/comp765-cseb-student-layer3-20261023`. Full runs/logs/checkpoints: `/home/brandon/.local/share/road-reason/comp765-cseb-student-layer3-20261023`. Separate bundle: `/home/brandon/.local/share/road-reason/comp765-cseb-student-layer3-20261023/bundle-seed0`. [Per-seed](per-seed.csv), [contrasts](contrasts.csv), [per-class](per-class.csv), [curves](training-curves.png), [accuracy/cost](accuracy-versus-cost.png), [runtime](runtime-summary.json), [reproducibility](reproducibility.json). Temporary prefix caches can be regenerated from hashed sources. Historical artifacts, thesis and slides were not modified. No remote push requested.

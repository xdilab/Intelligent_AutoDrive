# Prespecified ResNet layer3 follow-up

Date: October 23, 2026. The machine-readable source of truth is [protocol.json](protocol.json), frozen and hashed before the first full training run. This is a post-defense COMP 765 follow-up to the teacher–student study, separate from thesis experiments and from the earlier phrase controls.

## Question and primary endpoint

Does fine-tuning the last retained ResNet stage improve the existing distilled causal student? Primary endpoint: the paired change in official detector triplet mAP, **tuned-distilled minus fresh frozen-distilled**, over all 86 classes at IoU 0.5.

Four conditions × seeds 0, 1, 2:

| Condition | Encoder training | Objective |
|---|---|---|
| frozen-supervised | All frozen | Ground-truth BCE |
| frozen-distilled | All frozen | Ground-truth BCE + teacher BCE |
| tuned-supervised | layer3 only | Ground-truth BCE |
| tuned-distilled | layer3 only | Ground-truth BCE + teacher BCE |

The final deployment architecture is identical. ResNet50 ImageNet V2 ends at layer3 (stride16, 1024 channels); layer4 is absent. The stem, layer1 and layer2 remain frozen. BatchNorm running statistics are fixed in every stage; layer3 affine parameters train in the tuned arms. The unchanged head uses RoI features, scene GRU256, geometry and agent embeddings, predicting 184 independent outputs. The head has 1,108,568 parameters; layer3 has 7,098,368; semantic encoder plus head has 9,651,864. The shared YOLO detector is additional and unchanged.

## Training and pairing

All four arms start ImageNet V2 and the exact original per-seed random head, with identical video orders. No warmstart from a previously selected student. Head AdamW learning rate1e-3, tuned layer3 rate1e-5; decay0.01; gradient clip5; constant learning rates;30epochs. One32-frame contiguous sequence per update,64training sequences per epoch; full32-step recurrent backpropagation. State starts at zero per training segment. Every unannotated frame still updates scene state, with empty actors and no loss.

The same64train/12selection videos, start frames, labels, boxes, agent IDs, confidences, and teacher targets are reused by hash. Teacher targets are fixed independent Bernoulli probabilities, with the candidate confidence removed once. Supervised BCE plus teacherBCE weight1 in distilled arms; outputs1–183, slot0 excluded. No temperature or softmax. Teacher sees current−3..current+4 at10Hz; that richer context is training supervision only.

The frozen prefix throughlayer2 is cached losslessly as bfloat16 in temporary shared memory. All32 independent frames pass through layer3 together, with fixed BN statistics. RoIAlign and pooling are float32; resulting features round tohalf thenfloat, preserving the existing head convention. Fresh frozen arms use this same batch32 feature path. Batch32 and frame1 kernels can differ numerically; historical AP is not substituted as the primary control.

Minimum supervised development BCE over30epochs selects the checkpoint; first tie wins. No final AP is inspected for selection or tuning. Checkpoints restore optimizer, Python/NumPy/Torch/CUDA RNG, epoch, order position and history. CUDA RoIAlign backward can be nondeterministic; preflight checks restored next-step parameters at atol1e-6/rtol1e-5 and records observed maximum difference.

## Evaluation and analysis

Same official198videos,36,717scoredframes,6,163,968fixed candidates, unchanged metric/core and IoU0.5. All38,663current-image steps from each video start are processed causally, with state reset only at video boundaries. Same label-string-derived composition mapping; obsolete child arrays are not used. Detector confidence is applied once. Official agent AP comes from YOLO class/confidence.

Report all three seeds, each condition mean and sampleSD, paired deltas, and95% paired-t(df2) intervals. Secondary contrasts: tuning within supervised, KD within frozen, KD within tuned, and interaction (KD effect tuned minus KD effect frozen). Report all standard groups, Tail47, Deep28, Common39, and every class, including declines. Frequency subsets retain authoritative training-derived membership; unsupported classes remain included with standard evaluator handling. Secondary tests are exploratory, without multiplicity correction. Lack of significance is not equivalence. Seed intervals do not measure uncertainty over new videos.

The official benchmark was previously used by thesis research and shared detector checkpoint selection; it is not an independent full-system test. Semantic teacher/student training and selection excludes these198videos. The selection pool belongs to the teacher's historical development pool. See [PROVENANCE-ADDENDUM.md](PROVENANCE-ADDENDUM.md).

## Runtime and release policy

Prespecified distilled seed0 in each trainability arm, same raw video, same hardware and detector policy, four warmup plus32measured frames. Report loading, decode, preprocessing, detector, encoder/RoI, head, export, total throughput and median/p95 latency, parameters and memory. The architecture is identical; timing variation is not a structural speedup. Live confidence0.25 differs from the full-candidate AP operating point. File-processing throughput is not sensor-to-decision latency.

Export tuned layer3 weights into the portable ResNet bundle and verify evaluator-to-runner score parity. Test three complete videos for causal source indices, video-boundary resets, finite output, stable memory and blocked external model/cache reads. No teacher dependency at inference. Do not replace the working demo bundle automatically.

## Resources and preservation

GPU0 RTX A6000 only. Approximately10.91GiB temporary prefix cache,50.7GiB temporary full-evaluation score buffers, and3–4GiB persistent checkpoints/backups/bundles. Silent monitor every120seconds checks actual logs, counters, disk, and checkpoint backups; it does not promise arbitrary autonomous repair. No unrelated jobs, original caches/checkpoints, thesis or slides are changed. Temporary owned caches may be removed only after verified final results and provenance capture.

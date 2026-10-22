# Teacher–Student Distillation for Efficient Road-Event Recognition

COMP 765, Brandon L. Byrd, NC A&T, Dr. Jung Hee Kim. Study started October 21, 2026 after the thesis defense. No thesis or defense slides were modified.

Distillation produced **0.986 percentage points higher mean official detector triplet AP** than the identical label-only student. The interval includes zero; the three-seed estimate is inconclusive about the sign of the effect. The mean distilled student remains 9.473 points below the fixed Contextual Scene-Attention Expert Blend (CSEB) teacher. This gap is descriptive: the teacher has much larger training coverage and a future temporal window. The teacher AP is a historical result with matching checkpoint, frame/candidate manifests and evaluator source hashes. Exact numerical replay across environments is not established: fixed detector-only agentness differs by 0.000626 AP points and agent mAP by 0.000891 points. All new student comparisons use one environment/evaluator; the teacher gap is historical context, not a fresh matched teacher run.

## Controlled comparison

Frozen shared ImageNet ResNet50 through layer 3, 3×3 RoIAlign, a 256-unit scene GRU, geometry and agent embeddings, and 184 outputs. There are 9,651,864 semantic parameters, including 1,108,568 trainable parameters; YOLO is additional and was previously trained on 600 videos. New semantic-head training uses 64 videos and selection uses 12, with 32 contiguous frames per video, 30 epochs, and paired seeds 0/1/2. AdamW uses learning rate 0.001, weight decay 0.01, and gradient clipping at 5. Only the distillation weight changes from 0 to 1. Both losses use independent Bernoulli BCE over 183 semantic outputs; slot 0 is excluded. Teacher targets are detached, confidence removed, and generated only for annotated training inputs. Unannotated frames update state without loss. The complete data audit counts 2036 annotated training frames and 375 annotated selection frames. Ground-truth-positive training proposals cover 68/86 triplet classes; soft teacher probabilities are not counted as observed class coverage.

The speed reduction comes from the smaller student architecture. Distillation changes its training objective, not its inference cost; the matched comparison tests whether it improves accuracy at that fixed cost.

Final evaluation uses 36,717 scored frames from 198 official validation videos, 6,163,968 fixed detector candidates, and IoU 0.5. The student processes 38,663 current/past frames from video starts and resets only at video boundaries. Selection is minimum supervised development BCE. Official validation is disjoint from teacher/student semantic-head training and selection. However, all 198 videos and all 36,717 scored frames were used in the shared YOLO detector validation/checkpoint-selection pool, and the benchmark had prior thesis research use. The student selection pool belongs to teacher historical development. This is a reused validation benchmark, not a pristine independent test of the complete teacher/student systems. The pre-evaluation provenance addendum corrects the original protocol’s broader wording without changing the experiment.

## Official detector results

All values are AP percentage points; SD is sample SD over three seeds. The interval uses paired t(df = 2), not independent runs.

| Endpoint | Supervised mean ± SD | Distilled mean ± SD | Paired Δ |95% paired-t interval| Fixed teacher |
|---|---:|---:|---:|---:|---:|
|triplet|5.600 ± 0.689|6.586 ± 0.339|+0.986|[-0.672, +2.644]|16.059|
|tail47|2.639 ± 0.598|3.351 ± 0.517|+0.711|[-0.929, +2.351]|9.733|
|deep28|1.579 ± 0.493|2.646 ± 0.533|+1.067|[-1.098, +3.232]|7.000|
|common39|9.168 ± 0.822|10.485 ± 0.180|+1.317|[-0.410, +3.044]|23.683|
|action|11.607 ± 0.677|13.711 ± 0.333|+2.104|[+0.960, +3.248]|24.567|
|loc|14.258 ± 0.880|15.988 ± 0.533|+1.729|[-0.232, +3.691]|29.425|
|duplex|10.085 ± 0.948|11.721 ± 0.244|+1.636|[-0.726, +3.998]|24.789|

Individual seeds and every class are in [per-seed.csv](per-seed.csv) and [per-class.csv](per-class.csv). Tail47: 38 improve, 8 decline, 1 tie on mean AP across seeds. All 86: 72 improve, 13 decline, 1 tie; 85/86 classes have final GT support. Fixed frequency membership comes from 600 training videos, never validation counts.

## Runtime and temporal behavior

Same RTX A6000 GPU 0 while eight CPU workers computed detector AP; the GPU was otherwise idle. CPU contention was not eliminated, so these are workstation-session timings. Raw video, YOLO image size 1280, confidence threshold 0.25, NMS IoU 0.7, and maximum 300 detections; 4 warmup and 32 measured frames. Median serial processing: student 45.84 ms, teacher 5392.68 ms. Student p95: 51.76 ms; teacher p95: 5840.63 ms. Processing throughput: student 21.41 frames/s, teacher 0.19 frames/s. Median processing ratio 117.65×. Teacher also requires 0.4 s of future lookahead at 10 Hz; student requires none.

Stage-level load/decode/preprocess/detector/encoding/head/export times, GPU memory, resolutions and candidates are in [runtime-summary.json](runtime-summary.json), with a compact [stage timing CSV](runtime-stages-seconds.csv). Instantiated inference-model parameter totals including YOLO are 77,814,102 for the student and 1,103,373,104 for the teacher. See [accuracy versus parameter cost](accuracy-versus-parameters.png); this comparison avoids interpreting the timing and AP measurements as the same candidate operating point. The three complete-video isolated runner tests process 593 frames. Source-index checks forbid future reads; output scores are finite; checkpoint/evaluator/bundle replay is verified. Full GPU-allocation and CPU-RSS traces are retained, with post-warmup bounds checked. The longest tested source clip spans about 19.8 seconds; longer streaming duration remains unmeasured. File-processing throughput is not end-to-end live sensor latency; capture, display, scheduling and deployment behavior are unmeasured. Runtime uses confidence 0.25 while detector AP uses the unchanged historical full candidate set, so these are complementary measurements, not the same operating point.

## Historical context and limits

The original 24-frame pilot: 11.59% supervised versus 24.12% distilled proposal AP over 17 supported triplets. The completed 96-frame expansion: 17.67% versus 21.03% proposal AP over 45 classes. Neither is comparable numerically to this official detector endpoint, nor an independent teacher test.

The paired-seed interval describes training-seed variation on this fixed benchmark, not uncertainty over new videos. Only three seeds and 64 training videos; a frozen encoder and scene-only memory without per-object tracking; training 32-frame segments shorter than full streams; teacher provides richer future-context supervision. Absence of significance is not equivalence. No tuning on final AP, seed cherry-picking, architecture changes, or deployment-safety claims. No backbone-finetuning or other follow-up was mixed into the main comparison. More training coverage or fine-tuning would be separate future studies.

## Artifacts

- Protocol/code: `/data/repos/ROAD_Reason/research/post-proposal/comp765-cseb-student-20261021/`.
- Run root: `/home/brandon/.local/share/road-reason/comp765-cseb-student-20261021`; immutable split/freeze manifests, cache hashes, optimizer/RNG checkpoints and verified backups.
- Reproducibility: [reproducibility.json](reproducibility.json); training plots [PNG](training-curves.png)/[PDF](training-curves.pdf); accuracy/cost [PNG](accuracy-versus-cost.png)/[PDF](accuracy-versus-cost.pdf).
- Separate prespecified seed 0 bundle: `/home/brandon/.local/share/road-reason/comp765-cseb-student-20261021/bundle-seed0/`; existing realtime-v1 unchanged. Run `python run.py --bundle BUNDLE --video INPUT.mp4 --output NEW_OUTPUT --verify-isolation` with the documented environment.

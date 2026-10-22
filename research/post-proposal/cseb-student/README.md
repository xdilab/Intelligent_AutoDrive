# CSEB causal student

**[Run real-time inference](REALTIME.md)** · **[Expanded-data study](EXPANSION.md)** · **[COMP 765 controlled study](../comp765-cseb-student-20261021/README.md)**

The distilled pilot checkpoint is the current real-time v1 model. The original pilot protocol and measured results follow.

Authorized October 19, 2026. A small first experiment tests whether the fixed CSEB seed-0 blend can teach a much cheaper causal student. This borrows temporal reuse as a design principle from StreamingVLM (https://arxiv.org/html/2510.09608v2); it does not implement that paper's Qwen KV-cache method.

## Locked pilot

- Student: ImageNet ResNet50 V2 through layer3 (frozen), one full-frame 448x672 feature map shared across proposals; RoIAlign spatial mean, box geometry, detected-agent embedding, and a trainable 256-dimensional scene GRU plus 184-output head. No tracking or per-object temporal memory in this first pilot. Student sees only current/past frames; CSEB teacher sees its original centered eight-frame window.
- Inputs: original ROAD-Waymo JPEGs aligned with annotations. Fresh YOLO proposals, confidence0.25, size1280, IoU0.7, max300; unchanged teacher. No demo-validation training.
- Selection: first8 expert-training and first6 development videos under fixed SHA256 ranking. Development is divided into3 selection and3 held-out videos. Earliest fully annotated eight-frame window starting at or after frame40 per video; no prediction-based selection. Official validation and gate videos are untouched.
- Two identically initialized conditions: supervised BCE and supervised BCE plus weight1 Bernoulli score distillation. Teacher soft targets are the blend probabilities before detector confidence; detector confidence is applied once at evaluation. No softmax over multi-label outputs. Output0 is supplied by YOLO and excluded from head loss.
- Train only the new head for30 epochs, one sequence per update, AdamW lr0.001/decay0.01, seed0. Select each condition using minimum supervised BCE on selection videos. Held-out scores are computed only for selected checkpoints. Include a constant training-teacher-mean control.
- Match each proposal to a same-agent annotation at IoU>=0.5; remap annotation IDs through label strings. Never use child-composition arrays. Report proposal AP over positive-support classes with class coverage, teacher agreement and error. This is NOT official detector AP, full tail47 AP or a model promotion decision.
- Measure paired fresh teacher/student GPU pipeline latency, including YOLO and CPU score export, on12 held-out frames. Loading/disk decoding excluded and stated. A larger study needs object-size coverage, longer temporal windows, end-to-end backbone tuning, multiple seeds and official detector AP.

## Execution

Use the existing road_reason Python environment and demo runtime package path. Commands from this directory:

```bash
export PYTHONPATH=/home/brandon/.local/share/road-reason/cseb-demo-runtime/packages
export CUDA_VISIBLE_DEVICES=0
python -m unittest test_student.py
python pilot.py prepare --root /home/brandon/.local/share/road-reason/cseb-student-20261019
python pilot.py train --root /home/brandon/.local/share/road-reason/cseb-student-20261019
python pilot.py benchmark --root /home/brandon/.local/share/road-reason/cseb-student-20261019
```

Weights/caches/checkpoints remain outside Git. Atomic progress records and per-epoch checkpoints retain optimizer/RNG state. The bounded local pilot is supervised during this session; no cluster submission or unattended training is implied. Preparation estimate5–8min from~2.7s per teacher frame; training/evaluation provisional2–5min, refreshed from actual logs. Each phase uses a process timeout. Checkpoint replay/finiteness/hash receipts are written before completion.

## Student inference

`inference.StudentInference` loads the selected pilot checkpoint, verifies its manifest/model/backbone hashes, and supports `predict_frame(rgb_image)` with persistent scene state. Call `reset()` at video boundaries. Its feature rounding matches the frozen-feature training cache. The standalone `inference.py --run RUN --video VIDEO --output NEW_DIRECTORY --frames 120` command samples at10Hz and emits184 scores per proposal with detector confidence applied once.

The student-held-out videos are excluded from student training and checkpoint selection, but belong to the teacher's historical development pool. They are not a new independent teacher test set. The recurrent state is trained on short segments only; continuous long-video behavior needs separate evaluation before use.

## Executed pilot result — October 19

Completed 112 frames / 2,842 proposals across 14 videos:1,984 training proposals,308 selection proposals and550 held-out proposals. Both 30-epoch head runs completed in 7.56s after 302.74s target/feature preparation. Selected epochs: supervised 7, distilled 26, using selection BCE only.

Held-out **proposal mAP (%)**, on 24 frames from 3 student-held-out development videos:

| Model | Action | Location | Duplex | Triplet |
|---|---:|---:|---:|---:|
| Supervised student | 17.13 | 24.05 | 17.62 | 11.59 |
| Distilled student | 30.85 | 41.16 | 28.73 | 24.12 |
| CSEB teacher | 65.70 | 55.22 | 62.27 | 57.88 |
| Training-teacher mean | 19.66 | 16.44 | 15.22 | 12.07 |

Positive-support coverage: 11/22 action, 9/16 location, 17/49 duplex and 17/86 triplet classes. This is not official detector AP; unobserved classes and full tail47/deep28 metrics are unmeasured. The distilled student's gain supports a larger pilot, but it retains a substantial accuracy gap to the teacher. Single seed and short sequences prevent a general performance claim.

Paired A6000 pipeline median: student 38.41ms versus teacher 1.685s, 43.9x ratio of medians. Twelve frames measured; first frame of each of 3 videos excluded as warmup. Includes YOLO and score export; excludes model loading and disk decoding. These medians must not be compared directly with the earlier 120-frame demo's wall-clock throughput. Semantic student parameters: 8,543,296 frozen backbone + 1,108,568 trainable head; the common YOLO detector is additional.

Five focused tests passed. Fresh 8-frame student inference replay passed with maximum score difference 0; boxes and detector confidence matched exactly. Selected checkpoints have finite-weight/hash receipts and verified backups under the run directory. All bounded processes completed; no new cluster/unattended jobs remain.

Run directory: `/home/brandon/.local/share/road-reason/cseb-student-20261019/`. Machine-readable comparison: [pilot-results.json](pilot-results.json). Wiki evidence: `wiki/artifacts/cseb-student-20261019/`.

Next experiment should expand training videos and temporal duration, fine-tune the shared visual backbone, and retain the supervised control before attempting full detector/tail AP. This pilot does not validate continuous long-stream performance or reproduce StreamingVLM.

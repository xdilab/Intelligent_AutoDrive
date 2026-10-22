# Teacher–Student Distillation for Efficient Road-Event Recognition

Completed October 21, 2026: all six 30-epoch runs, all six official detector evaluations, raw-video runtime/causality checks, and the independent final audit passed. [Full findings](results/FINDINGS.md), [per-seed metrics](results/per-seed.csv), [all per-class metrics](results/per-class.csv), [training curves](results/training-curves.png), and [reproducibility record](results/reproducibility.json).

| Official AP@0.5 (%) | Supervised mean ± sample SD | Distilled mean ± sample SD | Paired change |
|---|---:|---:|---:|
| Triplet | 5.600 ± 0.689 | 6.586 ± 0.339 | +0.986 |
| Tail47 | 2.639 ± 0.598 | 3.351 ± 0.517 | +0.711 |
| Deep28 | 1.579 ± 0.493 | 2.646 ± 0.533 | +1.067 |
| Common39 | 9.168 ± 0.822 | 10.485 ± 0.180 | +1.317 |

The primary paired 95% t interval is **[-0.672, +2.644] points**: positive in all three observed pairs, but inconclusive with only three seeds. The fixed historical teacher is 16.059% triplet AP. This is reused official validation, including prior shared-YOLO checkpoint selection, not an independent system test.

Raw-video processing on RTX A6000: **45.84 ms median / 51.76 ms p95, 21.41 frames/s**, including YOLO/decode/export. Matched teacher median: 5392.68 ms, plus 0.4 s future context. These are serial file-processing timings with 32 measured frames and 56–67 candidates per frame, not live sensor latency. All 593 frames across three complete clips passed isolated inference, causal indexing and bounded-memory checks; evaluated and packaged scores agree exactly. Runtime uses confidence 0.25; AP uses historical full candidates.

The new bundle is `/home/brandon/.local/share/road-reason/comp765-cseb-student-20261021/bundle-seed0/`. It contains only the student, frozen ResNet, YOLO and standalone code; teacher hashes in its manifest are provenance, not inference dependencies. The existing realtime-v1 bundle remains unchanged. Compact evidence is in `results/records/`; weights and caches remain under the run root. The 25.35 GiB temporary prediction buffers were hashed and freed after successful evaluation/audit. The completed study's monitor timer is disabled after final backup verification; other experiment monitors remain unchanged.

Run the standalone bundle with the existing Python environment:

```bash
/home/brandon/miniconda3/envs/road_reason/bin/python \
  /home/brandon/.local/share/road-reason/comp765-cseb-student-20261021/bundle-seed0/run.py \
  --bundle /home/brandon/.local/share/road-reason/comp765-cseb-student-20261021/bundle-seed0 \
  --video INPUT.mp4 --output NEW_OUTPUT_DIRECTORY --verify-isolation
```

The commands below resume/verify the recorded server study and skip completed artifacts. Frozen manifests and code hashes bind this run; preserve this directory and checkpoints when making a separately versioned replication. `final_audit.py` independently checks completed results, and `archive_record.py` copies compact evidence into the repository without uploading it.

New COMP 765 study started October21,2026, after the thesis defense. The fixed thesis Contextual Scene-Attention Expert Blend (CSEB) checkpoint supplies training targets; the causal student and its matched supervised control are the study. This directory does not replace the working realtime-v1 bundle or historical pilot/expanded results.

Scientific protocol: [protocol.json](protocol.json), with the mandatory [pre-evaluation provenance correction](PROVENANCE-ADDENDUM.md). Target mapping and output/state conventions: [TARGETS.md](TARGETS.md). Student implementation is reused unchanged from `../cseb-student/model.py`. Student: frozen shared ResNet50 through layer3, RoI features, geometry and agent embedding, 256-unit scene GRU,184 outputs. Trainable parameters:1,108,568; frozen semantic encoder:8,543,296; detector additional.

Outputs: `/home/brandon/.local/share/road-reason/comp765-cseb-student-20261021/`. Exact video lists and overlap audit: `split-manifest.json`, `audit.json`. Three paired seeds0/1/2,64train and12selection videos,32 contiguous frames each; six30-epoch AdamW runs. Teacher targets only on annotated training frames; unannotated frames only update state. Final official validation is disjoint from semantic-head training/selection, but all 198 videos were used in shared-detector validation/checkpoint selection and prior thesis research. It is a reused benchmark, not an independent system test. Inference processes complete prefixes from frame1, resets only at video boundaries, and never reads a future student image.

Preparation reuses verified8-frame caches where possible. Only training receives teacher probabilities. `prepare.py` writes atomic per-video caches; `train.py` saves optimizer and all RNG state every16sequences and everyepoch. The matched per-seed data orders and initialization hashes are saved. Selection minimizes development supervised BCE; no final AP is used for tuning.

Run with the existing environment; do not install or upgrade shared packages:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/home/brandon/.local/share/road-reason/cseb-demo-runtime/packages
cd /data/repos/ROAD_Reason/research/post-proposal/comp765-cseb-student-20261021
/home/brandon/miniconda3/envs/road_reason/bin/python preflight.py
/home/brandon/miniconda3/envs/road_reason/bin/python pipeline.py prepare train
# Official evaluation/runtime stage (replay checks precede AP):
/home/brandon/miniconda3/envs/road_reason/bin/python pipeline.py evaluation_preflight evaluate runtime_audit analyze
```

Existing expanded study completed:96frames/12student-heldout development videos,45supported triplets; supervised17.67%, distilled21.03% proposalAP, seed0. These are historical proposal metrics, not official detector AP. The earlier24-frame pilot reported11.59%/24.12% over17classes. Its teacher-development overlap prevents an independent-teacher-test interpretation.

Persistent silent monitor: `research-comp765-student-monitor.timer`; service: `research-comp765-student.service`. It checks actual logs/progress, service status, disk space and checkpoint readability/hash-matched backups every2minutes. Desktop notifications remain disabled. It does not promise automatic conversational wakeup or arbitrary unattended repair.

The final standalone export includes `run.py` and the unchanged `model.py`, with all asset hashes. `--verify-isolation` blocks model/cache files outside the bundle. Complete-video tests bound retained GPU-allocation variation after32warmup frames to128MiB and CPU-RSS variation to256MiB; detailed traces remain available.

The commands above also support foreground reproduction. For persistent monitored runs, use the supplied systemd service; the watchdog checks that service identity. No remote publication is performed automatically.

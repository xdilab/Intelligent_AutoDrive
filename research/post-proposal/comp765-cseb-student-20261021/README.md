# Teacher–Student Distillation for Efficient Road-Event Recognition

New COMP 765 study started October21,2026, after the thesis defense. The fixed thesis CSEB checkpoint supplies training targets; the causal student and its matched supervised control are the study. This directory does not replace the working realtime-v1 bundle or historical pilot/expanded results.

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

The final standalone export includes `run.py` and the unchanged `model.py`, with all asset hashes. `--verify-isolation` blocks model/cache files outside the bundle. Complete-video tests bound retained GPU-allocation variation after32warmup frames to128MiB; detailed traces remain available.

The commands above also support foreground reproduction. For persistent monitored runs, use the supplied systemd service; the watchdog checks that service identity. No remote publication is performed automatically.

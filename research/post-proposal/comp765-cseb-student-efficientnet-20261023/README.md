# COMP 765 EfficientNet-B0 student follow-up

## Verified outcome — October 23, 2026

EfficientNet-B0 completed all three paired seeds and full official detector evaluation. It is smaller but less accurate, and it was slightly slower in the measured RTX A6000 raw-video sample. Keep the existing ResNet student as the stronger tested option on this server; the separate B0 bundle remains available for reproducibility.

| Matched seeds 0–2 | ResNet50 student | EfficientNet-B0 student |
|---|---:|---:|
| Supervised triplet mAP (mean ± sample SD) | 5.600 ± 0.689% | 4.285 ± 0.246% |
| Distilled triplet mAP (mean ± sample SD) | 6.586 ± 0.339% | 4.638 ± 0.073% |
| Semantic encoder + head parameters | 9,651,864 | 5,248,212 |
| Median serial raw-video processing | 45.59 ms/frame | 47.65 ms/frame |
| Encoder + RoI median | 4.45 ms/frame | 7.16 ms/frame |

B0 distillation gains: +0.678, +0.258 and +0.123 percentage points; mean +0.353, nominal 95% paired-t interval [−0.366,+1.072]. The interval includes zero and is inconclusive, not evidence of equivalence. These are three matched seeds on a fixed reused benchmark; the ten-seed ResNet extension is separate context. YOLO adds 68,162,238 parameters to both systems. Timing uses 4 warmup + 32 measured frames at confidence0.25; official AP uses historical full candidates. This is not a deployment-latency guarantee.

[Findings](results/FINDINGS.md) · [Every seed](results/per-seed.csv) · [Every class](results/per-class.csv) · [Accuracy/cost plot](results/accuracy-versus-cost.png) · [Independent audit](results/records/final-audit.json)

The prespecified seed0 bundle is `/home/brandon/.local/share/road-reason/comp765-cseb-student-efficientnet-20261023/bundle-seed0/`. It passed cached/fresh/evaluator replay with zero score difference and raw-video isolation/stable-memory checks over593frames across3videos. “Zero difference” means the same checkpoint agrees across implementation paths, not equal accuracy across architectures.

```bash
$PY /home/brandon/.local/share/road-reason/comp765-cseb-student-efficientnet-20261023/bundle-seed0/run.py \
  --bundle /home/brandon/.local/share/road-reason/comp765-cseb-student-efficientnet-20261023/bundle-seed0 \
  --video INPUT.mp4 --output NEW_OUTPUT_DIRECTORY --verify-isolation
```

Final arithmetic, source hashes, all38,663image identities, paired initialization/order and12checkpoint backups passed. Original studies and realtime bundle are unchanged. Temporary prediction arrays are hashed then released after verification; inputs/checkpoints support regeneration. No remote push.

Started October23,2026. Separate architecture follow-up to the completed ResNet student study. Frozen full EfficientNet-B0 features,1280 channels/stride32, shared448x672 input; same sceneGRU and semantic head widths, with input projections changed to1280. Same64train/12selection/198official videos,32-frame training sequences,30epochs, supervised BCE versus BCE+teacher BCE, seeds0–2. All three pairs will complete regardless of outcome.

Only current/past images enter the student. Reuse exact original candidate/label/teacher tensors, regenerate only small student feature caches. Full official detector AP@0.5, all86triplets and fixedTail47/Deep28/Common39; no crop-only substitution. Teacher/detector exposure and prior benchmark reuse remain disclosed in [PROVENANCE-ADDENDUM.md](PROVENANCE-ADDENDUM.md). Architecture changes include feature resolution, pretrained weights, channels and projection parameter counts; speed is measured, not inferred from parameter counts.

Protocol: [protocol.json](protocol.json). Run root: `/home/brandon/.local/share/road-reason/comp765-cseb-student-efficientnet-20261023/`. Existing studies and realtime bundle are preserved. No CV, slides or automatic remote push.

Use the existing road_reason environment with PYTHONPATH=/home/brandon/.local/share/road-reason/cseb-demo-runtime/packages, CUDA_VISIBLE_DEVICES=0 and PYTHONDONTWRITEBYTECODE=1. Commands and measured launch ETA are recorded with the run.

## Reproduction commands

The original run root contains immutable signatures and resumable checkpoints; rerunning there resumes verified work. A new independent rerun must use a separate root and regenerate its path-specific freeze/provenance record. Preserve original datasets/results.

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/home/brandon/.local/share/road-reason/cseb-demo-runtime/packages
cd /data/repos/ROAD_Reason/research/post-proposal/comp765-cseb-student-efficientnet-20261023
PY=/home/brandon/miniconda3/envs/road_reason/bin/python
$PY initialize.py
$PY preflight.py
$PY initialize.py --freeze
$PY pipeline.py prepare train evaluation_preflight
$PY pipeline.py runtime_audit evaluation_benchmark
$PY pipeline.py evaluate analyze
# Verify checkpoints/backups before the independent final audit.
$PY monitor.py
$PY verify_assets.py
$PY final_audit.py
$PY archive_record.py
```

The six-head full-candidate benchmark uses three complete official video prefixes and no AP. Inference resumes by completed video, AP by completed class, and training by optimizer/RNG checkpoint. The silent `research-comp765-student-efficientnet-monitor.timer` checks every two minutes while the study runs. Desktop banners remain disabled. The study timer was disabled after verified completion; other monitors remain unchanged.

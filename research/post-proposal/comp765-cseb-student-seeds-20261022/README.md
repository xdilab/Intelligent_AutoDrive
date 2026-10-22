# COMP 765 student distillation — fixed ten-seed extension

## Completed results — October 22, 2026

All seven added pairs completed training and official detector evaluation. Independent arithmetic, checkpoint, input-hash and original-artifact audits passed. All 28 new best/last checkpoints have verified backups.

| Cohort | Supervised triplet AP ± SD | Distilled triplet AP ± SD | Paired change | Nominal 95% paired interval |
|---|---:|---:|---:|---:|
| Original 3 pairs | 5.600 ± 0.689 | 6.586 ± 0.339 | +0.986 | [-0.672, +2.644] |
| New 7 pairs | 5.740 ± 0.195 | 6.372 ± 0.419 | +0.631 | [+0.162, +1.101] |
| Combined 10 pairs | 5.698 ± 0.368 | 6.436 ± 0.391 | +0.738 | [+0.346, +1.130] |

Values are detector AP@0.5 percentages; changes and intervals are percentage points. Distillation improved triplet AP in 9 of 10 pairs. Combined Tail47 AP is 2.604 → 3.252, with 39 tail classes improving, 7 declining and 1 tying in their mean AP. The combined interval is exploratory because the extension followed inspection of the original outcomes. It describes seed variation on the reused fixed benchmark, not uncertainty over unseen videos.

[Full findings](results/FINDINGS.md) · [Every seed](results/per-seed.csv) · [Every class](results/per-class.csv) · [Paired intervals](results/paired-intervals.png) · [Reproducibility](results/reproducibility.json) · [Final audit](results/records/final-audit.json)

Cross-validation remains optional and unlaunched; five folds are not required. The original inference bundle and runtime measurements are unchanged. Full prediction buffers are temporary and are released after verification; their hashes and inference receipts are archived so evaluation can be regenerated from the preserved checkpoints and inputs.

Authorized October22,2026 after the original three-seed result. Preserve the original study at `../comp765-cseb-student-20261021/`. Train seven new pairs, seeds3–9, and report original0–2, new-only3–9, and combined0–9 separately. The sample size is fixed at ten regardless of results; the combined analysis is exploratory because the extension was chosen after the original outcomes. No stopping when significance appears.

Architecture/data/objective/evaluator remain unchanged: frozen shared ResNet50 through layer3, RoI features, geometry/agent embeddings,256-unit scene GRU;64 training/12 selection videos;32-frame sequences;30 epochs; BCE versus BCE+teacher-probability BCE. Same cached data, paired initialization/order and minimum-supervised-development-BCE checkpoint rule. Reuse the original selected seed0 inference bundle and runtime evidence; no best-of-ten promotion.

Protocol: [protocol.json](protocol.json). Exposure caveat: [PROVENANCE-ADDENDUM.md](PROVENANCE-ADDENDUM.md). Optional cross-validation: [design memo](CROSS-VALIDATION-DESIGN.md), with no required fold count and no CV launch.

Run root: `/home/brandon/.local/share/road-reason/comp765-cseb-student-seeds-20261022/`. All original caches are read by hash; original results/checkpoints are immutable. The extension has independent checkpoints, backups, frozen manifests, logs and results. Official evaluation retains36,717 scored frames/198videos/6,163,968 candidates; fixed Tail47/Deep28/Common39 subsets. The shared detector previously used these videos for validation selection, so this is a reused benchmark, not an independent system test.

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/home/brandon/.local/share/road-reason/cseb-demo-runtime/packages
cd /data/repos/ROAD_Reason/research/post-proposal/comp765-cseb-student-seeds-20261022
/home/brandon/miniconda3/envs/road_reason/bin/python extension_preflight.py
/home/brandon/miniconda3/envs/road_reason/bin/python pipeline.py train evaluate analyze
```

Scientific protocol and numerical code are hashed before the new runs. Preflight checks exact training/scoring function parity, cached-to-fresh checkpoint replay, all new seed initializations, optimizer/RNG resume, causal state behavior, confidence application, full detector-agentness parity and AP fixtures. It benchmarks one64-sequence epoch per condition and fourteen heads on three complete official video prefixes; random-head benchmark output is not scientific AP.

Completed service: `research-comp765-student-seeds.service`. The silent checkpoint/progress monitor `research-comp765-student-seeds-monitor.timer` ran every two minutes and was disabled after successful completion and final backup verification. No desktop banners or promise of unattended conversational repair. ETA and resource assumptions are saved before launch. Outputs include separate cohort tables, every seed/class, paired intervals and plots, and provenance. No remote publication occurs automatically.

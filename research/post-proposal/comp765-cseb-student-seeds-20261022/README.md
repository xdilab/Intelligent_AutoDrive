# COMP 765 student distillation — fixed ten-seed extension

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

Persistent run: `research-comp765-student-seeds.service`. Silent checkpoint/progress monitor: `research-comp765-student-seeds-monitor.timer`, every two minutes. No desktop banners or promise of unattended conversational repair. ETA and resource assumptions are saved before launch. Outputs include separate cohort tables, every seed/class, paired intervals and plots, and provenance. No remote publication occurs automatically.

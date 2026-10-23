# ResNet layer3 accuracy follow-up — COMP 765, 2026-10-23

Completed: twelve trainings and twelve official detector evaluations; all final audits passed. See [full findings](results/FINDINGS.md), [per-seed results](results/per-seed.csv), [all per-class results](results/per-class.csv), [paired contrasts](results/contrasts.csv), [runtime measurements](results/runtime-summary.json), and [reproducibility](results/reproducibility.json).

| Three-seed mean detector AP (%) | Frozen distilled | Layer3-tuned distilled |
|---|---:|---:|
| Triplet | 6.541 | 6.806 |
| Tail47 | 3.278 | 3.201 |
| Deep28 | 2.571 | 2.152 |
| Common39 | 10.474 | 11.150 |

Primary paired gain: **+0.264 percentage points**,95% paired-t interval **[-0.467, +0.996]**, with only three seeds. Two seeds improve and one declines. Common39 improves, while mean Tail47 and Deep28 decrease;22tail classes improve,24decline and1ties. This is not evidence of a general accuracy upgrade or equivalence. The benchmark is reused and not an independent full-system test.

Distillation within the tuned student improves the mean from5.729% to6.806%; this secondary comparison does not establish that fine-tuning strengthens the effect of distillation (the interaction interval includes zero). The tuned model has the same9.65million semantic parameters and measured~45ms serial frame-processing latency as the frozen model. Timing is a32-frame sample at liveconf0.25, distinct from full-candidate AP; it is not a deployment-latency guarantee.

The tuned seed0 bundle is retained as an experimental variant at`/home/brandon/.local/share/road-reason/comp765-cseb-student-layer3-20261023/bundle-seed0`; the working demo was not replaced. [Training curves](results/training-curves.png) and [accuracy versus cost](results/accuracy-versus-cost.png) are included; the small horizontal timing difference is measurement variation, not a structural speedup.


A new accuracy experiment within the post-defense teacher–student study. The original thesis, prior student studies, and working inference bundle are preserved.

Four conditions, seeds 0–2: frozen/supervised, frozen/distilled, layer3-tuned/supervised, layer3-tuned/distilled. Primary comparison: tuned-distilled minus fresh frozen-distilled official detector triplet mAP. All start ImageNet V2 and identically seeded random heads. Same64 train /12 selection videos,32 contiguousframes,30epochs, labels/teacher/candidates and minimum development BCE selection. Headlr1e-3; tuned layer3lr1e-5; BN running statistics fixed. No wording, prototype, architecture or data-size intervention.

Protocol: `protocol.json`; exposure caveats: `PROVENANCE-ADDENDUM.md`. Official198video evaluation is a previously used validation benchmark, including shared detector selection; it is not an independent full-system test.

Runtime root: `/home/brandon/.local/share/road-reason/comp765-cseb-student-layer3-20261023`. Temporary frozen-prefix cache: `/dev/shm/comp765-cseb-student-layer3-20261023-prefix` (10.91GiB), regenerable from hashed original images. All prefix tensors may be preloaded toGPU. Frozen training features are recomputed in the same32-frame batch path as tuned training, motivating fresh controls. Inference remains single-current-frame causal; no teacher assets are needed.

Use `/home/brandon/miniconda3/envs/road_reason/bin/python`, `CUDA_VISIBLE_DEVICES=0`, `PYTHONDONTWRITEBYTECODE=1`, and `PYTHONPATH=/home/brandon/.local/share/road-reason/cseb-demo-runtime/packages`.

```bash
python initialize.py
python verify_assets.py
python preflight.py
python inference_benchmark.py
python initialize.py --freeze
python pipeline.py prepare train evaluation_preflight
python evaluate.py --benchmark-videos 2
python pipeline.py runtime_audit evaluate analyze
```

Reruns require a new ROOT and scratch paths; existing freezes are verified, never overwritten. `prepare.py` recreates missing temporary prefix tensors; `train.py` resumes optimizer and all RNG state. CUDA RoIAlign backward may have numerical nondeterminism; preflight reports resume error and checks tolerance. Inference resumes only at completed-video boundaries; AP resumes per class. No checkpoint is selected using final AP. Prespecified tuned-distilled seed0 export is separate from the live demo.

Persistent silent monitor service/timer templates are included. Check actual logs, counters, checkpoints and monitor heartbeat; timers do not promise arbitrary automated repair. New layer3 parameters change training cost and checkpoint size, but the deployed network has the same structure and parameter count.

# ResNet layer3 accuracy follow-up — COMP 765, 2026-10-23

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

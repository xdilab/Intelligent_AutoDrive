# ROAD-adapted encoder transfer into contextual DCB heads

Authorized by Brandon on October5,2026. Reuse completed Stage6 visual adaptations; do not train a new1B encoder.

Three encoder conditions: original frozen checkpoint, Stage6 ROAD focal-only best epoch3, Stage6 ROAD focal+triplet-contrastive best epoch3. Both adapted encoders originate from seed0 and update only the last visual block. Three downstream head seeds measure head-training variation conditional on each encoder, not encoder-adaptation seed variance.

Each condition regenerates identical crop/RoI/entire-frame caches (1024D) with float32 normalization/RoIAlign, BF16 encoder, FP32 pooling and FP16 storage. Original512D phrase bank stays fixed. The contextual model is byte-identical to contextual-roi-dcb/model.py: Scene Attention, DCB184 classification, optional0.001 all184/adapted-text contrastive auxiliary, no CompMLP. Train both experts for each of seeds0–2 and select the global composition blend on the exact45 gate-selection videos. Carry classification primitive scores forward. Same420/90/90 partitions, YOLO candidates, locked36,717 validation frames and detector AP@0.5; tail47/deep28/common39 retained.

Fresh frozen control is primary: historical #104 differs in crop-cache numerical path. #107 uses a different encoder pair and is a contextual reference, not the adaptation control. Upstream Stage6 auxiliary scope was triplet86/fixed-text; downstream auxiliary is all184/adapted-text. Reusing the same90 development videos for upstream checkpoint selection and downstream selection is declared; encoder train420 must be disjoint from development/gate/validation.

## Launch and safety

Canonical root `/work/bbyrd1/contextual-road-transfer-20261005`, child roots frozen/classification/contrastive. `submit.py --phase setup` requires an external passed review receipt with exact code hashes, creates input audit and GPU benchmark. The benchmark invokes `launch_after_benchmark.py` only after all3 encoder tests pass; it records ETA and invokes full submission. `submit.py` uses an exclusive nonblocking lock, persisted job registry and refuses duplicate/partial relaunch. Full cache uses24 tasks, at most8 concurrent. Each condition then runs compaction→GPU gradient/replay gate→6heads→3blends→9evaluations→summary, using afterok dependencies. Existing source checkpoints/data are never modified.

Cache shards are atomic per frame and validate contract hashes on reuse. Trainer saves model/optimizer/DCB/RNG/progress every100 steps and at phase boundaries. Time limits are at most48h, below7days. A failed dependency stops descendants; inspect actual logs and resume recorded tasks rather than duplicating submission or changing scientific protocol. Cache extraction is expensive but source/checkpoint fingerprints prevent stale-cache reuse.

Dedicated silent monitor checks scheduler, errors and30min log gaps every2min, snapshots all arms' checkpoints hourly, copies them locally and verifies readable finite model tensors plus resume state. No desktop banners. Generic NCShare watchdog also registers all job IDs/loglinks. Monitoring detects failures; it does not guarantee automatic repair for this new study. No persistent agent-repair scope is changed.

## Code ancestry

Head: exact contextual-roi-dcb/model.py. Training/evaluation/blending/compaction/checkpoint-preflight harness: contextual-roi-1b, with1024/512 dimensionality and direct original phrase adapter. Input crop functions: stage56-full/eval_input.py and base.py, copied and included in cache fingerprints. New strict encoder loader copies only verified last-block checkpoint parameters into the pinned pretrained original-size model. New preparation audits source metadata, hashes, partitions and original phrase bank; feature generation verifies frame hashes on reads.

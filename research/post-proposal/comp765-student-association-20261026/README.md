# Cross-actor association context pilot — COMP 765

Started2026-10-26, separate from the thesis and the student training studies. Tests whether explicit market-basket-style associations improve ROAD-Waymo duplex/triplet recognition beyond the existing scene GRU. No model retraining or GRU ablation is included. The working student bundle is unchanged.

The three conditions use the original frozen-ResNet distilled checkpoints for seeds0,1,2:

1. Original scores.
2. A bounded logit correction from other actors' original event predictions.
3. A context-free control using training-marginal antecedent frequencies with the same rule effects.

Rules come only from ground-truth actors in the original64training-video32frame segments. Strength selection uses the12development videos. Final evaluation reuses the exact198-video detector benchmark, with its prior-use and detector-selection caveats. This is exploratory evidence, not an untouched test.

The machine-readable protocol is [protocol.json](protocol.json). Results will be archived in `results/`; the authoritative live root is `/home/brandon/.local/share/road-reason/comp765-student-association-20261026/`.

## Rule meaning

Each transaction asks: for this target actor, is an event present on a different nearby actor? Items are the49duplex and86triplet label strings. Every target-agent/video contributes total weight1, distributed over its actor-frames. Ground-truth annotation IDs are mapped through `all_*_labels` strings into the curated outputs; the mis-indexed child arrays are never used.

A rule needs at least5joint actor-frames and3distinct videos. Conditional probability is shrunk toward the target-agent marginal with5video-equivalent pseudocounts. The log-odds change is clipped to[-1,1]; a minimum absolute log-lift of log(1.1) is required. Identity and same-event parent/child implications are excluded. No claim of field-wide methodological novelty is made.

At inference, context candidates have confidence≥0.25, class-agnostic NMS IoU0.5 and cap30. An actor's center must be within0.5 normalized image distance; IoU≥0.3 or intersection/min-area≥0.5 excludes overlapping candidates. These spatial/overlap rules also apply in mining. Context uses `max(q * original_event_probability)` over eligible other actors, filtered by their detector agent type. This is a reliability weight, not a second application of confidence to the output.

For target class j, the correction is `sum(evidence_i * effect_ij) / (1 + sum(evidence_i * rule_indicator_ij))`. Only classes compatible with the target's detector agent are changed. Final scores are `q * sigmoid(logit + strength * correction)`, with slot0=q and all primitive outputs unchanged. There is one pass, no iterative feedback and no future input. This does not enforce logical consistency between duplex and triplet heads.

## Reproduce

Environment: `/home/brandon/miniconda3/envs/road_reason/bin/python`, torch2.4.1+cu124, NumPy2.4.3, existing torchvision/Ultralytics packages. No dependency installation was needed.

For a fresh reproduction, copy this code directory to a **new unique sibling directory name**. The persistent output and `/dev/shm` directories follow that name automatically. Preserve the original experiment root, which contains frozen evidence. Original student assets remain referenced read-only.

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/brandon/.local/share/road-reason/cseb-demo-runtime/packages
PY=/home/brandon/miniconda3/envs/road_reason/bin/python
$PY mine.py
$PY preflight.py
$PY tune.py
$PY freeze_study.py
$PY asset_audit.py
$PY official.py --benchmark-videos 2
$PY official.py --workers 16
$PY runtime.py
$PY analyze.py
$PY audit.py
```

`official.py` resumes completed video prefixes only after flushing all nine score arrays. Missing arrays reset the completed-video list. AP resumes per class from signature-checked partial results. There are no training optimizer checkpoints to resume because no weights are updated. Do not delete scratch scores before AP and final audit finish. A reproduction needs roughly38GiB of scratch scores plus evaluator RAM; the current experiment used an idle A6000 GPU0 and16CPU evaluator workers.

For persistent execution, install the supplied systemd service/monitor/timer under a unique service name for a reproduction, updating their paths and `monitor.py`'s unit name before freezing. Run `monitor.py --self-test`, register the job in the wiki local-studies registry, verify fresh heartbeat/checkpoint evidence, then start the pipeline. Silent checks run every2minutes; no desktop banners. The service has a4-hour cap. Monitoring records repeated/escalated alerts; arbitrary unattended repair is not promised for this new study.

## Teacher-free raw video

The adapter uses the existing seed0 bundle and frozen rule file; it does not copy or replace the bundle:

```bash
$PY run_video.py \
  --bundle /home/brandon/.local/share/road-reason/comp765-cseb-student-20261021/bundle-seed0 \
  --rules /home/brandon/.local/share/road-reason/comp765-student-association-20261026/rules.json \
  --strength 1 \
  --video /data/datasets/road_waymo/videos/train_00169.mp4 \
  --output /tmp/association-video-demo
```

Use an output directory that does not already exist. An asset-access audit rejects external model/feature caches. The runner decodes only through the sampled present frame, samples at10Hz, and initializes state at video start. The same `correction.py` is used in detector evaluation and this adapter. Production runtime candidate filtering differs from the full-candidate AP benchmark; this is documented, not conflated.

## Evidence and limitations

- `rules.csv` gives every retained antecedent/consequent, actor-frame and video support, raw/shrunk confidence, lift and bounded effect.
- `development-selection.json` contains every seed/strength result, with proposal AP explicitly labeled.
- `freeze.json`, `provenance.json`, `preflight.json` and the copied split manifest establish source/checkpoint identity.
- Per-seed detector results retain all184output metrics. Corrected primitive metrics are inherited only after exact score-equality assertions; original baseline AP is independently replayed and checked against historical results.
- Three checkpoints provide paired variation, not a large independent sample. The final benchmark has prior use; there is no final-AP tuning. A non-significant result is not equivalence.
- Ground-truth presence during mining differs from uncertain inference evidence. Sparse rule coverage, correlated actor-frames and short training segments constrain rare-event conclusions. Any gain may include score calibration. This experiment does not prove what the GRU learned internally.

A setup-only import failure occurred because `select.py` shadowed Python's standard-library module. It was renamed `tune.py` before development evaluation; the failed log is preserved. Scientific settings were unchanged.

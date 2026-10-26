# Full-video mining and two-event association rules

COMP 765 extension started October 26, 2026. This tests whether additional training-only mining data and two-event rules improve an unchanged causal road-event student beyond its scene GRU. It extends the completed short-segment association pilot; it is separate from the thesis and does not retrain the student.

**Status:** the three-seed block is complete and audited. [Results](results/FINDINGS.md). At the user's request, the unchanged seven-condition comparison is being extended to seeds 0–9 in `../comp765-student-association-context-seeds-20261026/`; these three-seed findings remain an immutable intermediate record.

## Fixed comparison

| Condition | Mining frames | Context used at inference |
|---|---|---|
| `baseline` | None | Original student only |
| `short_single` | Original 32-frame segments | Single-event rules from current nearby actors |
| `short_frequency` | Same short segments | Training-marginal event presence |
| `full_single` | All annotated frames in the same 64 videos | Single-event rules from current nearby actors |
| `full_frequency` | Same full videos | Training-marginal event presence |
| `full_pair` | Same full videos | Full single-event correction plus residual two-event rules |
| `full_pair_frequency` | Same full videos | Matched marginal single/pair presence |

All conditions use the same original distilled checkpoints, seeds 0, 1 and 2, with the frozen ResNet50 encoder through layer3, shared actor RoI features, geometry, agent embeddings and scene GRU. Architecture, weights, candidate lists, temporal state, score assembly and evaluator remain fixed. Strength is fixed at the pilot's value of 1; no new development or final-AP tuning occurs.

The primary contrasts are `full_single − short_single` (data scale), `full_pair − full_single` (rule complexity), and `full_pair − baseline` (total practical effect). Matched frequency controls help distinguish current-scene information from context-free calibration. Three paired checkpoints give exploratory seed uncertainty, not an independent test-set guarantee.

## What “two events” means

A transaction is a target actor and the observed event labels on other nearby actors. A pair is two distinct event labels, potentially on the same or different neighboring actors. Trivial identity and duplex/triplet parent-child implications are excluded. The pair effect is an increment only when its shrunk target conditional exceeds both constituent single conditionals or falls below both. At most 16 rules per consequent are retained using training-only support/effect ranking.

At inference, pair evidence is the minimum of the two original event evidences. Corrections are computed once without feedback, remain bounded, and modify only duplex/triplet outputs. Final detector confidence is applied once. No teacher, teacher features, future frames or extra recurrent state are required.

See [protocol.json](protocol.json) for the exact formulas, thresholds, controls and limitations.

## Data and evaluation

- Same 64 training video identities: 12,270 annotated frames, 269,600 target actor-frames. Of these, 11,948 frames contain usable target actors.
- Pilot short segments: 2,036 annotated frames, 45,766 target actor-frames. Full-video mining supplies additional ground-truth supervision without changing student weights.
- Same 12 development videos; no new wording/rule/strength selection.
- Same 198 official validation videos: 36,717 scored frames, 38,663 continuous causal prefix steps, 6,163,968 fixed candidates, detector AP at IoU 0.5.
- Same fixed training-derived Tail47, Deep28 and Common39 membership. IDs are mapped through label strings, never `duplex_childs`/`triplet_childs`.
- The validation benchmark has prior research and detector-selection exposure. It is not an untouched full-system test.

## Reproduction

Code directory:
`/data/repos/ROAD_Reason/research/post-proposal/comp765-student-association-context-20261026`

Run directory:
`/home/brandon/.local/share/road-reason/comp765-student-association-context-20261026`

Use the existing `road_reason` environment; do not upgrade shared dependencies. Source assets and checkpoint hashes are recorded in `asset-audit.json`, `preflight.json`, `provenance.json` and `freeze.json` in the run directory. `split-manifest.json` preserves exact video IDs and prefix endpoints. Original assets remain read-only.

```bash
cd /data/repos/ROAD_Reason/research/post-proposal/comp765-student-association-context-20261026
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/home/brandon/.local/share/road-reason/cseb-demo-runtime/packages
PYTHON=/home/brandon/miniconda3/envs/road_reason/bin/python
# Fresh isolated run root required for mining; never overwrite a completed study.
$PYTHON mine.py --benchmark-videos 2
$PYTHON mine.py
$PYTHON asset_audit.py
$PYTHON preflight.py
$PYTHON freeze_study.py
$PYTHON official.py --benchmark-videos 2
$PYTHON official.py --workers 16
$PYTHON runtime.py
$PYTHON replay_audit.py
$PYTHON analyze.py
$PYTHON audit.py
$PYTHON archive.py
```

`official.py` flushes all conditions before committing each completed video. The full run resumes the benchmark receipt rather than reprocessing its two videos. Partial class-AP files allow metric evaluation to resume. Old AP results are reused only after exact full score-array SHA256 reproduction; twelve new condition/seed detector metric runs are computed.

Raw-video adapter, using the original standalone bundle without modifying it:

```bash
$PYTHON run_video.py \
  --bundle /home/brandon/.local/share/road-reason/comp765-cseb-student-20261021/bundle-seed0 \
  --rules /home/brandon/.local/share/road-reason/comp765-student-association-context-20261026/rules.json \
  --strength 1 --video /path/to/video.mp4 --output /path/to/new-output
```

## Artifacts and monitoring

- `rules.json`: all full single/pair effects and training marginals; `short-rules.json`: immutable pilot rules.
- `single-rules.csv`, `pair-candidates.csv`: readable rules, supports, confidences and retention decisions.
- `runs/<condition>-seed<seed>/detector-results.json`: individual official detector metrics.
- `analysis/comparison.json`, `per-seed.csv`, `per-class.csv`, `FINDINGS.md`: all contrasts, classes and limitations.
- `runtime/<video>/report.json`, compressed predictions, `runtime-summary.json`, `replay-audit.json`: raw-video timing and independent replay.
- `score-diagnostics.json`, `final-audit.json`: output hashes and integrity checks.
- `results/`: immutable compact archive created only after the final audit passes.

A silent systemd timer checks every two minutes for pipeline failures, stalled progress, log errors, frozen-asset changes and original checkpoint/backup identity. Desktop banners remain disabled. Logs and heartbeats are persistent; arbitrary automatic repair or conversational wakeup is not guaranteed. Active-session checks inspect real progress and hardware health.

No thesis files, defense slides, original checkpoints, production inference bundle, workbook or remote branch are modified by this study.

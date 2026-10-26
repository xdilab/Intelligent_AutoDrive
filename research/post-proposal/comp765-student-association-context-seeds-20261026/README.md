# Full-video mining and two-event rules — ten paired seeds

User-requested COMP 765 extension, October 26, 2026. This extends the [audited three-seed block](../comp765-student-association-context-20261026/results/FINDINGS.md) to **all ten original distilled checkpoints, seeds 0–9**, without changing its seven conditions, rules, strength, model architecture, checkpoint selection or evaluation protocol. No student retraining is required.

**Status: complete.** All 70 condition/seed results passed the final audit: 210 aggregate-statistics checks and 11,340 exported per-class numeric checks. See [findings](results/FINDINGS.md), [per-seed results](results/analysis/per-seed.csv), and [all class outcomes](results/analysis/per-class.csv).

The user requested ten seeds after seeing the first three outcomes. The added block was fixed before its corrected-condition results, and all seven additional seeds must finish regardless of outcomes. Report the original 0–2, added 3–9, and combined 0–9 cohorts separately. The combined paired interval uses 9 degrees of freedom. This is an exploratory replication on a reused benchmark, not an untouched video test or retrospective confirmatory preregistration.

## Seven unchanged conditions

| Condition | Mining data | Evidence |
|---|---|---|
| `baseline` | None | Original student |
| `short_single` | Original 32-frame segments | Current neighboring actors, one-event rules |
| `short_frequency` | Same segments | Training-marginal presence |
| `full_single` | All annotated frames of the same 64 videos | Current neighboring actors, one-event rules |
| `full_frequency` | Same full videos | Training-marginal presence |
| `full_pair` | Same full videos | Single rules plus residual two-event conjunctions |
| `full_pair_frequency` | Same full videos | Matched marginal single/pair evidence |

Primary contrasts: `full_single − short_single` (mining-data scale), `full_pair − full_single` (rule complexity), and `full_pair − baseline` (total effect). Frequency controls use the same effects and denominators with training-marginal evidence. Strength remains 1. No additional AP tuning or seed selection occurs.

The original frozen ResNet50-through-layer3/scene-GRU student stays unchanged. Full mining covers 12,270 annotated frames and 269,600 target actor-frames, versus 2,036 frames and 45,766 actor-frames in the original segments. The frozen bank contains 3,970 single-event effects and 1,532 residual pair effects. A pair consists of two different event labels on eligible neighboring actors; both labels may belong to one neighbor. Identity and parent/child implications are excluded. See [protocol.json](protocol.json) and the [parent protocol](../comp765-student-association-context-20261026/protocol.json) for exact definitions and limitations.

## Execution and assets

Code:
`/data/repos/ROAD_Reason/research/post-proposal/comp765-student-association-context-seeds-20261026`

Run root:
`/home/brandon/.local/share/road-reason/comp765-student-association-context-seeds-20261026`

Original checkpoints 0–2:
`/home/brandon/.local/share/road-reason/comp765-cseb-student-20261021/runs/distilled-seedN/best.pt`

Additional checkpoints 3–9:
`/home/brandon/.local/share/road-reason/comp765-cseb-student-seeds-20261022/runs/distilled-seedN/best.pt`

Both training data manifests are byte-identical. All ten selected checkpoints and their verified backups passed hash, shape, finite-output, reset and seven-condition score checks. The wrapper imports the parent's frozen numerical inference and official evaluator unchanged.

Additional seeds run in batches `[3,4,5,6]` and `[7,8,9]`. Each batch processes every causal prefix, writes all seven conditions, verifies exact baseline score hashes against the original seed study, computes six new corrected-condition AP results per seed, audits all score arrays, and releases only its own audited temporary arrays. The first batch's benchmark prefix is resumed rather than recomputed. Completed video and partial class-AP receipts support recovery.

This yields **70 condition/seed results**: 21 audited parent results, 7 exact baseline replays, and 42 newly computed corrected-condition detector evaluations. Detector AP uses the unchanged 198 videos, 36,717 scored frames, 38,663 recurrent steps, 6,163,968 candidates and IoU 0.5. Tail47/Deep28/Common39 remain training-derived. Prior validation and detector-selection exposure is disclosed.

## Reproduction

Use the existing environment and a fresh isolated study/run directory; never overwrite a completed freeze or archive. Do not upgrade shared dependencies.

```bash
cd /data/repos/ROAD_Reason/research/post-proposal/comp765-student-association-context-seeds-20261026
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/home/brandon/.local/share/road-reason/cseb-demo-runtime/packages
PYTHON=/home/brandon/miniconda3/envs/road_reason/bin/python
$PYTHON prepare.py
$PYTHON official.py --benchmark-videos 2
$PYTHON official.py --workers 16
$PYTHON analyze.py
$PYTHON audit.py
$PYTHON archive.py
```

`prepare.py` refuses to overwrite a freeze. On an existing run, resume `official.py` directly after verifying its frozen inputs. Completed batches are not rerun. No teacher or new visual-feature generation is needed.

## Results and integrity records

- `protocol.json`, `freeze.json`, `provenance.json`, `execution-revision.json`: protocol, source/environment and immutable identities.
- `preflight.json`: ten-checkpoint validation and training/input equivalence.
- `runs/<condition>-seedN/detector-results.json`: all 70 per-seed official results.
- `batches/seeds3-6/` and `batches/seeds7-9/`: image records, original-baseline replay, per-class metrics, score hashes, batch audits and cleanup receipts.
- `analysis/comparison.json`: original, added and combined cohort statistics and paired intervals.
- `analysis/per-seed.csv`, `analysis/per-class.csv`: all endpoints and every duplex/triplet class.
- `analysis/FINDINGS.md` and figures: full outcomes and limitations, including negative or inconclusive findings.
- `final-audit.json`: independent arithmetic checks, source/checkpoint identities and complete-result coverage.
- `results/`: compact immutable archive created only after the final audit passes.

The existing prespecified seed-0 full-pair raw-video adapter and its audited 593-frame runtime/replay record are reused by hash. Adding seeds does not change inference cost or justify a new speed claim. Its [runner](../comp765-student-association-context-20261026/run_video.py) uses the original standalone student bundle and frozen rule JSON, without teacher assets, cached features or future frames. No production model promotion occurs.

During execution the silent monitor checked logs, progress gaps, frozen files and all ten checkpoint backups every 120 seconds. Its dedicated timer was retired after successful completion and a final clean heartbeat. Desktop banners remain disabled. The service allows five hours; measured launch estimates and assumptions are recorded in `launch-plan.json`. Monitoring detects and records failures; arbitrary automatic repair or conversational wakeup is not guaranteed.

No thesis, presentation slides, original checkpoints, working demo, workbook, or remote branch is changed by this extension.

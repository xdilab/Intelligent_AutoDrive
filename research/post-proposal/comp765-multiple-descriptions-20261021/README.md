# COMP 765 — Multiple Class Descriptions for Long-Tail Road-Event Recognition

New coursework begun2026-10-21, separate from the completed master's thesis. See [PROTOCOL.md](PROTOCOL.md) and [protocol.json](protocol.json) for the frozen design, novelty audit, inherited wording limitations, endpoints and analysis rules.

Two freshly trained standalone semantic crop phrase heads differ only in the frozen prototype matrix: one exact canonical description versus five fixed descriptions averaged in text embedding space. Both use the paired InternVideo2 CLIP_S text tower, the same frozen crop caches,184 output rows,524,985 trainable parameters, ten epochs and paired seeds0/1/2. Row0 agentness remains single-description in both conditions. There are183 semantic classes, not184.

Code reuses the established repeated-seed `PhraseHead` and focal loss plus the unchanged published detector AP implementation. It does not require visual extraction or dependency installation.

## Reproduction

Existing environment: `/home/brandon/miniconda3/bin/python`. Run outside the restricted sandbox to access GPU0. The output path defaults to `/home/brandon/.local/share/road-reason/comp765-multiple-descriptions-20261021` and can be set with `--root` for a fresh reproduction. Do not overwrite the frozen original run.

```bash
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python prepare.py prototypes --root NEW_OUTPUT
python prepare.py audit --root NEW_OUTPUT
python validate.py validation --root NEW_OUTPUT
python validate.py benchmark --root NEW_OUTPUT
python prepare.py freeze --root NEW_OUTPUT
python pipeline.py --root NEW_OUTPUT
```

`pipeline.py` runs six training jobs, six complete detector evaluations, then analysis. An interrupted pipeline can be invoked again with the same root: training resumes from atomic checkpoints; detector AP resumes completed classes after regenerating scores in RAM. It rejects changed frozen code/assets. Do not modify phrases or settings after examining validation AP.

Training commands: `python train.py --root OUTPUT`; evaluation: `python evaluate.py --root OUTPUT --workers 8`; aggregation: `python analyze.py --root OUTPUT`. Both condition configurations and the seed list are in `protocol.json`. Checkpoint selection is final epoch10, with no validation or development selection.

## Artifacts

- `descriptions.json`: complete human-readable916-string manifest,183×5 semantic descriptions and one unchanged special row, explicit glossary and caveats.
- `prototypes-{single,multi}.pt`, `description-embeddings.pt`: ordered matrices and separately encoded descriptions.
- `tokenization.json`, `geometry.json`, `cosine-*.npy`: token limits, within-class uniqueness, norms, cosine geometry and canonical-to-aggregate similarities.
- `provenance.json`, `preflight-assets.json`, `training-frames.json`, `frequency-groups.json`, `freeze.json`: exact asset hashes, versions, splits, dimensions, class membership and pre-run freeze.
- `validation.json`, `benchmark.json`, `launch.json`: test evidence, measured throughput and launch ETA.
- `runs/{single,multi}-seed{0,1,2}/`: per-run initialization, epoch loss/order hashes, resumable/final checkpoints, detector AP and completion records.
- `analysis/RESULTS.md`, `comparison.json`, `per-seed.csv`, `per-class-triplet.csv`: complete paired analysis. These exist only after all six evaluations finish.
- `logs/`, `job.json`, `progress.json`, `monitor/`, `verified-checkpoint-backup/`: runtime and silent monitoring evidence.

Detector AP is measured on the exact corrected36,717-frame shared manifest at IoU.5. This is not proposal classification AP or development crop AP. Report negative/inconclusive outcomes, all per-class differences, and the limitations of a three-seed paired-t interval. Prototype geometry and added wording quality remain confounded with the practical averaging intervention.

No thesis, slide, existing result or workbook is edited by these commands. Large assets and checkpoints stay outside Git.

## Completed October21,2026

All six full detector evaluations completed and passed the independent final audit. Tail47: single5.7952±0.1040 versus multi5.6608±0.0883; paired change−0.1344percentage points,95%paired-t interval[−0.5006,+0.2317]. All three observed deltas are negative, but this small study does not establish a reliable decline or equivalence. See [full results](results/RESULTS.md), [all phrases](results/descriptions.json), [all per-class differences](results/per-class-triplet.csv), and [reproducibility record](results/reproducibility.json).

Primary run code revision: `3f1a43a`; full SHA and frozen source hashes are in `results/freeze.json`. The result-audit helper was added after completion; it verifies the original frozen files without changing them. Recheck a completed run with `python verify_results.py --root OUTPUT`. The study-specific monitor timer was disabled after successful completion; its final heartbeat, alerts and12 verified checkpoint backups are preserved. Install it for an authorized resumed run with `python install_monitor.py --root OUTPUT`.

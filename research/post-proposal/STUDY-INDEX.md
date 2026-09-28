# Research code and evidence index

Snapshot: September 28, 2026. Canonical experiment development belongs here.

## Code

Earlier localizer, VLM, BDD-X and crop-head work is under `../../experiments/` (relative to this directory: `../../experiments` resolves to the repository experiments directory). Original Stage 4/5/6 crop experiments also live in `experiments/exp12_phrase_head` at repository root. Stage numbers describe architecture; experiment folder numbers describe study history.

| Study directory | Configuration reference |
|---|---|
| `contextual-gate/` | `protocol.json` |
| `contextual-language-control-review/` | README / source files |
| `contextual-roi/` | `protocol.json` |
| `contextual-roi-all184/` | `protocol.json` |
| `contextual-roi-comp/` | `protocol.json` |
| `contextual-roi-concat/` | `protocol.json` |
| `contextual-roi-dcb/` | `protocol.json` |
| `contextual-roi-langctl/` | `protocol.json` |
| `monitoring/` | README / source files |
| `reporting/` | README / source files |
| `stage56-full/` | `protocol.json` |
| `stage7-stage5/` | `protocol.json` |

## Historical sources and results

- `historical/`: September 14 consolidation; `code-inventory.json` verifies 149 original files.
- `historical/supplement-20260928/`: 60 additional wiki scripts/revisions, including earlier loss/evaluation implementations and presentation generators. These are archival copies, not replacements for active modules. They can depend on companion wiki assets and absolute paths.
- `results-snapshots/20260928/`: compact JSON/CSV snapshots from 22 study/evidence families. Original filenames and status fields are preserved. Partial, progress, smoke, development, detector and retrieval metrics must not be conflated. A snapshot is not a claim that a running experiment finished.
- `sync-manifest-20260928.json`: original paths, SHA-256 and byte sizes for every newly copied file.
- `reporting/`: workbook and comparison tools, including the latest published-model tail analysis.

## External data and checkpoints

Data, large caches and checkpoints are deliberately outside the Git synchronization scope. Existing files were neither moved nor deleted.

- Wiki evidence: `/data/repos/wiki/artifacts/`; study-specific directories mirror the snapshot paths above.
- Local original crop caches/checkpoints: `/data/repos/ROAD_Reason/experiments/exp12_phrase_head/` and `crop_full/`.
- Local published baseline code/checkpoints: `/data/repos/PedestrianIntent++/ROAD_plus_plus_Baseline/` (external dependency).
- NCShare full adaptation: `/work/bbyrd1/stage56-full-20260914/`; canonical local code `stage56-full/`. Deployment and collected result records are included in the results snapshot.
- Other study data/output paths are recorded in each study's `protocol.json`, `pipeline.py`, and result signatures; do not substitute paths or label orders silently.

This synchronization does not redeploy code into running NCShare jobs, restart training, or certify remote checkpoint backup freshness. Git synchronization protects code and compact evidence, not the large external assets.

# Research Progress Log

## Session: April 2026

---

## Project Direction

Research focus: **Scene understanding and action detection/prediction using a VLM as a reasoner**, grounded in the ROAD-Waymo (ROAD++) dataset.

- Not pedestrian intent prediction — shifted to full scene understanding
- Primary benchmark: ROAD-Waymo (`road_waymo_trainval_v1.1.json`)
- Supervisor: Dr. Moradi
- Workstation: 2× NVIDIA RTX A6000 (49GB each)
- Baseline code: `/data/repos/PedestrianIntent++/ROAD_plus_plus_Baseline/`
- Dataset: `/data/datasets/ROAD_plusplus/`

---

## Approach 1 — 3D-RetinaNet Baseline (COMPLETE)

### Model
- **Architecture**: ResNet50 + I3D (3D convolutions, Kinetics-400 pretrained)
- **Detection**: Anchor-based tube detection with FPN (P3–P7)
- **Classification**: 6 multi-label sigmoid heads (agent_ness, agent, action, loc, duplex, triplet)
- **Training**: 30 epochs, LR=0.0041, milestones at [20, 25], batch size 4, 8-frame clips, 600px

### Training Command
```bash
CUDA_VISIBLE_DEVICES=0,1 python main.py /data/datasets/ROAD_plusplus/ output/ kinetics-pt/ \
  --MODE=train --ARCH=resnet50 --MODEL_TYPE=I3D \
  --DATASET=road_waymo --TEST_DATASET=road_waymo \
  --TRAIN_SUBSETS=train --VAL_SUBSETS=val \
  --SEQ_LEN=8 --TEST_SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.0041 --MIN_SIZE=600
```

### Results (Val f-mAP @ IoU 0.5)

| model | epoch | metric | split | iou | agent_ness | agent | action | loc | duplex | triplet |
|-------|-------|--------|-------|-----|-----------|-------|--------|-----|--------|---------|
| 3D-RetinaNet-I3D | 25 | f-mAP | val | 0.5 | 23.35 | 17.76 | 15.28 | 13.73 | 13.44 | 9.17 |

Best checkpoint: **epoch 25** (`model_000025.pth`)  
Checkpoint location: `output/road_waymo/cache/resnet50I3D600-Pkinetics-b4s8x1x1-road_waymo-alltn-h3x3x3/`

### vs. Paper Baseline (Table 3, val f-mAP %)
| Model | Agents | Actions | Locations | Duplexes | Events |
|-------|--------|---------|-----------|----------|--------|
| Paper I3D-08 | 15.7 | 12.3 | 12.4 | 10.5 | 6.2 |
| **Ours ep25** | **17.76** | **15.28** | **13.73** | **13.44** | **9.17** |

Beats paper baseline on all metrics. Note: paper numbers are from test set via `gen_dets`; ours are training-time val subset. True comparison requires test set submission (test labels withheld by authors).

### Notes
- Metrics are **frame-mAP (f-mAP)**, not accuracy. mAP is standard for multi-label detection with class imbalance.
- Video-mAP (tube-level) not computed — requires `gen_dets` mode which was skipped after training converged.
- Model converged by epoch 25, flat through epoch 30.

---

## Approach 2 — Neuro-Symbolic 3D-RetinaNet with T-norm Constraints (IN PROGRESS)

### Overview
Same architecture as Approach 1, with an additional **constraint violation loss** based on valid duplex/triplet combinations from the annotation JSON.

### T-norm Loss Implementation

**File**: `tnorm_loss.py` (also at `modules/tnorm_loss.py` in baseline repo)

Two t-norms implemented:
- **Gödel**: `T(a,b) = min(a,b)` — best on ROAD-Waymo per paper Table 7
- **Łukasiewicz**: `T(a,b) = max(0, a+b-1)`

For each invalid duplex (agent_i, action_j) pair not in `duplex_childs`, violation = `T(p_agent_i, p_action_j)`.  
For each invalid triplet (agent_i, action_j, loc_k) not in `triplet_childs`, violation = `T(T(p_agent_i, p_action_j), p_loc_k)`.

Loss applied only to **positive anchors** (not all non-ignored anchors — applying to all caused OOM on A6000 due to 3,452 invalid triplet pairs × large anchor count).

**Invalid pairs**: 181 duplex pairs (220 possible − 39 valid), 3,452 triplet triples (3,520 possible − 68 valid)

### Changes to Baseline Codebase
- `modules/tnorm_loss.py` — new file, `TNormConstraintLoss` module
- `modules/detection_loss.py` — imports and instantiates `TNormConstraintLoss` in `FocalLoss`, returns `tnorm_loss` as 3rd value
- `train.py` — unpacks 3rd return value, logs `con-loss`, adds to total loss
- `main.py` — adds `--TNORM` and `--TNORM_LAMBDA` CLI args, passes `args.childs` from dataset

### Training Command
```bash
CUDA_VISIBLE_DEVICES=0,1 python main.py /data/datasets/ROAD_plusplus/ output/ kinetics-pt/ \
  --MODE=train --ARCH=resnet50 --MODEL_TYPE=I3D \
  --DATASET=road_waymo --TEST_DATASET=road_waymo \
  --TRAIN_SUBSETS=train --VAL_SUBSETS=val \
  --SEQ_LEN=8 --TEST_SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.0041 --MIN_SIZE=600 \
  --TNORM=godel --TNORM_LAMBDA=1.0
```

Log: `output/train_tnorm.log`

### Status
- Training started April 1 2026, currently at epoch 1
- `con-loss` ~0.02 at epoch 1 (constraint violations small at start, expected)
- Expected to complete in ~2 days

### Expected Outcome (from paper Table 7)
| Model | Agents | Actions | Locations |
|-------|--------|---------|-----------|
| I3D baseline | 19.05 | 14.80 | 15.31 |
| + Gödel | **19.70** | **15.59** | **16.03** |
| + Łukasiewicz | 18.07 | 15.13 | 15.37 |

Gödel t-norm expected to give modest but consistent improvements, especially on location detection.

---

## Annotation Analysis — Corner Cases & Advisor Scenario Mapping

### Scripts
- `analysis/count_combinations.py` — frequency counter over all 3.3M boxes
- `analysis/map_examples.py` — maps Dr. Moradi's 9 scenarios to closest valid labels

### Key Findings

**Dataset scale**: 3,304,353 annotated boxes, 798 videos. Severe class imbalance — `Car-Stop` alone = 1,363,455 instances (~39% of all duplexes).

**Corner cases (bottom 15% by frequency)**:
- 7 of 49 valid duplexes are corner cases (threshold ≤ 3,365)
- 12 of 86 valid triplets are corner cases (threshold ≤ 1,312)
- `Cyc-MovTow-RhtPav` has only **61 instances** — near zero-shot

**Critically underrepresented agents**: `EmVeh` (emergency vehicles) and `Cyc` (cyclists) dominate the corner case list — highest safety relevance, lowest data support.

**Advisor scenario mapping (9 examples)**:
- 6/9 → direct_match (physical event representable in labels)
- 3/9 → partial_match (label captures *what*, not *why* or *what to do*)
  - Cyclist arm signalling: label has position, not gesture
  - Plastic bag ambiguity: label has large vehicle stopped, not semantic context
  - Human road gesture: label has crossing pedestrian, not active instruction

**Conclusion**: Structured labels are necessary but not sufficient. The VLM reasoning layer (Approach 3) is motivated by this expressiveness gap.

Full report: `FINDINGS_REPORT.md`  
CSVs: `analysis/combination_counts.csv`, `analysis/example_mapping.csv`

---

## Approach 3 — Logic-Constrained Generative VLM (PLANNED)

**Status**: Architecture design phase. Training infrastructure (Approaches 1 & 2) must complete first.

### Plan
- Fine-tune a generative VLM (LLaVA or InstructBLIP) on ROAD-Waymo video clips
- Joint output: structured JSON labels + natural language reasoning
- Apply Gödel t-norm constraint loss to structured label logits
- Text reasoning trained to be consistent with constrained predictions

### T-norm + VLM Integration Options
- **Option A** (token logit extraction): Extract logits for label tokens, apply t-norm — full gradient, most complex
- **Option B** (constrained decoding): Mask invalid token combinations at decode time — no gradient but cleaner output
- **Option C** (data-implicit): Construct training prompts that never contain invalid combinations — simplest, viable starting point

Recommended sequence: start with C, add B, stretch goal A.

### Hardware
- 2× RTX A6000 (49GB each, ~98GB total after baseline training completes)
- LLaVA-1.6 (7B) fits in ~14GB per GPU at 4-bit; full fine-tune needs ~40GB+

---

## Paper Reference

`ROAD_waymo.pdf` — Khan et al., "ROAD-Waymo: Action Awareness at Scale for Autonomous Driving", arXiv:2411.01683v2, Nov 2024.

Key tables:
- **Table 3**: Frame-level baselines (f-mAP Val/Test) — our comparison target
- **Table 7**: T-norm neuro-symbolic results (Approach 2 target)

---

## Repo Structure

```
ROAD_Reason/
├── APPROACHES.md              # Full 6-approach research roadmap
├── PROGRESS.md                # This file — session log
├── FINDINGS_REPORT.md         # Corner case & advisor scenario analysis for Dr. Moradi
├── ROAD_waymo.pdf             # Paper reference
├── README.md
├── tnorm_loss.py              # T-norm constraint loss module
├── analysis/
│   ├── count_combinations.py  # Script 1: frequency counter
│   ├── map_examples.py        # Script 2: advisor scenario mapper
│   ├── combination_counts.csv # Output: all 135 valid combo frequencies
│   ├── example_mapping.csv    # Output: 9 advisor scenarios → labels
│   └── baseline_val_metrics.csv  # Approach 1 best epoch results
├── viz/                       # Annotated frame visualisations
└── EMAIL_TO_DR_MORADI_DRAFT.md
```

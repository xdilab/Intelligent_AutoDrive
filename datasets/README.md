# Training Datasets — ROAD_Reason Approach 3

**Purpose:** Implementation-level reference for Dr. Moradi meeting (2026-04-16 notes)  
**Model:** Qwen2.5-VL-7B with three task-specific heads, one per dataset  
**Training order (confirmed 2026-04-10):** ROAD-R → BDD-X → CoVLA → joint

---

## Three Datasets at a Glance

| Property | ROAD-Waymo | CoVLA | BDD-X |
|----------|-----------|-------|-------|
| **Full name** | ROAD-Waymo / ROAD-R | Comprehensive Vision-Language-Action | Berkeley DeepDrive eXplanation |
| **Paper** | Salmank et al., 2024 | Arai et al., arXiv:2408.10845, 2025 | Kim et al., ECCV 2018 |
| **Location** | USA | Tokyo, Japan | USA (diverse) |
| **Scale** | 798 videos, 153,534 frames, 3.3M boxes | 10,000 clips, 6,000,000 frames | 7,000 videos, 26,538 action segments |
| **Annotation type** | Bounding boxes + compositional labels | Frame-level captions + ego state + trajectory | Action-level text + justification |
| **Annotation method** | Manual (expert) | Auto (rule-based + VideoLLaMA2-7B) | Manual (driving instructors) |
| **Primary supervision** | agent/action/loc detection + t-norm constraints | caption + trajectory | action + justification (causal) |
| **DSDAG node** | Scene structure (compositional) | Y (action) + W (reason) + trajectory | Y (action) + W (reason) |
| **Output type** | DoubleX (49) + TripleX (86) predictions at mAP@0.5 | Caption text + 60×3 trajectory | Action text + justification text |
| **Local path** | `/data/datasets/road_waymo/` | `/data/datasets/CoVLA/mini/` (full pending) | `/data/datasets/BDD-X/` |

---

## Why Each Dataset

**ROAD-Waymo** provides the core structured perception task: detect agents and classify their compositional behavior using 5-level labels (agent → action → location → duplex → triplet). Constraints (t-norm loss) enforce logical consistency of co-predictions. This is the primary training signal for scene understanding and the main contribution of Approach 3.

**BDD-X** provides high-quality causal reasoning supervision. Driving instructors wrote explicit "because..." justifications for ~26K action segments. This is the best available W-node (reason) supervision — human-authored, US driving context, explicit causal language. Small scale but high quality.

**CoVLA** provides scale for the same supervision types as BDD-X. 6M frame-level captions + ego trajectories cover speed/steering/maneuver context. Lower reasoning quality (auto-generated) but massive coverage. Also provides trajectory regression targets (60×3 waypoints, 3-second horizon).

---

## Model Architecture (Approach 3)

```
Qwen2.5-VL-7B (shared ViT backbone, 3584-dim hidden)
     │
     ├── ROAD-R head ───────── 5 classification heads (agent/action/loc/duplex/triplet)
     │                         Loss: BCE×5 + T-norm constraint (λ=0.1)
     │                         Eval: mAP@0.5 per head
     │
     ├── BDD-X head ─────────  Language model: action string + justification string
     │                         Loss: cross-entropy (action) + cross-entropy (justification)
     │                         Eval: METEOR / CIDEr / BLEU
     │
     └── CoVLA head ──────────  Language model: plain_caption + risk
                               Trajectory MLP: [hidden → 60×3]
                               Loss: cross-entropy (captions) + MSE (trajectory)
                               Eval: METEOR + ADE/FDE
```

Source: `wiki/methods/qwen25-vl-multitask.md`

---

## Training Order and Current Status

| Stage | Dataset | Status (as of 2026-04-16) |
|-------|---------|--------------------------|
| 1 | ROAD-R (Exp1: GT boxes, frozen ViT) | ✅ Epoch 6 complete — agent mAP 0.357, duplex mAP 0.123 |
| 1b | ROAD-R (Exp1b: FCOS detection head) | 🔄 Training in progress (started 2026-04-17, 15 epochs) |
| 2 | BDD-X fine-tuning | ⏳ Not started |
| 3 | CoVLA fine-tuning | ⏳ Not started |
| 4 | Joint training (all 3) | ⏳ Not started |

Training order confirmed by Dr. Moradi 2026-04-10.

---

## Key Numbers (All Verified)

### ROAD-Waymo
- 798 train + 202 test videos (paper claims 1,000 — use JSON numbers)
- 153,534 annotated frames (paper claims 198K)
- 3,304,353 total boxes (paper claims 3.9M)
- 10 agents · 22 actions · 16 locations → 49 valid duplexes · 86 valid triplets
- Clip length: 8 consecutive annotated frames (CLIP_LEN=8 in config.py)
- T-norm: Łukasiewicz in Exp1 config; paper Table 7 recommends Gödel (open TODO)
- Constraint violation rate after training: 0.021% (t-norm effective)

### CoVLA
- 10,000 clips, 6,000,000 frames, 83.3 hours
- 1928×1208 @ 20 FPS, H.265 video
- Trajectory: 60×3 points, ~3 seconds at 20 FPS
- Mini subset: 50 scenes × 600 frames = 30,000 frames (available)
- Full dataset: download pending

### BDD-X
- 7,000 unique videos, 26,538 annotated action segments
- Mean segment: 8.8 seconds, mean justification: 7.9 words
- 5,588 train / 698 val / 698 test

---

## T-Norm Loss (ROAD-Waymo specific)

The t-norm loss penalizes simultaneous prediction of agent+action (or agent+action+location) combinations that are semantically impossible.

**Paper formula (Łukasiewicz):**
```
violation(a, b) = max(0, p(agent_i) + p(action_j) - 1)  for each invalid pair
L_tnorm = mean(violations)
```

**Implementation (tnorm_loss.py):** Two t-norms available (Gödel / Łukasiewicz), λ weighting, nested composition for triplets: `T(T(agent, action), location)`. Flat prediction vector [N, 49] fed from ROADLoss.

Details: `road-waymo/dataset.md` → T-Norm Loss section  
Code: `road-waymo/examples/tnorm-forward.py`

---

## Constraint Examples

Valid: `Ped-Stop-RhtPav` (pedestrian stopped on right pavement)  
Valid: `Car-MovAway-OutgoLane` (car driving away in outgoing lane)  
Invalid: `Ped-Red` (pedestrians don't display red lights)  
Invalid: `TL-MovAway` (traffic lights don't move)  
Invalid: `Car-PushObj` (cars don't push objects)

49 valid duplexes / 220 possible → **171 invalid**  
86 valid triplets / 3,520 possible → **3,434 invalid**

Full lists: `road-waymo/examples/duplex-triplet-labels.json`

---

## Folder Structure

```
datasets/
├── README.md                        ← this file
├── road-waymo/
│   ├── dataset.md                   ← full ROAD-Waymo reference
│   └── examples/
│       ├── annotation-frame.json    ← real annotated frame (train_00407, frame 1)
│       ├── duplex-triplet-labels.json ← all 49 duplexes + 86 triplets
│       ├── constraint-violation.md  ← valid vs. invalid combo examples
│       ├── tnorm-forward.py         ← t-norm forward pass with annotations
│       └── clip-target-structure.md ← tensor layout for one training clip
├── covla/
│   ├── dataset.md                   ← full CoVLA reference
│   └── examples/
│       ├── caption-frame.jsonl      ← real caption entry (scene 2022-07-14, frame 0)
│       ├── states-frame.jsonl       ← real states entry (ego speed, steering, trajectory)
│       ├── traffic-lights-frame.jsonl ← real TL detection: green signal
│       ├── front-car-frame.jsonl    ← real front-car entry: no lead at frame 44
│       └── caption-pair.md          ← annotated plain_caption + risk with DSDAG mapping
└── bdd-x/
    ├── dataset.md                   ← full BDD-X reference
    └── examples/
        ├── annotation-row.csv       ← real CSV row (header + 5-slot submission)
        ├── action-justification-pairs.md ← 5 real action+justification pairs
        ├── split-sample.txt         ← first 5 lines of train.txt
        ├── column-schema.md         ← all 61 columns with parsing code
        └── top-actions.md           ← top action verbs + BDD-X vs. CoVLA table
```

---

## Questions This Document Answers

From Dr. Moradi's 2026-04-16 meeting notes:

| Question | Answer location |
|----------|-----------------|
| How many frames in an image sequence? | `road-waymo/dataset.md` → Input Format → CLIP_LEN=8 |
| Model architecture | This README → Model Architecture section |
| Output: DoubleX and TripleX? | `road-waymo/dataset.md` → Output Format |
| Examples of possible/impossible combos | `road-waymo/examples/constraint-violation.md` |
| How constraints are coded | `road-waymo/dataset.md` → How Constraints Are Stored |
| T-norm loss calculation in code | `road-waymo/examples/tnorm-forward.py` |
| How t-norm differs from paper | `road-waymo/dataset.md` → Paper vs. Implementation Differences |
| Full list of Agent, Actions, Locations | `road-waymo/dataset.md` → class tables |
| CoVLA data flow | `covla/dataset.md` → Data Flow |
| BDD-X annotation schema | `bdd-x/dataset.md` → CSV Annotation Schema |

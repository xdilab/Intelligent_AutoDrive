# ROAD-Waymo Annotation Analysis: Corner Cases and Scenario Representability

**Prepared for:** Dr. Moradi  
**Date:** April 2026  
**Author:** Brandon Byrd  

---

## 1. Overview

This report summarises a programmatic analysis of the ROAD-Waymo annotation file (`road_waymo_trainval_v1.1.json`) conducted to:

1. Identify **corner cases** — valid agent/action/location combinations that occur rarely in the dataset and are therefore unlikely to be learned reliably by standard detection models.
2. Assess how well the **scenario examples you proposed** (school bus, signalling cyclist, distracted pedestrian, etc.) are representable within ROAD-Waymo's structured label vocabulary.

The analysis underpins the motivation for Approach 3 of the research plan: a **logic-constrained generative VLM** capable of reasoning through scenarios that structured labels cannot fully encode.

---

## 2. Method

### 2.1 Dataset

- **Annotation file:** `road_waymo_trainval_v1.1.json`
- **Total annotated boxes:** 3,304,353 across 798 videos (600 train / 198 val)
- **Label hierarchy:** agent (10 classes) × action (22 classes) × location (16 classes)
- **Valid combinations:** 49 duplexes (agent + action), 86 triplets (agent + action + location), enforced via `duplex_childs` and `triplet_childs` constraint lists

### 2.2 Script 1 — Frequency Counter (`count_combinations.py`)

All 3.3M annotated boxes were iterated. For each box, every `duplex_id` and `triplet_id` was resolved to its label string and counted, retaining only combinations that appear in the validated constraint lists (49 duplexes, 86 triplets). Combinations were ranked by frequency and flagged as **corner cases** if they fell below either:

- An absolute count threshold (default: **< 50 instances**), or
- The **bottom 15% by frequency** across the valid combination set

Both thresholds are configurable CLI arguments.

### 2.3 Script 2 — Advisor Example Mapper (`map_examples.py`)

Each of the 9 advisor-provided scenarios was represented as a set of keyword tokens derived from the scenario description. Each token was matched against all 135 valid label strings (49 duplexes + 86 triplets) using a combined score:

```
score = 0.7 × keyword_coverage + 0.3 × sequence_similarity
```

The highest-scoring label was selected and classified as:
- **direct_match** (score ≥ 0.5): the scenario is reasonably representable by the label
- **partial_match** (0.2 ≤ score < 0.5): the label captures the physical event but not the intent or social context
- **not_representable** (score < 0.2): no adequate label exists

Frequency in the dataset was cross-referenced from Script 1's output.

---

## 3. Results

### 3.1 Label Distribution — Severe Class Imbalance

The dataset is dominated by a small number of common combinations:

| Rank | Combination | Count |
|------|-------------|-------|
| 1 | Car-Stop | 1,363,455 |
| 2 | Car-MovAway | 411,705 |
| 3 | Car-MovTow | 309,991 |
| 4 | Car-MovAway-OutgoLane | 302,426 |
| 5 | Car-MovTow-IncomLane | 286,573 |
| 6 | Ped-Stop | 190,902 |
| 7 | Car-Brake | 161,757 |
| 8 | MedVeh-Stop | 160,324 |
| 9 | Car-Stop-Jun | 159,359 |
| 10 | Ped-MovAway | 146,822 |

`Car-Stop` alone accounts for ~39% of all duplex instances. This is a known characteristic of ego-vehicle datasets — the majority of visible agents are stationary cars in traffic.

---

### 3.2 Corner Cases — Duplex Level

**7 of 49 valid duplexes** are corner cases (bottom 15% by frequency, threshold: ≤ 3,365 instances):

| Combination | Count | Notes |
|-------------|-------|-------|
| TL-Amber | 938 | Transient traffic light state |
| EmVeh-Stop | 1,069 | Emergency vehicle stopped — extremely rare scenario |
| Cyc-Stop | 1,559 | Cyclist stopped — hard to distinguish from parked bike |
| MedVeh-TurRht | 1,648 | Medium vehicle turning right |
| MedVeh-IncatRht | 1,935 | Medium vehicle indicating right |
| Cyc-MovTow | 3,233 | Cyclist approaching ego vehicle |
| MedVeh-IncatLft | 3,365 | Medium vehicle indicating left |

**Key observation:** Emergency vehicles (`EmVeh`) and cyclists (`Cyc`) are systematically underrepresented — exactly the agents whose behaviour requires the most nuanced interpretation.

---

### 3.3 Corner Cases — Triplet Level

**12 of 86 valid triplets** are corner cases (threshold: ≤ 1,312 instances):

| Combination | Count | Notes |
|-------------|-------|-------|
| Cyc-MovTow-RhtPav | 61 | **Near zero-shot** — cyclist approaching on right pavement |
| Ped-MovTow-Jun | 633 | Pedestrian approaching at junction |
| Bus-HazLit-OutgoLane | 658 | Bus with hazard lights on outgoing lane |
| MedVeh-IncatLft-IncomLane | 691 | Vehicle indicating left on incoming lane |
| Car-HazLit-IncomLane | 730 | Car with hazard lights in opposing traffic |
| Cyc-MovAway-RhtPav | 810 | Cyclist receding on right pavement |
| Ped-Mov-OutgoLane | 849 | Pedestrian moving in the road (not pavement) |
| MedVeh-IncatRht-Jun | 907 | Vehicle indicating right at junction |
| Ped-MovAway-Jun | 1,145 | Pedestrian leaving junction area |
| Car-MovTow-VehLane | 1,254 | Car approaching in same lane |
| Cyc-MovTow-IncomLane | 1,277 | Cyclist approaching in opposing lane |
| Ped-PushObj-LftPav | 1,312 | Pedestrian pushing object on left pavement |

**`Cyc-MovTow-RhtPav` with only 61 instances is effectively a zero-shot scenario** for any model trained on this dataset without additional reasoning capability.

---

### 3.4 Advisor Scenario Mapping

| # | Scenario | Closest Label | Match | Frequency |
|---|----------|--------------|-------|-----------|
| 1 | School bus discharging children | Ped-Stop-BusStop | direct | 8,559 |
| 2 | Cyclist signalling left turn | Cyc-MovTow-IncomLane | **partial** | 1,277 |
| 3 | Pedestrian on phone, unlikely to cross | Ped-Stop-LftPav | direct | 72,601 |
| 4 | Pedestrian looking at road, may cross | Ped-XingFmLft | direct | 35,471 |
| 5 | Car brakes suddenly | Car-Brake-VehLane | direct | 53,210 |
| 6 | Stiff rule-based stop (plastic bag) | LarVeh-Stop-VehLane | **partial** | 1,644 |
| 7 | Ambulance with sirens, must yield | Car-MovTow-IncomLane | direct | 286,573 |
| 8 | Road blocked, human gesturing left | Ped-XingFmLft-Jun | **partial** | 32,292 |
| 9 | Long-tail school bus scenario | Ped-Stop-BusStop | direct | 8,559 |

**6 of 9 scenarios are direct matches** — the physical event is representable in ROAD-Waymo labels.  
**3 of 9 are partial matches** — the label captures *what* is happening but not *why* or *what the AV should do*.

---

## 4. Key Findings

### Finding 1: The label vocabulary is necessary but not sufficient

ROAD-Waymo's 251 logical constraints produce a clean, validated label set. However, labels describe *observable states* (`Cyc-IncatLft`) not *interpretable intent* ("the cyclist intends to turn left — slow down"). A standard detection model trained on these labels learns to classify what it sees; it does not reason about consequence.

### Finding 2: Emergency and non-car agents are critically underrepresented

`EmVeh-Stop` appears only 1,069 times in 3.3M boxes. A standard 3D-RetinaNet trained on this distribution will systematically under-detect and misclassify emergency vehicle behaviour. This is not a data collection failure — it reflects real-world base rates. It is, however, exactly the scenario where misclassification has the highest safety cost.

### Finding 3: Near-zero-shot scenarios exist within valid labels

`Cyc-MovTow-RhtPav` (61 instances) is a logically valid and safety-relevant scenario — a cyclist on the pavement approaching the vehicle — that no current model can be expected to handle reliably from data alone.

### Finding 4: Intent cannot be encoded in the current label structure

Scenarios 2, 6, and 8 are partial matches because the label captures the physical observation but not the semantic interpretation. For example:
- Scenario 2 ("cyclist arm signalling left"): the label `Cyc-MovTow-IncomLane` captures position and direction, but not the gesture or its implication for AV speed.
- Scenario 7 ("ambulance requiring yield"): maps to `Car-MovTow-IncomLane` because there is no `EmVeh`-specific label for the siren+yield obligation.
- Scenario 8 ("road blocked, human gesturing"): maps to a crossing label, missing the active human instruction to deviate from the GPS path.

**These three scenarios represent the core motivation for VLM-based reasoning:** the visual observation is clear, but understanding what the AV should *do* requires contextual, social, or legal knowledge that cannot be encoded in bounding box labels.

---

## 5. Implications for Research Direction

| Research Step | Justified By |
|---------------|-------------|
| **Approach 1** (3D-RetinaNet baseline): Establish detection performance on all 49 duplexes and 86 triplets | Section 3.1 — establishes the performance ceiling of label-only models |
| **Approach 2** (T-norm constraint loss): Improve compliance with valid combination rules | Section 3.2/3.3 — 7 duplex and 12 triplet corner cases are precisely where constraint violations are most likely |
| **Approach 3** (Constrained VLM): Generate natural language reasoning alongside structured labels | Section 3.4 — 3 of 9 advisor scenarios require intent reasoning beyond any structured label; the VLM bridges this gap |

The analysis confirms that ROAD-Waymo provides a strong structural foundation but has an inherent expressiveness ceiling. Approach 3 is not a redundant upgrade to Approach 1 — it addresses a qualitatively different class of problem.

---

## 6. Data Availability

All analysis scripts and output CSVs are available in `/data/repos/ROAD_Reason/analysis/`:

- `count_combinations.py` — frequency counter (configurable thresholds)
- `map_examples.py` — advisor scenario mapper
- `combination_counts.csv` — full frequency table for all 135 valid combinations
- `example_mapping.csv` — full mapping of 9 advisor scenarios to ROAD-Waymo labels

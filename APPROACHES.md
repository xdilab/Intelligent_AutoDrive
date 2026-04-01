# Research Approaches: Logic-Constrained Scene Reasoning for Autonomous Driving

## Context

This document outlines the planned research directions for PhD work on pedestrian/driver intent prediction,
using the ROAD-Waymo (ROAD++) dataset as the primary benchmark. The overarching goal is to produce
**text-based reasoning about scene intent** (e.g., "pedestrian is waiting at a junction — likely to cross
once the light changes") that is grounded in propositional logic constraints encoded in the dataset.

Supervisor: Dr. Moradi

---

## Constraint Background

ROAD-Waymo encodes implicit constraints via its compositional label structure:
- **39 valid duplexes** (agent + action) out of 242 possible combinations
- **68 valid triplets** (agent + action + location) out of 3,872 possible combinations
- Stored in `duplex_childs` and `triplet_childs` in `road_waymo_trainval_v1.0.json`

The ROAD-Waymo paper introduces a **neuro-symbolic baseline** integrating these requirements into the loss
via **t-norms** (differentiable algebraic operations mapping Boolean logic to [0,1]), following ROAD-R [Marconato et al., 2022].

---

## Approach 1: 3D-RetinaNet Baseline (Replication)

**Status:** Starting point — not novel but required for comparison.

**Method:**
- Kinetics-pretrained 3D-RetinaNet with anchor-based tube detection
- Multi-label classification heads for agent, action, location, duplex, triplet
- Evaluated on frame-mAP and video-mAP per label type

**Purpose:** Establish baseline numbers on ROAD-Waymo before adding constraints or new architectures.

**Repo:** https://github.com/salmank255/ROAD_plus_plus_Baseline

---

## Approach 2: Neuro-Symbolic 3D-RetinaNet (T-norm Constraint Loss)

**Status:** Described in ROAD-Waymo paper — replication only, not a novel contribution.

**Method:**
- Same as Approach 1 but adds a constraint violation loss term
- T-norm (e.g., Łukasiewicz) maps Boolean requirements to differentiable penalties:
  ```
  # "LarVeh cannot perform Xing" — penalise co-prediction
  violation = max(0, p(LarVeh) + p(Xing) - 1)
  ```
- Constraint set derived from `duplex_childs` / `triplet_childs` validity mappings
- Loss = detection loss + λ · constraint violation loss

**Purpose:** Confirm constraint loss improves compliance on ROAD-Waymo (replicating paper results).

---

## Approach 3: Logic-Constrained Generative VLM with Text Reasoning (NOVEL)

**Status:** Primary novel contribution. No existing work applies t-norm constraints to a generative VLM.

**Method:**
- Fine-tune a generative VLM (LLaVA or InstructBLIP) on ROAD-Waymo video clips
- Model produces two outputs jointly:
  1. **Structured labels** — `{agent, action, location, intent}` (JSON-format)
  2. **Natural language reasoning** — e.g., "Pedestrian is at a junction waiting to cross. Red traffic light suggests they will wait."
- Apply t-norm constraint loss to the structured label output logits using ROAD-Waymo's `triplet_childs`
- Text reasoning is trained to be consistent with the constrained structured predictions

**Why novel:**
- No existing work applies neuro-symbolic t-norm constraints to a generative VLM
- No existing work produces constrained natural language intent reasoning from ROAD-Waymo
- Directly addresses explainability requirements for AV decision-making

**Evaluation:**
- Detection accuracy (frame-mAP, video-mAP) vs Approach 1 & 2
- Constraint violation rate (% predictions violating triplet validity)
- Text reasoning quality (BLEU/ROUGE vs human-written rationales, or LLM-as-judge)

**Candidate models:** LLaVA-1.6, InstructBLIP, VL-JEPA

---

## Approach 4: V-JEPA 2 + Intent Head (World Model Backbone)

**Status:** Novel application — no published work applies JEPA to pedestrian crossing intent.

**Method:**
- Use pretrained V-JEPA 2 encoder (frozen or lightly fine-tuned) as spatiotemporal feature extractor
- Add lightweight intent prediction head (MLP or Transformer) on top of JEPA features
- Train on ROAD-Waymo action/location labels and binary crossing intent
- V-JEPA learns physically predictable scene dynamics (motion, causality) without pixel reconstruction

**Why relevant:**
- V-JEPA 2 is SOTA on action *anticipation* (Epic-Kitchens-100) — closest published analog to intent
- Drive-JEPA (arXiv:2601.22032) demonstrates the V-JEPA → AV pipeline on trajectory planning
- Pretrained weights are public; no pretraining compute required

**Extension:** Add t-norm constraint loss to the intent head outputs (combining Approaches 2 + 4).

**Key papers:**
- V-JEPA 2: arXiv:2506.09985
- VL-JEPA: arXiv:2512.10942
- Drive-JEPA: arXiv:2601.22032

---

## Approach 5: LeWM World Model for Future Scene Prediction

**Status:** Exploratory — adaptation to driving video is novel, current benchmarks are robotics only.

**Method:**
- LeWM (arXiv:2603.19312) — first JEPA to train end-to-end from raw pixels on a single GPU
- ~15M parameters, trains in hours on a single RTX 3090/4090
- Adapt to ROAD-Waymo clips: predict future latent scene states from current observations
- Apply t-norm constraint loss across predicted future frames (future states must also satisfy logic requirements)
- Use surprise detection (anomalous latent predictions) for unusual pedestrian behaviour flagging

**Why relevant:**
- Workstation-feasible (single GPU, hours to train) — no HPC cluster required
- World model that predicts future scene states is a natural fit for intent prediction
- t-norm constraints on future predictions is unexplored

**GitHub:** https://github.com/lucas-maes/le-wm

---

## Approach 6: JEPA + VLM Hybrid with Constrained Text Output (Long-term)

**Method:**
- VL-JEPA (arXiv:2512.10942): predicts continuous text embeddings rather than generating tokens autoregressively
- 50% fewer parameters than standard VLMs with equivalent performance
- Fine-tune on ROAD-Waymo with t-norm constraint loss on label outputs + text reasoning supervision
- Combines the efficiency of JEPA with the language output required by Dr. Moradi

**Why novel:** No existing work combines JEPA-style predictive learning with neuro-symbolic constraints and natural language reasoning output.

---

## Summary Table

| Approach | Model | Novel? | Workstation? | Text Output? | Constraints? |
|----------|-------|--------|-------------|--------------|-------------|
| 1. 3D-RetinaNet baseline | 3D-RetinaNet | No (replication) | Yes | No | No |
| 2. Neuro-symbolic RetinaNet | 3D-RetinaNet + t-norm | No (in paper) | Yes | No | Yes |
| 3. Constrained VLM reasoning | LLaVA / InstructBLIP | **Yes** | Partial | **Yes** | **Yes** |
| 4. V-JEPA 2 intent head | V-JEPA 2 + MLP | **Yes** | Yes | No | Optional |
| 5. LeWM scene prediction | LeWM | **Yes** | **Yes** | No | Optional |
| 6. VL-JEPA constrained | VL-JEPA + t-norm | **Yes** | Partial | **Yes** | **Yes** |

**Primary novel contribution:** Approach 3 — logic-constrained generative VLM with natural language scene reasoning.
**Secondary contributions:** Approaches 4, 5 (world model / JEPA for intent).
**Starting point:** Approach 1 (baseline replication, in progress).

---

## Dataset

- ROAD-Waymo (ROAD++): 798 annotated videos, 153,534 frames, 3.3M bounding boxes
- 600 train / 198 val / 202 test (no labels)
- Annotation file: `road_waymo_trainval_v1.0.json`
- Baseline code: https://github.com/salmank255/ROAD_plus_plus_Baseline

## Key References

- ROAD dataset & 3D-RetinaNet: Singh et al., IEEE TPAMI 2022
- ROAD++ / ROAD-Waymo: Salmank et al., 2023
- ROAD-R (t-norm constraints): Marconato et al., arXiv:2210.01597
- V-JEPA 2: arXiv:2506.09985
- VL-JEPA: arXiv:2512.10942
- Drive-JEPA: arXiv:2601.22032
- LeWM: arXiv:2603.19312

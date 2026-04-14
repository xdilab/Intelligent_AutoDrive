# Intelligent_AutoDrive

Logic-constrained scene understanding and reasoning for autonomous driving, built on the ROAD-Waymo (ROAD++) dataset. Primary contribution: a generative VLM architecture with neuro-symbolic t-norm constraints that produces structured scene labels and natural language reasoning grounded in the dataset's compositional label structure.

Supervisor: Dr. Moradi | NC A&T State University

---

## Research Goal

ROAD-Waymo labels capture *what* happens (e.g., `Ped-Wait2X-RhtPav`) but not *why*. Identical triplets arise from different causal origins — a pedestrian waiting at a junction with genuine crossing intent vs. one simply pausing. This project develops models that distinguish causal origins and produce grounded natural language explanations alongside structured predictions.

---

## Research Roadmap

| Approach | Model | Novel? | Status |
|----------|-------|--------|--------|
| 1. 3D-RetinaNet baseline | 3D-RetinaNet | No | Replication |
| 2. Neuro-symbolic RetinaNet | 3D-RetinaNet + t-norm | No (in paper) | Replication |
| **3. Qwen2.5-VL multi-task** | **Qwen2.5-VL-7B + task heads** | **Yes** | **Active** |
| **4. Constrained VLM reasoning** | **OpenMixer + DSDAG + VLT** | **Yes** | **Primary contribution** |
| 5. V-JEPA 2 intent head | V-JEPA 2 + MLP | Yes | Planned |
| 6. LeWM scene prediction | LeWM (~15M params) | Yes | Exploratory |
| 7. JEPA + VLM hybrid | VL-JEPA + t-norm | Yes | Long-term |

See [`docs/`](docs/) for public-facing documentation.

---

## Approach 3 (Active): Qwen2.5-VL Multi-Task Fine-Tuning

Shared Qwen2.5-VL-7B backbone fine-tuned sequentially on three driving datasets with task-specific I/O modules:

| Task | Dataset | Input | Output | Loss |
|------|---------|-------|--------|------|
| T1 | BDD-X | Video clip | Action text + justification text | Cross-entropy |
| T2 | ROAD-R | Video clip | Tube detections + triplet labels | L_det + L_cls + λ·L_tnorm |
| T3 | CoVLA | Video frame | plain_caption + risk text + 10×3 trajectory | L_lm + λ·L_traj |

Training order (confirmed): ROAD-R → BDD-X → CoVLA → joint.

**Experiment 1 (ROAD-R):** ViT encoder, video frames + ROAD-Waymo labels as input, triplet + duplex combinations as output, Łukasiewicz t-norm loss.

---

## Approach 4 (Primary): OpenMixer + DSDAG + VLT

```
Video clip → CLIP-ViP (frozen) → OpenMixer backbone
                                       │
                   ┌───────────────────┴───────────────────┐
                   │                                       │
         Structured head                         DSDAG causal head
         raw triplet logits                      VLT → reasoning embedding
                   │                                       │
                   └──────────── f(reasoning) ────────────┘
                                      │
                             reweighted triplet logits
                                      │
                            T-norm constraint loss
                                      │
                         Triplet mAP  +  "Why" reasoning
```

**Key components:**
- **OpenMixer** — DETR-style open-vocabulary detector on frozen CLIP-ViP features
- **DSDAG** — causal DAG encoding hidden danger state W that distinguishes causal origins of identical surface actions
- **VLT** — Vision-Language Transformer with sparsity loss producing grounded reasoning embedding
- **Logit reweighting** — `L_final = L_raw ⊙ f(r)`: causal reasoning gates structured predictions end-to-end

---

## Constraint Background

ROAD-Waymo encodes implicit constraints via its compositional label structure:
- **49 valid duplexes** (agent + action) out of 220 possible combinations
- **86 valid triplets** (agent + action + location) out of 3,520 possible combinations

**T-norm constraint loss (Łukasiewicz):**
```python
violation = max(0, p(agent_i) + p(action_j) - 1)  # for each invalid pair
L_tnorm = sum(violation over all invalid combinations)
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate road_reason
```

---

## Baselines

Three SmolVLM-500M-Instruct zero-shot baselines on ROAD-Waymo val frames.

### Zero-shot (flat label lists)

```bash
python baseline/smolvlm_inference.py
# options: --n_videos 20 --frames_per_video 10 --model HuggingFaceTB/SmolVLM-Instruct
```

### Constraint-aware (49 valid duplexes + 86 valid triplets injected into prompt)

```bash
python baseline/smolvlm_constrained.py
```

### GT-conditioned reasoning

Feeds ground-truth triplets alongside the image; model must explain why and predict agent intent. Isolates reasoning capability from detection.

```bash
python baseline/smolvlm_gt_reasoning.py
# --prefer_ped (default) biases frame selection toward pedestrian scenes
```

### Evaluate

```bash
python baseline/eval_preds.py --preds baseline/results/smolvlm_preds.json
python baseline/eval_preds.py --preds baseline/results/constrained_preds.json
```

---

## Repository Structure

```
Intelligent_AutoDrive/
├── baseline/               # SmolVLM zero-shot baselines + evaluation
├── analysis/               # Dataset statistics and visualization
├── datasets/               # Dataset utilities and exploration scripts
├── docs/                   # Public-facing documentation
├── viz/                    # Visualization outputs
├── tnorm_loss.py           # T-norm (Łukasiewicz) constraint loss
├── convert_waymo_to_coco.py
├── extract_videos2jpgs.py
└── environment.yml
```

> `papers/` and `working_docs/` are local only (gitignored).

---

## Key Numbers

| Dataset | Videos | Annotated Frames | Tubes | Boxes |
|---------|--------|-----------------|-------|-------|
| ROAD-Waymo | 798+202 test | 153,534 | 41,935 | 3.3M |

Agent distribution: Car 2.2M | Ped 712K | MedVeh 239K | TL 58K | LarVeh 40K

---

## Related Repos

- [xdilab/AutoDrive_Perception](https://github.com/xdilab/AutoDrive_Perception) — ROS 1 real-time perception pipeline (YOLOv10 + EfficientNet classifiers)

## Key References

- ROAD++ / ROAD-Waymo: Salmank et al., 2023
- ROAD-R (t-norm constraints): Marconato et al., arXiv:2210.01597
- OpenMixer: Bao et al., WACV 2025
- BDD-X: Kim & Mooney, ECCV 2018
- CoVLA: Arai et al., arXiv:2408.10845

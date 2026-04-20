# ROAD-Waymo Dataset Reference

**For:** Dr. Moradi meeting — implementation-level detail  
**Sources:** `road_waymo_trainval_v1.1.json`, `tnorm_loss.py`, `experiments/exp1_road_r/`  
**Paper:** Salmank et al., "ROAD-Waymo: A Large-Scale Dataset for Road Agent Activity Detection," 2024

---

## Dataset Summary

| Property | Value | Source |
|----------|-------|--------|
| Total train videos | 798 | JSON (paper claims 1,000 — do not use paper number) |
| Test videos (unannotated) | 202 | JSON |
| Annotated frames | 153,534 | JSON (paper claims 198K) |
| Agent tubes | 41,935 | JSON (paper claims 54K) |
| Total annotation boxes | 3,304,353 | JSON (paper claims 3.9M) |
| Frame resolution | 1920 × 1280 | JSON per-frame width/height |
| Annotation file | `road_waymo_trainval_v1.1.json` | `/data/datasets/road_waymo/` |
| Frame images | JPEG, 1 frame per file | `/data/datasets/road_waymo/rgb-images/` |

> **Note:** All numbers above are verified from the JSON. The published paper overcounts. Always cite JSON-derived numbers. Source: `wiki/datasets/road-plusplus.md`

---

## Agent Distribution (10 classes)

| Index | Label | Instances | % of total |
|-------|-------|-----------|------------|
| 0 | Ped | 712,640 | 21.6% |
| 1 | Car | 2,197,049 | 66.5% |
| 2 | Cyc | 11,894 | 0.4% |
| 3 | Mobike | 8,002 | 0.2% |
| 4 | SmalVeh | 2,755 | 0.1% |
| 5 | MedVeh | 238,522 | 7.2% |
| 6 | LarVeh | 39,889 | 1.2% |
| 7 | Bus | 33,904 | 1.0% |
| 8 | EmVeh | 2,139 | 0.1% |
| 9 | TL | 57,722 | 1.7% |

Source: `wiki/datasets/road-plusplus.md`

---

## Action Classes (22 classes)

| Index | Label | Meaning |
|-------|-------|---------|
| 0 | Red | Traffic light: red |
| 1 | Amber | Traffic light: amber |
| 2 | Green | Traffic light: green |
| 3 | MovAway | Moving away from camera |
| 4 | MovTow | Moving toward camera |
| 5 | Mov | Moving (lateral) |
| 6 | Rev | Reversing |
| 7 | Brake | Braking |
| 8 | Stop | Stopped |
| 9 | IncatLft | Left indicator active |
| 10 | IncatRht | Right indicator active |
| 11 | HazLit | Hazard lights active |
| 12 | TurLft | Turning left |
| 13 | TurRht | Turning right |
| 14 | MovRht | Moving right (lane change) |
| 15 | MovLft | Moving left (lane change) |
| 16 | Ovtak | Overtaking |
| 17 | Wait2X | Waiting to cross |
| 18 | XingFmLft | Crossing from left |
| 19 | XingFmRht | Crossing from right |
| 20 | Xing | Crossing |
| 21 | PushObj | Pushing an object |

Source: `analysis/stats_full.json` — `action_labels` array

---

## Location Classes (16 classes)

| Index | Label | Meaning |
|-------|-------|---------|
| 0 | VehLane | General vehicle lane |
| 1 | OutgoLane | Outgoing lane (same direction as ego) |
| 2 | OutgoCycLane | Outgoing cycle lane |
| 3 | OutgoBusLane | Outgoing bus lane |
| 4 | IncomLane | Incoming lane (opposite to ego) |
| 5 | IncomCycLane | Incoming cycle lane |
| 6 | IncomBusLane | Incoming bus lane |
| 7 | Pav | Pavement (generic) |
| 8 | LftPav | Left pavement |
| 9 | RhtPav | Right pavement |
| 10 | Jun | Junction |
| 11 | xing | Pedestrian crossing |
| 12 | BusStop | Bus stop |
| 13 | parking | Parking area |
| 14 | LftParking | Left parking |
| 15 | rightParking | Right parking |

Source: `analysis/stats_full.json` — `loc_labels` array

---

## Valid Combinations: Duplexes and Triplets

### Counts
| Level | Valid | Possible | Invalid |
|-------|-------|----------|---------|
| Duplexes (agent+action) | 49 | 220 (10×22) | 171 |
| Triplets (agent+action+loc) | 86 | 3,520 (10×22×16) | 3,434 |

### All 49 valid duplexes
See `examples/duplex-triplet-labels.json` for the full array.

Grouped by agent:
- **Ped (9):** MovAway, MovTow, Mov, Stop, Wait2X, XingFmLft, XingFmRht, Xing, PushObj
- **Car (13):** MovAway, MovTow, Brake, Stop, IncatLft, IncatRht, HazLit, TurLft, TurRht, MovRht, MovLft, XingFmLft, XingFmRht
- **Cyc (3):** MovAway, MovTow, Stop
- **Mobike (1):** Stop
- **SmalVeh (0):** none
- **MedVeh (10):** MovAway, MovTow, Brake, Stop, IncatLft, IncatRht, HazLit, TurRht, XingFmLft, XingFmRht
- **LarVeh (4):** MovAway, MovTow, Stop, HazLit
- **Bus (5):** MovAway, MovTow, Brake, Stop, HazLit
- **EmVeh (1):** Stop
- **TL (3):** Red, Amber, Green

### All 86 valid triplets
See `examples/duplex-triplet-labels.json` for the full array.

Examples:
- `Ped-Stop-RhtPav` (idx 12) — pedestrian stopped on right pavement
- `Car-MovAway-OutgoLane` (idx 23) — car driving away in outgoing lane
- `Car-MovTow-Jun` (idx 27) — car approaching junction
- `TL-Red` has no location (TLs don't use location labels — hence no TL triplets)

Source: `analysis/stats_full.json` — `duplex_labels`, `triplet_labels`, `duplex_childs`, `triplet_childs`

---

## How Constraints Are Stored in the JSON

```python
# Root-level keys in road_waymo_trainval_v1.1.json:
data["agent_labels"]    # list[10] — agent class names
data["action_labels"]   # list[22] — action class names
data["loc_labels"]      # list[16] — location class names
data["duplex_labels"]   # list[49] — valid duplex names ("Ped-Stop", etc.)
data["triplet_labels"]  # list[86] — valid triplet names ("Ped-Stop-RhtPav", etc.)
data["duplex_childs"]   # list[49×2] — valid [agent_idx, action_idx] pairs
data["triplet_childs"]  # list[86×3] — valid [agent_idx, action_idx, loc_idx] triples
```

The `TNormConstraintLoss.__init__` inverts these to get the **invalid** set:
```python
# tnorm_loss.py:56-63
valid_d = set(map(tuple, duplex_childs))
invalid_d = [(i, j) for i in range(n_agents) for j in range(n_actions)
             if (i, j) not in valid_d]
# → 171 pairs, stored as GPU buffer (register_buffer)
```

---

## T-Norm Loss: Implementation

### Two t-norm functions (`tnorm_loss.py:29-34`)

```python
def _godel(p_a, p_b):
    return torch.min(p_a, p_b)          # T(a,b) = min(a,b)

def _lukasiewicz(p_a, p_b):
    return torch.clamp(p_a + p_b - 1.0, min=0.0)  # T(a,b) = max(0, a+b-1)
```

### Forward pass logic (`tnorm_loss.py:75-102`)

```python
def forward(self, preds):  # preds: [N, 49]
    loss = preds.new_zeros(1)

    # Duplex violations: 171 invalid (agent, action) pairs
    p_agent  = preds[:, AGENT_OFFSET + inv_d[:, 0]]   # [N, 171]
    p_action = preds[:, ACTION_OFFSET + inv_d[:, 1]]  # [N, 171]
    loss += violation_fn(p_agent, p_action).mean()

    # Triplet violations: 3434 invalid (agent, action, loc) triples
    # t-norm applied TWICE — nested composition
    p_agent  = preds[:, AGENT_OFFSET + inv_t[:, 0]]
    p_action = preds[:, ACTION_OFFSET + inv_t[:, 1]]
    p_loc    = preds[:, LOC_OFFSET + inv_t[:, 2]]
    loss += violation_fn(violation_fn(p_agent, p_action), p_loc).mean()

    return self.lam * loss   # lam = LAMBDA_TNORM = 0.1
```

### How ROADLoss calls it (`losses.py:109-114`)

```python
agentness = torch.ones(N, 1, ...)         # all GT boxes are positive → 1.0
flat = torch.cat([agentness, preds["agent"], preds["action"], preds["loc"]], dim=1)
# flat shape: [N, 1+10+22+16] = [N, 49]
L_tnorm = self.tnorm_loss(flat)           # scalar
```

Total loss: `L_total = L_cls + L_tnorm`

---

## Paper vs. Implementation Differences

| Aspect | Paper description | Implementation |
|--------|-------------------|----------------|
| T-norm type | Łukasiewicz is described; Table 7 says Gödel is best | `config.py:54` uses `"lukasiewicz"` with comment "per email; paper Table 7 says godel is best — compare both" |
| Loss scaling | Basic violation sum | Multiplied by `LAMBDA_TNORM = 0.1` (configurable λ) |
| Triplet computation | Single constraint per triplet | Nested t-norm: `T(T(agent, action), location)` |
| Duplex vs triplet | Both described | Both implemented as separate `register_buffer` arrays |
| Agentness | Not mentioned | Hardcoded to 1.0 (all GT anchors are positive by construction) |

---

## Input Format (Training)

| Property | Value | Source |
|----------|-------|--------|
| Frames per clip | 8 (`CLIP_LEN`) | `config.py:24` |
| Frame sampling | Consecutive annotated frames | `dataset.py:133-148` |
| Clip stride | 16 frames (`CLIP_STRIDE`) | `config.py:32` |
| Image size (processed) | 448×448 | `config.py:28-29` |
| Image format | RGB JPEG | `dataset.py:248` |
| Box format | Normalized [x1,y1,x2,y2] ∈ [0,1] | `dataset.py:179` |
| Labels | Multi-hot tensors (5 levels) | `dataset.py:186-208` |

### Per-frame target tensors

| Tensor | Shape | Encoding |
|--------|-------|----------|
| `boxes` | [n, 4] | Normalized [x1,y1,x2,y2] |
| `agent` | [n, 10] | Multi-hot, from `agent_ids` |
| `action` | [n, 22] | Multi-hot, from `action_ids` |
| `loc` | [n, 16] | Multi-hot, from `loc_ids` |
| `duplex` | [n, 49] | Multi-hot, from `duplex_ids` |
| `triplet` | [n, 86] | Multi-hot, reconstructed by name lookup (NOT from `triplet_ids` in JSON) |

> **Important:** `triplet_ids` in the raw JSON are NOT 0-indexed into `triplet_labels`. The dataset loader ignores them and reconstructs triplet targets via `_triplet_lookup`. See `dataset.py:14-17` for the explanation and `dataset.py:197-208` for the reconstruction.

---

## Output Format

The model outputs 5 sigmoid-activated heads:

| Head | Shape | Evaluation |
|------|-------|------------|
| `agent` | [N, 10] | mAP@0.5, per-class P/R/F1 |
| `action` | [N, 22] | mAP@0.5, per-class P/R/F1 |
| `loc` | [N, 16] | mAP@0.5, per-class P/R/F1 |
| `duplex` (DoubleX) | [N, 49] | mAP@0.5 |
| `triplet` (TripleX) | [N, 86] | mAP@0.5 |

Additional metric: **constraint violation rate** — fraction of predictions where an invalid (agent, action) pair is simultaneously ≥ 0.5 threshold. See `eval.py:93-127`.

**Exp1 results (epoch 6, GT boxes, no detection head):**
- Agent mAP: 0.357, Action mAP: varies (Stop F1=0.731, rare actions ≈0.000), Duplex mAP: 0.123
- Constraint violation rate: 0.000211 (0.021%) — t-norm loss effective
- Source: `experiments/exp1_road_r/logs/eval_results.json`, `wiki/findings/exp1-vs-retinanet-baseline.md`

---

## Data Flow

```
road_waymo_trainval_v1.1.json
         │
         ▼
ROADWaymoDataset (dataset.py)
  └─ clips = (video_name, [fid×8])
         │
         ▼
__getitem__(idx)
  ├─ pil_frames: List[PIL.Image] × 8   (1920×1280 → 448×448)
  └─ frame_targets: List[Optional[dict]] × 8
         │
         ▼
Qwen2.5-VL-7B ViT encoder (frozen in Exp1)
  └─ ROI-pool on GT boxes → [N, VIT_DIM=3584] features
         │
         ▼
Five classification heads (agent/action/loc/duplex/triplet)
  └─ Each: Linear(3584 → n_classes) + sigmoid
         │
         ▼
ROADLoss = ROADClassificationLoss (BCE×5) + TNormConstraintLoss (λ=0.1)
         │
         ▼
Eval: mAP@0.5 per head + constraint violation rate
```

---

## Local Paths

| Content | Path |
|---------|------|
| Annotation JSON | `/data/datasets/road_waymo/road_waymo_trainval_v1.1.json` |
| RGB frames | `/data/datasets/road_waymo/rgb-images/<video>/NNNNN.jpg` |
| Stats JSON | `analysis/stats_full.json` |
| T-norm loss | `tnorm_loss.py` |
| Training config | `experiments/exp1_road_r/config.py` |
| Dataset loader | `experiments/exp1_road_r/dataset.py` |
| Loss functions | `experiments/exp1_road_r/losses.py` |
| Eval script | `experiments/exp1_road_r/eval.py` |

---

## Examples in This Folder

| File | Contents |
|------|----------|
| `examples/annotation-frame.json` | Real annotated frame from train_00407 with decoded labels |
| `examples/duplex-triplet-labels.json` | All 49 duplexes and 86 triplets as JSON arrays |
| `examples/constraint-violation.md` | Valid vs. invalid combinations explained with examples |
| `examples/tnorm-forward.py` | T-norm forward pass code with annotations |
| `examples/clip-target-structure.md` | Full tensor layout for one training clip |

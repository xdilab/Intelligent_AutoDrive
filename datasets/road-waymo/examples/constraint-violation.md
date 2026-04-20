# ROAD-Waymo: Valid vs. Invalid Combinations

Source: `analysis/stats_full.json` (duplex_labels, triplet_labels), `tnorm_loss.py`

---

## What a "Valid Combination" Means

A **duplex** is a valid (agent, action) pair that can realistically occur in driving.
A **triplet** is a valid (agent, action, location) triple.
All valid combinations are listed in `duplex_labels` (49) and `triplet_labels` (86).
All OTHER combinations are **invalid** by exclusion — the t-norm loss penalizes
their co-prediction.

---

## Duplex Examples

### Valid duplexes (agent can perform this action)
| Duplex | Why valid |
|--------|-----------|
| `Ped-Stop` | Pedestrian can stop on pavement |
| `Car-IncatLft` | Car can signal left turn |
| `TL-Red` | Traffic light can be red |
| `Bus-HazLit` | Bus can activate hazard lights |
| `Ped-Xing` | Pedestrian can cross the road |

### Invalid duplexes (physically/semantically impossible)
| Pair | Why invalid |
|------|-------------|
| `Ped-Red` | Pedestrians don't display red lights |
| `TL-MovAway` | Traffic lights don't move away |
| `Car-PushObj` | Cars don't push objects |
| `Ped-IncatLft` | Pedestrians don't use indicator signals |
| `TL-Brake` | Traffic lights don't brake |

**Count:** 220 possible pairs (10 × 22) − 49 valid = **171 invalid duplexes**

---

## Triplet Examples

### Valid triplets (agent + action + realistic location)
| Triplet | Why valid |
|---------|-----------|
| `Ped-Stop-RhtPav` | Pedestrian standing still on right pavement |
| `Car-MovAway-OutgoLane` | Car driving away in outgoing lane |
| `Car-MovTow-Jun` | Car approaching a junction |
| `Bus-Stop-BusStop` | Bus stopped at a bus stop (not in list — see below) |
| `Ped-Xing-Jun` | Pedestrian crossing at junction |

### Invalid triplets (valid duplex but wrong location)
| Triplet | Why invalid |
|---------|-------------|
| `Ped-Stop-VehLane` | Not in triplet_labels: pedestrian stopping in vehicle lane is excluded |
| `Car-MovAway-LftPav` | Cars don't drive on pavements |
| `TL-Red-IncomLane` | TLs don't have lane locations |
| `Bus-Stop-BusStop` | Not in triplet_labels: despite being intuitive, this triplet is absent from annotations |

**Count:** 3520 possible triples (10 × 22 × 16) − 86 valid = **3434 invalid triplets**

---

## How Constraints Are Stored in the Dataset JSON

```python
# road_waymo_trainval_v1.1.json root keys:
data["duplex_childs"]   # list of [agent_idx, action_idx]  — 49 valid duplexes
data["triplet_childs"]  # list of [agent_idx, action_idx, loc_idx] — 86 valid triplets

# Example duplex_childs entries (first 3):
# [[6, 3], [6, 4], [6, 7]]
# → [LarVeh-MovAway, LarVeh-MovTow, LarVeh-Brake]  (agent idx 6 = LarVeh)

# Example triplet_childs entries (first 3):
# [[6, 4, 3], [6, 4, 8], [6, 7, 3]]
# → [LarVeh-MovTow-OutgoBusLane, LarVeh-MovTow-LftPav, LarVeh-Brake-OutgoBusLane]
```

TNormConstraintLoss (tnorm_loss.py:56–63) inverts these to build the invalid set:
```python
valid_d = set(map(tuple, duplex_childs))
invalid_d = [(i, j) for i in range(n_agents) for j in range(n_actions)
             if (i, j) not in valid_d]
# → 171 invalid (agent, action) pairs stored as register_buffer for GPU efficiency
```

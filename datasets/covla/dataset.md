# CoVLA Dataset Reference

**For:** Dr. Moradi meeting — implementation-level detail  
**Sources:** `/data/datasets/CoVLA/mini/`, `wiki/datasets/covla.md`, `wiki/papers/covla-2025.md`  
**Paper:** Arai et al., "CoVLA: Comprehensive Vision-Language-Action Dataset for Autonomous Driving," arXiv:2408.10845, Turing Inc., 2025

---

## Dataset Summary

| Property | Value | Source |
|----------|-------|--------|
| Total clips | 10,000 (30s each, 20 FPS) | Paper arXiv:2408.10845 |
| Total frames | 6,000,000 | Paper |
| Total hours | 83.3h | Paper |
| Camera resolution | 1928×1208 (H.265) | Verified from mini metadata.json |
| Frame rate | 20 FPS | Verified from mini |
| Collection location | Tokyo, Japan | Paper |
| Train split | 7,000 scenes (70%) | Paper |
| Val split | 1,500 scenes (15%) | Paper |
| Test split | 1,500 scenes (15%) | Paper |
| Mini subset | 50 scenes × 600 frames = 30,000 frames | Verified from mini |
| Local mini path | `/data/datasets/CoVLA/mini/` | Verified |
| Local full path | `/data/datasets/CoVLA/full/` | Download pending |

**Weather distribution (mini, 50 scenes):** sunny 25 · cloudy 21 · rainy 4  
**Road type distribution (mini):** wide road 33 · narrow road 17

---

## File Structure

Each scene produces four JSONL files (one JSON object per frame, concatenated without newlines):

```
CoVLA/mini/
├── index.csv                           ← flat index: scene × frame → all file paths
├── metadata.json                       ← {"image_size": [1928, 1208], "frequency": 20}
├── captions/<scene>.jsonl              ← per-frame captions + scene metadata
├── states/<scene>.jsonl                ← per-frame ego state + trajectory + camera matrices
├── front_car/<scene>.jsonl             ← per-frame leading vehicle detection
├── traffic_lights/<scene>.jsonl        ← per-frame traffic light detections
├── images/<scene>/0000.png ... 0599.png ← 600 PNG frames per scene
└── video_samples/<scene>.mp4           ← 30-second H.265 video clip
```

---

## Annotation Schema

### captions/*.jsonl

One JSON object per frame. All fields verified from mini.

| Field | Type | Description |
|-------|------|-------------|
| `plain_caption` | str | **What ego is doing:** behavior + observable objects. Supervision for Y node. |
| `rich_caption` | str | Extended: `plain_caption` + weather + road type + risk sentence concatenated |
| `risk` | str | **Why the driver should be careful.** Supervision for W node. |
| `risk_correct` | bool | Whether risk description is factually correct (VLM quality flag) |
| `risk_yes_rate` | float | Model confidence in risk description |
| `weather` | str | `sunny` / `cloudy` / `rainy` / `heavy rain` / `night` |
| `weather_rate` | float | Confidence in weather label |
| `road` | str | `wide road` / `narrow road` |
| `road_rate` | float | Confidence in road type |
| `is_tunnel` | bool | In tunnel |
| `is_highway` | bool | On highway |
| `has_pedestrian` | bool | Pedestrian visible in frame |
| `has_carrier_car` | float | Carrier/commercial vehicle probability |

**Example `plain_caption`:**
> "The ego vehicle is moving at a moderate speed and turning left. There is a traffic light near the ego vehicle displaying a green signal."

**Example `risk`:**
> "to pay attention to the traffic light and other vehicles on the road. The driver should also be cautious of the wet road conditions due to the rain"

See `examples/caption-frame.jsonl` for a real entry.

---

### states/*.jsonl

One JSON object per frame. Full ego state + trajectory + camera calibration.

**Ego state fields:**

| Field | Type | Description |
|-------|------|-------------|
| `ego_state.vEgo` | float | Speed m/s |
| `ego_state.vEgoRaw` | float | Raw speed measurement |
| `ego_state.aEgo` | float | Acceleration m/s² |
| `ego_state.steeringAngleDeg` | float | Steering angle (°), positive = left |
| `ego_state.steeringTorque` | float | Torque on steering wheel |
| `ego_state.brake` | float | Brake pedal pressure |
| `ego_state.brakePressed` | bool | Brake active |
| `ego_state.gas` | float | Gas pedal pressure |
| `ego_state.gasPressed` | bool | Gas active |
| `ego_state.gearShifter` | str | `drive` / `reverse` / `park` / `neutral` |
| `ego_state.leftBlinker` | bool | Left turn signal |
| `ego_state.rightBlinker` | bool | Right turn signal |
| `ego_state.orientations_calib` | list[3] | Orientation in calibrated frame |
| `ego_state.orientations_ecef` | list[3] | Orientation in ECEF frame |
| `ego_state.orientations_ned` | list[3] | Orientation in NED frame (yaw usable) |
| `ego_state.positions_ecef` | list[3] | GPS position in ECEF coordinates (m) |
| `ego_state.velocities_calib` | list[3] | 3D velocity vector (calibrated frame) |
| `ego_state.velocities_ecef` | list[3] | 3D velocity vector (ECEF) |
| `ego_state.accelerations_calib` | list[3] | 3D acceleration (calibrated) |
| `ego_state.accelerations_device` | list[3] | 3D acceleration (device frame) |
| `ego_state.angular_velocities_calib` | list[3] | Angular velocity (calibrated) |
| `ego_state.angular_velocities_device` | list[3] | Angular velocity (device) |
| `ego_state.timestamp` | int | Unix timestamp (ms) |

**Trajectory:**

| Field | Type | Description |
|-------|------|-------------|
| `trajectory` | list[60×3] float | Future path in vehicle frame |
| `trajectory_count` | int | Always 60 |

Format: 60 3D points covering ~3 seconds at 20 FPS.  
First point is always `[0, 0, 0]` (current position).  
Axes: x=forward, y=lateral (positive = left), z=vertical.  
Source: arXiv:2408.10845

**Camera calibration:**

| Field | Type | Description |
|-------|------|-------------|
| `intrinsic_matrix` | list[3×3] | Camera intrinsics: fx=fy=2648, cx=964, cy=604 |
| `extrinsic_matrix` | list[4×4] | Camera pose (rotation + translation) |
| `image_path` | str | Relative path to PNG frame |
| `frame_id` | int | Frame index within scene (0–599) |

See `examples/states-frame.jsonl` for a real entry (trajectory truncated to 5 points for readability).

---

### front_car/*.jsonl

Stored as a **JSON list** (not a dict) per frame. One entry per frame, regardless of detection.

| Field | Type | Description |
|-------|------|-------------|
| `frame_id` | int | Frame index |
| `has_lead` | bool | Leading vehicle detected |
| `lead_prob` | float | Detection confidence |
| `lead_x` | float\|null | Longitudinal distance to lead (m) |
| `lead_y` | float\|null | Lateral offset of lead (m) |
| `lead_speed_kmh` | float\|null | Lead vehicle speed (km/h) |
| `lead_a` | float\|null | Lead vehicle acceleration |

See `examples/front-car-frame.jsonl` for a real entry.

---

### traffic_lights/*.jsonl

Stored as a **JSON list** per frame. Empty list `[]` if no traffic lights detected.

| Field | Type | Description |
|-------|------|-------------|
| `index` | int | Detection index (per frame) |
| `class` | str | Signal state: `green` / `red` / `amber` |
| `bbox` | list[4] | [x1, y1, x2, y2] in pixels (1928×1208 coordinate space) |

Detected by OpenLenda-s (mentioned in paper arXiv:2408.10845).  
See `examples/traffic-lights-frame.jsonl` for a real entry.

---

## Caption Generation Pipeline

Source: arXiv:2408.10845, Section 3

1. **Rule-based layer** — extracts factual constraints from CAN bus data (speed, steering, blinkers, braking), GNSS, OpenLenda-s traffic light detections, radar+camera fusion for leading vehicles
2. **VLM layer** — VideoLLaMA2-7B prompted with rule-based constraints; 60-frame window, first+last frames sampled for visual grounding; constraint-augmented prompts suppress hallucination
3. **Output** — 100,000 VLM-generated captions + 6,000,000 rule-augmented captions (rule-based covers all frames; VLM covers a 100K subset)

---

## Relevance to Approach 3 (Qwen2.5-VL Multi-Task)

| CoVLA field | DSDAG node | Supervision signal | Why |
|-------------|------------|--------------------|-----|
| `plain_caption` | Y (action node) | What action is occurring | Behavior description with observable grounding |
| `risk` | W (reason node) | Causal origin motivating action | Explicitly causal "why" content |
| `trajectory` [60×3] | Trajectory head | Waypoint regression | 3s horizon, vehicle-frame coordinates |

Used in **Stage 1 pre-training** of Approach 3 causal head alongside BDD-X.
CoVLA provides scale (6M frames); BDD-X provides quality (human-authored reasoning).
Source: `wiki/methods/qwen25-vl-multitask.md`, `wiki/comparisons/bdd-x-vs-covla.md`

**Domain limitation:** Collection is Tokyo, Japan — different traffic rules, Japanese signage, narrow urban streets. Domain gap to ROAD-Waymo (US) is a known risk. No mitigation strategy documented yet.

---

## Output Format (Model Output for CoVLA Task)

The Approach 3 CoVLA task head produces two outputs:

| Output | Format | Supervision | Metric |
|--------|--------|-------------|--------|
| Caption | Language tokens | Cross-entropy vs. `plain_caption`/`risk` | METEOR / CIDEr / BLEU |
| Trajectory | [60, 3] float tensor | MSE vs. `trajectory` array | ADE (Average Displacement Error), FDE (Final Displacement Error) |

Source: `wiki/methods/qwen25-vl-multitask.md`

---

## Input to Model (Approach 3)

```
Input per clip:
  video:        60 frames × 1928×1208 (H.265 decoded → resized for Qwen image processor)
  annotations:  4 JSONL files (captions, states, front_car, traffic_lights) per scene
  frame_id:     0–599 (600 frames per 30s clip)

Model sees:
  visual tokens: Qwen2.5-VL-7B ViT processes video frames → token sequence
  task prompt:   "Describe the driving scene and predict trajectory"
  supervision:   plain_caption (Y), risk (W), trajectory[60×3] (trajectory head)
```

---

## Data Flow

```
/data/datasets/CoVLA/mini/<scene>/
  captions/<scene>.jsonl  ─────────────────────┐
  states/<scene>.jsonl    ─── ego state ────────┤
                          ─── trajectory ───────┤──► Supervision targets
  front_car/<scene>.jsonl ─── lead info ────────┤     Y: plain_caption
  traffic_lights/<scene>.jsonl ─ TL class/bbox ─┘     W: risk
  images/<scene>/NNNN.png                              traj: trajectory[60×3]
         │
         ▼
Qwen2.5-VL-7B video encoder (ViT + spatial merger)
  └─ Visual token sequence for 60-frame window
         │
         ▼
  ┌──────┴──────────────────┐
  ▼                         ▼
Caption LM head         Trajectory MLP head
(language model)        Linear(hidden → 60×3)
  │                         │
  ▼                         ▼
L_caption (cross-entropy)  L_traj (MSE or smooth-L1)
  └──────────┬──────────────┘
             ▼
         L_total (Stage 1)
```

---

## Examples in This Folder

| File | Contents |
|------|----------|
| `examples/caption-frame.jsonl` | Real caption entry (frame 0, scene 2022-07-14--14-32-55--10_first) |
| `examples/states-frame.jsonl` | Real states entry (ego speed, steering, trajectory — 5 of 60 points shown) |
| `examples/traffic-lights-frame.jsonl` | Real TL detection: green signal at pixel coords |
| `examples/front-car-frame.jsonl` | Real front-car entry: no lead vehicle at frame 44 |
| `examples/caption-pair.md` | Annotated plain_caption + risk with DSDAG mapping and ego state context |

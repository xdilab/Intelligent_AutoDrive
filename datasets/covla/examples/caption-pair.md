# CoVLA Caption Pair — Annotated

Source: `/data/datasets/CoVLA/mini/captions/2022-07-14--14-32-55--10_first.jsonl` (frame 0)

---

## Scene Context

- Scene: `2022-07-14--14-32-55--10_first`
- Frame: 0 (first frame of clip)
- Weather: **rainy** (confidence: 0.939)
- Road type: **wide road** (confidence: 0.939)
- Pedestrians: none detected

---

## plain_caption → Y (action node in DSDAG)

> "The ego vehicle is moving at a moderate speed and turning left. There is a traffic light near the ego vehicle displaying a green signal."

**What it captures:** Ego behavior (speed + direction) + observable scene objects (traffic light state).
**Supervision role:** Grounds the Y node — *what action is occurring right now*.

---

## risk → W (reason node in DSDAG)

> "to pay attention to the traffic light and other vehicles on the road. The driver should also be cautious of the wet road conditions due to the rain"

**What it captures:** Causal reasoning — WHY the driver needs to act with care.
**Supervision role:** Grounds the W node — *causal origin motivating the action*.

---

## rich_caption (full context)

> "The ego vehicle is moving at a moderate speed and turning left. There is a traffic light near the ego vehicle displaying a green signal. It is rainy. The car is driving on a wide road. No pedestrians appear to be present. What the driver of ego vehicle should be careful is to pay attention to the traffic light and other vehicles on the road. The driver should also be cautious of the wet road conditions due to the rain"

**Note:** `rich_caption = plain_caption + weather + road + risk` concatenated.

---

## Corresponding ego state (from states/ file, same frame)

| Field | Value | Meaning |
|-------|-------|---------|
| `vEgo` | 7.00 m/s (≈ 25 km/h) | Moderate speed in urban turn |
| `aEgo` | −0.37 m/s² | Slight deceleration |
| `steeringAngleDeg` | 1.35° | Turning left (small angle) |
| `brakePressed` | true | Brake engaged (slowing) |
| `leftBlinker` | true | Left turn signal active |
| `gearShifter` | drive | Forward gear |
| `trajectory[0]` | [0, 0, 0] | Current position |
| `trajectory[5]` | [1.81, −0.060, 0.004] | 0.25s ahead: 1.81m forward, curving left |
| `trajectory[59]` | [16.30, 2.82, −0.321] | 3.0s ahead: vehicle completes the left turn |

**Trajectory:** 60 points × 3D (x=forward, y=lateral, z=vertical) in vehicle frame.
Horizon: ~3 seconds at 20 FPS. Supervision signal for trajectory regression head.

---

## Generation pipeline (arXiv:2408.10845)

1. Rule-based layer: CAN data (speed, steering, blinker) + traffic light detector (OpenLenda-s) → factual constraints
2. VideoLLaMA2-7B: prompted with constraints over 60-frame window → natural language output
3. Result: `risk_correct=true`, `risk_yes_rate=0.64` (model confidence in risk description)

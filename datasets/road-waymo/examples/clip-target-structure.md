# ROAD-Waymo: Training Sample (Clip) Structure

Source: `experiments/exp1_road_r/dataset.py`, `experiments/exp1_road_r/config.py`

---

## One Training Sample = One Clip

A clip is `CLIP_LEN = 8` consecutive annotated frames from one video.
Config: `experiments/exp1_road_r/config.py` (CLIP_LEN=8, CLIP_STRIDE=16, MIN/MAX_PIXELS=448×448)

```
clip = (video_name, [fid_0, fid_1, ..., fid_7])   # 8 frame IDs
```

---

## Return Value of `__getitem__`

```python
pil_frames, frame_targets = dataset[idx]

pil_frames:    List[PIL.Image]           # 8 RGB images, 1920×1280 → resized to 448×448
frame_targets: List[Optional[dict]]      # 8 entries; None if frame has no annotations
```

---

## Per-Frame Target Dict

For a frame with `n` annotated boxes:

```python
{
    "boxes":   torch.tensor  shape=[n, 4]   dtype=float32
    #           normalized [x1, y1, x2, y2] ∈ [0, 1]

    "agent":   torch.tensor  shape=[n, 10]  dtype=float32   # multi-hot
    #           indices: Ped=0, Car=1, Cyc=2, Mobike=3, SmalVeh=4,
    #                    MedVeh=5, LarVeh=6, Bus=7, EmVeh=8, TL=9

    "action":  torch.tensor  shape=[n, 22]  dtype=float32   # multi-hot
    #           indices: Red=0, Amber=1, Green=2, MovAway=3, MovTow=4,
    #                    Mov=5, Rev=6, Brake=7, Stop=8, IncatLft=9, IncatRht=10,
    #                    HazLit=11, TurLft=12, TurRht=13, MovRht=14, MovLft=15,
    #                    Ovtak=16, Wait2X=17, XingFmLft=18, XingFmRht=19,
    #                    Xing=20, PushObj=21

    "loc":     torch.tensor  shape=[n, 16]  dtype=float32   # multi-hot
    #           indices: VehLane=0, OutgoLane=1, OutgoCycLane=2, OutgoBusLane=3,
    #                    IncomLane=4, IncomCycLane=5, IncomBusLane=6, Pav=7,
    #                    LftPav=8, RhtPav=9, Jun=10, xing=11, BusStop=12,
    #                    parking=13, LftParking=14, rightParking=15

    "duplex":  torch.tensor  shape=[n, 49]  dtype=float32   # multi-hot
    #           directly from anno["duplex_ids"] (0-based into duplex_labels)

    "triplet": torch.tensor  shape=[n, 86]  dtype=float32   # multi-hot
    #           reconstructed by cross-product: all valid (agent_name, action_name, loc_name)
    #           combos found in _triplet_lookup (dataset.py:197-208)
    #           NOTE: does NOT use anno["triplet_ids"] from JSON — those are raw values
}
```

---

## Concrete Example

Box b_33 from video train_00407, frame 1:
```
box      = [0.613, 0.477, 0.628, 0.526]   # Ped standing near right side mid-image
agent    = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # Ped=1 at index 0
action   = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # Stop=1 at index 8
loc      = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]   # RhtPav=1 at index 9
duplex   = [0, 0, 0, 1, 0, ...]   # Ped-Stop=1 at index 3
triplet  = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, ...]   # Ped-Stop-RhtPav=1 at index 12
```

---

## Flat Prediction Vector for T-Norm Loss

`losses.py:109-113` assembles this before calling `tnorm_loss.forward()`:

```
[agentness | agent×10 | action×22 | loc×16]   total = 49 dims
     ↑
  hardcoded 1.0 — all GT boxes are positives, agentness is always active
```

Only these 49 dims are used by the t-norm loss.
Duplex [49] and triplet [86] heads exist only in the model output and BCE loss.

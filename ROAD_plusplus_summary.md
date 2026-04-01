# ROAD++ (Road-Waymo) Dataset Exploration Summary

## Dataset Overview

The **ROAD++ dataset** (Road-Waymo) is a large-scale, richly annotated autonomous driving benchmark for multi-label spatiotemporal detection of road agents, their actions, and their scene locations. It is released through the **ROAD++ challenge** ("The Second Workshop & Challenge on Event Detection for Situation Awareness in Autonomous Driving") and extends the annotation scheme of the original [ROAD dataset](https://github.com/gurkirt/road-dataset) (Singh et al., IEEE TPAMI 2022) to Waymo Open Dataset video footage.

- **Full name:** ROAD++ (Road-Waymo Dataset)
- **Repository:** https://github.com/salmank255/Road-waymo-dataset
- **Challenge page:** https://sites.google.com/view/road-plus-plus/home
- **Baseline code:** https://github.com/salmank255/ROAD_plus_plus_Baseline
- **Video source:** [Waymo Open Dataset](https://waymo.com/open/) — front-camera clips extracted from Waymo TFRecord sequences
- **Annotation authors:** [Visual Artificial Intelligence Laboratory](https://cms.brookes.ac.uk/staff/FabioCuzzolin/), Oxford Brookes University
- **License:** CC BY-NC-SA 4.0 (non-commercial research use only); video data additionally governed by the [Waymo Open Dataset License Agreement for Non-Commercial Use](https://waymo.com/open/terms/)
- **Sensor modality:** Monocular RGB video, front camera only (1920 × 1280), 10 FPS clips

### Purpose and Scope

ROAD++ is designed to benchmark systems that simultaneously detect:

1. **Agent type** — what category of road agent is present (Pedestrian, Car, Cyclist, Traffic Light, etc.)
2. **Agent action** — what the agent is doing (Moving-Away, Turning-Right, Braking, Crossing-from-Left, etc.)
3. **Agent location** — where in the scene the agent is situated (Outgoing Lane, Junction, Right Pavement, etc.)
4. **AV action** — what the ego autonomous vehicle itself is doing (AV-Moving, AV-TurningRight, AV-Stopped, etc.)
5. **Duplex events** — composite agent+action situation labels (e.g., `Ped-Xing`, `Car-TurRht`)
6. **Triplet events** — composite agent+action+location situation labels (e.g., `Ped-Xing-Jun`, `Car-MovAway-OutgoLane`)

The annotation structure is identical to the original ROAD dataset, making ROAD++ a large-scale, geographically diverse complement to ROAD's UK dashcam footage.

---

## Dataset Statistics

All statistics in this section are computed directly from `road_waymo_trainval_v1.0.json`. No values are taken from the README or paper.

### High-Level Counts

| Metric | Value |
|--------|-------|
| Total videos in annotation file | 798 |
| Train split videos | 600 |
| Val split videos | 198 |
| Test videos (separate directory, no annotations) | 202 |
| Total frames (summed `numf`) | 158,081 |
| Train split frames | 118,877 |
| Val split frames | 39,204 |
| Video resolution | 1920 × 1280 |
| Frame rate | 10 FPS |
| Total annotated frames (`annotated == 1`) | 153,534 |
| Train annotated frames | 115,602 |
| Val annotated frames | 37,932 |
| Total bounding box annotations | 3,304,353 |
| Train bounding box annotations | 2,515,836 |
| Val bounding box annotations | 788,517 |

**Note on README claims:** The README states "198K frames from 1000 videos." The actual annotation JSON covers 798 videos and 158,081 total frames. An additional 202 test-split videos exist in a separate directory (`test_videos/`) but have no associated annotation entries in the released JSON — they are unannotated hold-out clips for challenge submission. The split definitions (train=600, val=198) are encoded in each video's `split_ids` field.

**Note on naming convention:** All 798 annotated videos are named `train_00000` through `train_00797` in the JSON database keys and in the video files on disk. The split assignment (train vs. val) is determined by the `split_ids` field, not by the filename prefix. Videos `train_00000`–`train_00599` are in the train split; `train_00600`–`train_00797` are in the val split. The test videos in `test_videos/` are named `val_00000.mp4`–`val_00201.mp4` but have no annotations in the trainval JSON.

### Video Length Distribution

| Metric | Value |
|--------|-------|
| Min frames per video | 171 |
| Max frames per video | 200 |
| Mean frames per video | 198.1 |
| Median frames per video | 198.0 |
| Approximate video duration | ~19.8 seconds (at 10 FPS) |

The tight range (171–200 frames) reflects that all clips are extracted as fixed-length segments from Waymo TFRecords at 10 FPS.

### Annotation Volume Summary

| Label Type | Total Label Instances | Avg per Box |
|------------|----------------------|-------------|
| Agent labels (bounding box agent class) | 3,304,516 | 1.00 |
| Action labels (multi-label per box) | 3,576,722 | 1.08 |
| Location labels (multi-label per box) | 3,478,578 | 1.05 |
| Duplex labels (derived agent+action) | 3,562,231 | 1.08 |
| Triplet labels (derived agent+action+loc) | 2,614,911 | 0.79 |

Agent labels are essentially 1:1 with boxes (one agent class per box), consistent with the single-assignment design. Action and location labels are slightly above 1.0 per box, confirming they are multi-label. Triplet labels average below 1.0 per box because not every agent+action combination has a matching location class in the task label set.

### Train / Val / Test Splits

| Split | Videos | Frames | Annotated Frames | Bounding Boxes |
|-------|--------|--------|-----------------|----------------|
| Train | 600 | 118,877 | 115,602 | 2,515,836 |
| Val | 198 | 39,204 | 37,932 | 788,517 |
| Test | 202 | — | — (no labels released) | — |
| **Total (train+val)** | **798** | **158,081** | **153,534** | **3,304,353** |

### Tube Counts

| Tube Type | Count | Description |
|-----------|-------|-------------|
| Agent tubes | 41,935 | Spatial tracks of individual agents across frames |
| Action tubes | 45,050 | Per-action-class tracking of agent behavior over time |
| Location tubes | 44,043 | Per-location-class tracking of where agents are |
| Duplex tubes | 44,879 | Compositional agent+action event tracks |
| Triplet tubes | 33,457 | Compositional agent+action+location event tracks |
| AV action tubes | 813 | Ego-vehicle action tracks (one per AV action segment per video) |

### Agent Tube Statistics

| Metric | Value |
|--------|-------|
| Total agent tubes | 41,935 |
| Min tube length (frames) | 1 |
| Max tube length (frames) | 199 |
| Mean tube length (frames) | 78.8 |
| Median tube length (frames) | 58.0 |

**Per-class agent tube breakdown:**

| Class | Full Name | Tube Count | Mean Length (frames) | Median Length (frames) |
|-------|-----------|-----------|---------------------|----------------------|
| Ped | Pedestrian | 9,573 | 74.4 | 58.0 |
| Car | Car | 28,231 | 77.9 | 55.0 |
| Cyc | Cyclist | 161 | 73.9 | 64.0 |
| Mobike | Motorbike | 108 | 74.1 | 66.0 |
| SmalVeh | Small Vehicle | 31 | 91.1 | 82.0 |
| MedVeh | Medium Vehicle | 2,547 | 92.3 | 75.0 |
| LarVeh | Large Vehicle | 379 | 105.9 | 97.0 |
| Bus | Bus | 290 | 115.4 | 101.5 |
| EmVeh | Emergency Vehicle | 26 | 81.4 | 53.0 |
| TL | Traffic Light | 589 | 98.0 | 79.0 |

Cars dominate the tube population (67.3% of all agent tubes), followed by Pedestrians (22.8%). Larger, slower-moving agents (Bus, Large Vehicle, Traffic Light) have longer tube lengths on average, reflecting their extended presence in the scene.

---

## Visualizations

**Real annotated frame samples:** `/data/repos/PedestrianIntent++/ROAD_plusplus/viz/ROAD_real_annotated_frames.png`

A 5-row × 4-column grid showing real Waymo front-camera frames extracted from five high-density train videos (`train_00331`, `train_00117`, `train_00240`, `train_00355`, `train_00064`). Four temporally-spaced frames are shown per video. Each frame has bounding boxes drawn in a distinct color per agent class (see legend), with a short label text `agent_class|first_action`. The AV ego-vehicle action (e.g., `AV-Mov`, `AV-Stop`) appears in the top-left corner of each frame. Frame captions show the frame number within the video, the number of bounding boxes in that frame, and the AV action.

The frames reveal the visual character of the Waymo front-camera footage: wide-angle 1920×1280 images of US city streets, capturing intersections with multiple simultaneous cars, pedestrians, traffic lights, and occasional cyclists. High-density frames with 50–80+ boxes per frame illustrate why ROAD++ is challenging for small-object detection — many agents appear as very small bounding boxes far from the camera.

**Agent tube timeline:** `/data/repos/PedestrianIntent++/ROAD_plusplus/viz/ROAD_tube_timeline.png`

A swimlane diagram for two train videos (`train_00240` and `train_00355`), each with 177 and 156 agent tubes respectively. The top 60 tubes (sorted by agent class then start frame) are displayed as horizontal bars. Bar color encodes agent class using the same legend as the frame grid. Bar horizontal span shows the first and last frame the tube is active. The diagrams illustrate that Car tubes (green) are most numerous and often span nearly the full 199-frame clip, while Pedestrian tubes (sky blue) tend to be shorter (consistent with the 58-frame median) and Traffic Light tubes (gray) persist for long stretches.

---

## Annotation Format

### File Format: JSON (single combined file)

All train and validation annotations are stored in a single JSON file:

```
road_waymo_trainval_v1.0.json   (~1 GB)
```

This file holds the complete dataset-level label taxonomy, per-video frame-level annotations, and per-video tube-level structures for all 798 annotated videos.

### Top-Level JSON Keys

```python
dict_keys([
    'all_input_labels',     # all classes across all label types
    'all_av_action_labels', # superset of AV action classes
    'av_action_labels',     # AV action classes used in final task (9 classes)
    'agent_labels',         # agent classes used in final task (10 classes)
    'action_labels',        # action classes used in final task (22 classes)
    'loc_labels',           # location classes used in final task (16 classes)
    'duplex_labels',        # composite agent+action labels (49 classes)
    'triplet_labels',       # composite agent+action+location labels (86 classes)
    'old_loc_labels',       # earlier location label list (16 classes, identical to loc_labels)
    'label_types',          # ['agent', 'action', 'loc', 'duplex', 'triplet']
    'all_duplex_labels',    # superset of all possible duplex combinations (152 classes)
    'all_triplet_labels',   # superset of all possible triplet combinations (1620 classes)
    'all_loc_labels',       # superset of location classes (16 classes)
    'all_agent_labels',     # superset of agent classes (11 classes, adds 'OthTL')
    'all_action_labels',    # superset of action classes (22 classes)
    'duplex_childs',        # list of [agent_id, action_id] pairs indexed by duplex label id
    'triplet_childs',       # list of [agent_id, action_id, loc_id] triples indexed by triplet label id
    'db'                    # per-video annotation database
])
```

| Top-level key | Type | Description |
|---------------|------|-------------|
| `label_types` | list[str] | `['agent', 'action', 'loc', 'duplex', 'triplet']` |
| `agent_labels` | list[str] | 10 agent classes used in the final task |
| `action_labels` | list[str] | 22 action classes used in the final task |
| `loc_labels` | list[str] | 16 location classes used in the final task |
| `duplex_labels` | list[str] | 49 composite agent+action class names |
| `triplet_labels` | list[str] | 86 composite agent+action+location class names |
| `av_action_labels` | list[str] | 9 AV ego-vehicle action classes |
| `all_*_labels` | list[str] | Superset of annotated classes per type (before filtering) |
| `duplex_childs` | list[list[int]] | `duplex_childs[i] = [agent_id, action_id]` for duplex class `i` (indexed into `all_agent_labels` / `all_action_labels`) |
| `triplet_childs` | list[list[int]] | `triplet_childs[i] = [agent_id, action_id, loc_id]` for triplet class `i` |
| `db` | dict | Per-video annotations keyed by video name (e.g. `'train_00000'`) |

**Important clarification on duplex/triplet indexing:** Annotation `duplex_ids` and `triplet_ids` fields in frame-level `annos` dicts index into `all_duplex_labels` and `all_triplet_labels` respectively (not into the shorter `duplex_labels` / `triplet_labels` lists). The maximum observed `duplex_id` in annotations is 137, within the 152-class `all_duplex_labels` space. `duplex_childs` and `triplet_childs` are also indexed relative to `all_*_labels`.

### Per-Video Annotation Structure

Access a video with `db['train_00000']`. Each video entry contains:

```python
dict_keys([
    'numf',            # total frame count
    'split_ids',       # list of split assignments: ['all', 'train'] or ['all', 'val']
    'frames',          # per-frame annotation dictionary
    'agent_tubes',     # agent-level spatiotemporal tracks
    'action_tubes',    # action-level tracks
    'loc_tubes',       # location-level tracks
    'duplex_tubes',    # composite agent+action tracks
    'triplet_tubes',   # composite agent+action+location tracks
    'av_action_tubes'  # ego-vehicle action segments
])
```

| Video-level key | Type | Description |
|-----------------|------|-------------|
| `split_ids` | list[str] | Always includes `'all'` plus either `'train'` or `'val'` |
| `numf` | int | Total frames in the video (171–200, typically 198) |
| `frames` | dict | Per-frame annotation dictionary keyed by frame number string |
| `agent_tubes` | dict | All agent tubes, keyed by UUID string tube ID |
| `action_tubes` | dict | All action tubes, keyed by tube ID |
| `loc_tubes` | dict | All location tubes, keyed by tube ID |
| `duplex_tubes` | dict | All duplex (composite) tubes, keyed by tube ID |
| `triplet_tubes` | dict | All triplet (composite) tubes, keyed by tube ID |
| `av_action_tubes` | dict | AV action tubes for the ego-vehicle |

**Note:** The README describes a `frame_labels` field for per-frame AV actions stored as an array, but the actual released JSON does not include this field. AV action labels are instead found per-frame in `frames[f]['av_action_ids']`.

### Per-Frame Annotation Structure

Access a frame with `db['train_00000']['frames']['1']`:

```python
dict_keys([
    'annotated',       # 0 or 1 — whether this frame has annotations
    'rgb_image_id',    # physical frame index for JPEG extraction
    'width',           # 1920
    'height',          # 1280
    'annos',           # dict of bounding box annotations for this frame
    'av_action_ids'    # list of AV action class IDs for this frame
])
```

| Frame-level key | Type | Description |
|-----------------|------|-------------|
| `annotated` | int (0/1) | 1 if this frame has ground-truth annotations |
| `rgb_image_id` | str | ID of the extracted JPEG frame file (e.g. `'00001'`) |
| `width` | int | Frame width in pixels (always 1920) |
| `height` | int | Frame height in pixels (always 1280) |
| `av_action_ids` | list[int] | AV action label indices for this frame (index into `av_action_labels`) |
| `annos` | dict | Per-box annotations keyed by a string key (e.g. `'b_1'`, `'b_12019'`) |

### Per-Annotation (Bounding Box) Structure

Each key in `annos` maps to a dictionary with complete box data stored inline (not as an external reference):

```python
annos['b_1'] = {
    'box':        [0.4119, 0.4524, 0.4238, 0.4672],  # [xmin, ymin, xmax, ymax] normalized
    'agent_ids':  [1],                                 # Car
    'action_ids': [3],                                 # MovAway
    'loc_ids':    [1],                                 # OutgoLane
    'duplex_ids': [9],                                 # Car-MovAway
    'triplet_ids':[109],                               # Car-MovAway-OutgoLane
    'tube_uid':   '283ae350-0e37-460c-9d1a-d5d62a277a61'
}
```

| Key | Type | Description |
|-----|------|-------------|
| `box` | list[float] | Normalized bounding box `[xmin, ymin, xmax, ymax]`, values in `[0.0, 1.0]` |
| `agent_ids` | list[int] | Agent class index (into `agent_labels`); typically a single element |
| `action_ids` | list[int] | Action class indices (into `action_labels`), multi-label |
| `loc_ids` | list[int] | Location class indices (into `loc_labels`), multi-label |
| `duplex_ids` | list[int] | Duplex event class indices (into `all_duplex_labels`) |
| `triplet_ids` | list[int] | Triplet event class indices (into `all_triplet_labels`); may be empty |
| `tube_uid` | str | UUID string linking this box to a tube in `agent_tubes` |

**Coordinate convention:**
```
box = [xmin, ymin, xmax, ymax]   # values in [0.0, 1.0]
# Pixel coordinates:
x1_px = xmin * width   (1920)
y1_px = ymin * height  (1280)
```

### Tube-Level Annotation Structure

Each tube entry (e.g. `db['train_00000']['agent_tubes']['283ae350-...']`) contains:

```python
{
    'label_id': 1,          # Car (index into agent_labels)
    'annos': {
        '1':   'b_1',       # frame 1 → annotation key 'b_1'
        '2':   'b_2',       # frame 2 → annotation key 'b_2'
        ...
    }
}
```

| Key | Type | Description |
|-----|------|-------------|
| `label_id` | int | Primary class index for this tube (into respective label list) |
| `annos` | dict | Frame-keyed dict mapping frame ID string → annotation key string |

Tube annotation keys refer back to the same annotation dict in `frames[frame_id]['annos'][anno_key]`, providing a two-way link between tube-level and frame-level views.

### Concrete Annotation Example

```json
{
  "agent_labels": ["Ped", "Car", "Cyc", "Mobike", "SmalVeh", "MedVeh", "LarVeh", "Bus", "EmVeh", "TL"],
  "action_labels": ["Red", "Amber", "Green", "MovAway", "MovTow", "Mov", "Rev", "Brake", "Stop",
                    "IncatLft", "IncatRht", "HazLit", "TurLft", "TurRht", "MovRht", "MovLft",
                    "Ovtak", "Wait2X", "XingFmLft", "XingFmRht", "Xing", "PushObj"],
  "av_action_labels": ["AV-Stop", "AV-Mov", "AV-TurRht", "AV-TurLft", "AV-MovRht",
                       "AV-MovLft", "AV-Ovtak", "AV-Rev", "AV-Brake"],
  "db": {
    "train_00407": {
      "numf": 199,
      "split_ids": ["all", "train"],
      "frames": {
        "1": {
          "annotated": 1,
          "rgb_image_id": "00001",
          "width": 1920,
          "height": 1280,
          "av_action_ids": [1],
          "annos": {
            "b_1": {
              "box": [0.4119, 0.4524, 0.4238, 0.4672],
              "agent_ids": [1],
              "action_ids": [3],
              "loc_ids": [1],
              "duplex_ids": [9],
              "triplet_ids": [109],
              "tube_uid": "283ae350-0e37-460c-9d1a-d5d62a277a61"
            }
          }
        }
      },
      "agent_tubes": {
        "283ae350-0e37-460c-9d1a-d5d62a277a61": {
          "label_id": 1,
          "annos": {
            "1":  "b_1",
            "2":  "b_2",
            "3":  "b_3"
          }
        }
      }
    }
  }
}
```

In the above: frame `"1"` of `train_00407` has one annotation `b_1`. It is a Car (`agent_ids=[1]`) moving away (`action_ids=[3]` = MovAway) in the outgoing lane (`loc_ids=[1]` = OutgoLane). The AV is moving (`av_action_ids=[1]` = AV-Mov). The agent tube `283ae350-...` is a Car tube spanning at least frames 1–3.

---

## Label Taxonomy

### Agent Labels (10 final task classes)

| ID | Short Name | Full Meaning | Tubes | Box Instances | % of Boxes |
|----|------------|--------------|-------|---------------|------------|
| 0 | Ped | Pedestrian | 9,573 | 712,640 | 21.6% |
| 1 | Car | Car | 28,231 | 2,197,049 | 66.5% |
| 2 | Cyc | Cyclist | 161 | 11,894 | 0.4% |
| 3 | Mobike | Motorbike | 108 | 8,002 | 0.2% |
| 4 | SmalVeh | Small Vehicle | 31 | 2,755 | 0.1% |
| 5 | MedVeh | Medium Vehicle | 2,547 | 238,522 | 7.2% |
| 6 | LarVeh | Large Vehicle | 379 | 39,889 | 1.2% |
| 7 | Bus | Bus | 290 | 33,904 | 1.0% |
| 8 | EmVeh | Emergency Vehicle | 26 | 2,139 | 0.1% |
| 9 | TL | Traffic Light | 589 | 57,722 | 1.7% |
| | **Total** | | **41,935** | **3,304,516** | |

The superset (`all_agent_labels`, 11 classes) adds `OthTL` (Other Traffic Light), which is excluded from the final task label set.

Cars dominate the dataset (66.5% of boxes), reflecting Waymo urban driving scenes. Pedestrians are the second most frequent class (21.6%). Rare classes include Small Vehicle (0.1%), Emergency Vehicle (0.1%), and Motorbike (0.2%) — these are heavily long-tail and present class imbalance challenges.

### Action Labels (22 final task classes)

Note: Action labels 0–2 (Red, Amber, Green) are traffic-light signal states, not behavioral actions of other agents.

| ID | Short Name | Full Meaning | Box Instances | % |
|----|------------|--------------|---------------|---|
| 0 | Red | Traffic light: Red | 36,723 | 1.0% |
| 1 | Amber | Traffic light: Amber | 938 | 0.0% |
| 2 | Green | Traffic light: Green | 19,258 | 0.5% |
| 3 | MovAway | Moving Away (from ego AV) | 613,256 | 17.1% |
| 4 | MovTow | Moving Toward (ego AV) | 494,052 | 13.8% |
| 5 | Mov | Moving (lateral / general) | 81,195 | 2.3% |
| 6 | Rev | Reversing | 1,906 | 0.1% |
| 7 | Brake | Braking / Decelerating | 188,451 | 5.3% |
| 8 | Stop | Stopped / Stationary | 1,767,110 | 49.4% |
| 9 | IncatLft | Indicating Left | 28,169 | 0.8% |
| 10 | IncatRht | Indicating Right | 20,947 | 0.6% |
| 11 | HazLit | Hazard Lights On | 18,821 | 0.5% |
| 12 | TurLft | Turning Left | 19,566 | 0.5% |
| 13 | TurRht | Turning Right | 21,979 | 0.6% |
| 14 | MovRht | Moving Right (lane change) | 6,871 | 0.2% |
| 15 | MovLft | Moving Left (lane change) | 7,256 | 0.2% |
| 16 | Ovtak | Overtaking | 329 | 0.0% |
| 17 | Wait2X | Waiting to Cross | 54,928 | 1.5% |
| 18 | XingFmLft | Crossing from Left | 72,395 | 2.0% |
| 19 | XingFmRht | Crossing from Right | 75,591 | 2.1% |
| 20 | Xing | Crossing (general) | 41,659 | 1.2% |
| 21 | PushObj | Pushing Object | 5,322 | 0.1% |
| | **Total** | | **3,576,722** | |

`Stop` (Stopped/Stationary) is by far the most common action class (49.4%), reflecting that a large proportion of scene agents are stationary vehicles in traffic. `MovAway` and `MovTow` together account for 30.9% of action labels. Rare classes with fewer than 10,000 instances include `Rev`, `Ovtak`, `PushObj`, `MovRht`, `MovLft`, `HazLit`, and all traffic-light signal labels.

### Location Labels (16 final task classes)

| ID | Short Name | Full Meaning | Box Instances | % |
|----|------------|--------------|---------------|---|
| 0 | VehLane | Vehicle Lane (ego lane) | 199,141 | 5.7% |
| 1 | OutgoLane | Outgoing (ego-direction) Lane | 472,362 | 13.6% |
| 2 | OutgoCycLane | Outgoing Cycle Lane | 1,674 | 0.0% |
| 3 | OutgoBusLane | Outgoing Bus Lane | 3,916 | 0.1% |
| 4 | IncomLane | Incoming (oncoming) Lane | 443,049 | 12.7% |
| 5 | IncomCycLane | Incoming Cycle Lane | 1,910 | 0.1% |
| 6 | IncomBusLane | Incoming Bus Lane | 3,304 | 0.1% |
| 7 | Pav | Pavement (general) | 70,385 | 2.0% |
| 8 | LftPav | Left Pavement | 208,315 | 6.0% |
| 9 | RhtPav | Right Pavement | 248,812 | 7.2% |
| 10 | Jun | Junction / Intersection | 553,439 | 15.9% |
| 11 | xing | Pedestrian Crossing | 16,209 | 0.5% |
| 12 | BusStop | Bus Stop | 10,071 | 0.3% |
| 13 | parking | Parking Area (general) | 60,597 | 1.7% |
| 14 | LftParking | Left Parking | 574,962 | 16.5% |
| 15 | rightParking | Right Parking | 610,432 | 17.5% |
| | **Total** | | **3,478,578** | |

The most frequent location classes are RightParking (17.5%), LeftParking (16.5%), and Junction (15.9%). Parking areas are extremely common in the Waymo street footage. Junction is the third most common context, reflecting that Waymo clips heavily feature intersection scenarios. Cycle lanes, bus lanes, and pedestrian crossings are rare.

### AV Action Labels (9 classes)

| ID | Short Name | Full Meaning | Frame Instances | % |
|----|------------|--------------|-----------------|---|
| 0 | AV-Stop | AV Stopped | 29,162 | 19.0% |
| 1 | AV-Mov | AV Moving (forward) | 117,340 | 76.5% |
| 2 | AV-TurRht | AV Turning Right | 2,723 | 1.8% |
| 3 | AV-TurLft | AV Turning Left | 2,693 | 1.8% |
| 4 | AV-MovRht | AV Moving Right (lane change) | 1,164 | 0.8% |
| 5 | AV-MovLft | AV Moving Left (lane change) | 731 | 0.5% |
| 6 | AV-Ovtak | AV Overtaking | — | — |
| 7 | AV-Rev | AV Reversing | 41 | 0.0% |
| 8 | AV-Brake | AV Braking | 26 | 0.0% |
| | **Total** | | **153,880** | |

The AV is moving forward (AV-Mov) in 76.5% of annotated frames, stopped (AV-Stop) in 19.0%, and performing turns in 3.6%. Reversing and hard braking are extremely rare events. Note that `AV-Ovtak` (overtaking) has no observed instances in the released annotation file, though it is defined in the label set.

### Duplex Labels (49 final task classes)

Duplex labels combine one agent class with one action class into a composite situation descriptor. They are derived labels, not independently annotated: each duplex label at index `i` has its components defined by `duplex_childs[i] = [agent_id, action_id]` (indexing into `all_agent_labels` and `all_action_labels`).

The 49 final-task duplex labels include:

| Agent | Associated Actions |
|-------|--------------------|
| Ped | MovAway, MovTow, Mov, Stop, Wait2X, XingFmLft, XingFmRht, Xing, PushObj |
| Car | MovAway, MovTow, Brake, Stop, IncatLft, IncatRht, HazLit, TurLft, TurRht, MovRht, MovLft, XingFmLft, XingFmRht |
| Cyc | MovAway, MovTow, Stop |
| Mobike | Stop |
| MedVeh | MovAway, MovTow, Brake, Stop, IncatLft, IncatRht, HazLit, TurRht, XingFmLft, XingFmRht |
| LarVeh | MovAway, MovTow, Stop, HazLit |
| Bus | MovAway, MovTow, Brake, Stop, HazLit |
| EmVeh | Stop |
| TL | Red, Amber, Green |

The superset (`all_duplex_labels`) contains 152 possible combinations; the final 49 task labels cover only the agent+action pairs that actually appear with sufficient frequency.

### Triplet Labels (86 final task classes)

Triplet labels combine agent+action+location into the most specific situation descriptor. The 86 final-task triplets are drawn from 1,620 possible combinations in `all_triplet_labels`. Each is defined by `triplet_childs[i] = [agent_id, action_id, loc_id]`. Not every annotation box has triplet labels (some agent+action+location combinations do not match any of the 86 final-task triplets).

Example triplet labels include:
- `Ped-Xing-Jun` — pedestrian crossing at a junction
- `Car-MovAway-OutgoLane` — car moving away in the outgoing lane
- `Car-Stop-VehLane` — car stopped in the vehicle lane
- `Bus-Stop-Jun` — bus stopped at a junction
- `Ped-Wait2X-RhtPav` — pedestrian waiting to cross from the right pavement

---

## Data Organization

### Local Directory Structure (as downloaded)

```
/data/datasets/ROAD_plusplus/
    road_waymo_trainval_v1.0.json     # Combined train+val annotations (~1 GB)
    videos/                            # 798 annotated train+val MP4 clips
        train_00000.mp4
        train_00001.mp4
        ...
        train_00797.mp4
    test_videos/                       # 202 unannotated test MP4 clips
        val_00000.mp4
        val_00001.mp4
        ...
        val_00201.mp4
```

### Recommended Baseline Directory Structure

Per the ROAD++ baseline and README:

```
road-waymo/
    road_waymo_trainval_v1.0.json
    videos/
        train_00000.mp4
        ...
        train_00797.mp4
    rgb-images/                        # Created by extract_videos2jpgs.py
        train_00000/
            00001.jpg
            00002.jpg
            ...
        ...
```

### Video File Naming

```
{lowercase_split}_{NNNNN}.mp4
```

- All 798 annotated videos are named `train_00000.mp4` through `train_00797.mp4` regardless of their JSON split assignment
- The test videos (no annotations) are named `val_00000.mp4` through `val_00201.mp4` and reside in a separate `test_videos/` directory
- The JSON database key for each video omits the `.mp4` extension: `'train_00000'`

### Frame Naming After Extraction

```
rgb-images/train_00000/00001.jpg   # frame 1 (1-indexed)
rgb-images/train_00000/00002.jpg   # frame 2
...
```

Frame numbering is 1-indexed. The JSON `frames` dictionary also uses 1-indexed string keys (`"1"`, `"2"`, ...), matching the extracted JPEG filenames.

### JSON Annotation Access Pattern

```python
import json

with open('road_waymo_trainval_v1.0.json') as f:
    data = json.load(f)

# Dataset-level label lists
agent_labels     = data['agent_labels']     # 10 classes
action_labels    = data['action_labels']    # 22 classes
loc_labels       = data['loc_labels']       # 16 classes
av_action_labels = data['av_action_labels'] # 9 classes

# Per-video iteration
for video_name, video_data in data['db'].items():
    split = video_data['split_ids']       # ['all', 'train'] or ['all', 'val']
    numf  = video_data['numf']            # int
    agent_tubes = video_data['agent_tubes']

    for frame_id, frame_data in video_data['frames'].items():
        if frame_data['annotated'] != 1:
            continue
        av_action = frame_data['av_action_ids']
        for anno_key, anno in frame_data['annos'].items():
            box    = anno['box']         # [xmin, ymin, xmax, ymax] in [0,1]
            agents = anno['agent_ids']
            acts   = anno['action_ids']
            locs   = anno['loc_ids']
```

---

## API and Utility Scripts

The repository (`salmank255/Road-waymo-dataset`) provides five utility scripts:

| File | Purpose |
|------|---------|
| `README.md` | Download instructions, data structure documentation, and annotation schema reference |
| `extract_videos2jpgs.py` | Extracts JPEG frames from all MP4 videos using ffmpeg |
| `frames_clips.py` | Assembles per-video MP4 clips from pre-extracted Waymo JPEG frames (for users starting from raw TFRecords) |
| `convert_waymo_to_coco.py` | Converts raw Waymo TFRecord sequences to COCO-format JSON |
| `plot_annotations.py` | Draws bounding boxes onto frames (uses an intermediate per-video JSON format, not the released combined JSON) |
| `plot.gif` | Example animation showing annotated bounding boxes |
| `LICENSE` | CC BY-NC-SA 4.0 full license text |

### `extract_videos2jpgs.py` — Frame Extraction

Extracts all frames from every `.mp4` file in the `videos/` subdirectory and writes them as high-quality JPEGs under `rgb-images/`.

```bash
python extract_videos2jpgs.py <path-to-road-waymo-folder>
```

Calls `ffmpeg -i <video> -q:v 1 <out_dir>/%05d.jpg` for each video. Skips a video if its output directory already contains 10 or more frames (enables resumable extraction). Requires `ffmpeg` on `PATH`.

### `frames_clips.py` — Clip Assembly

Assembles MP4 clips from per-frame JPEGs extracted from raw Waymo TFRecords. Supports `--MODE train`, `val`, `test`, or `all`. Only needed if building the video files from scratch rather than downloading the pre-built MP4s from Waymo's Community Contributions.

### `convert_waymo_to_coco.py` — TFRecord Conversion

Converts raw Waymo TFRecord sequences to COCO-format JSON. Filters to the front camera (`camera_image.name == 1`). Requires `tensorflow` and the `waymo-open-dataset` Python package.

### `plot_annotations.py` — Visualization

Reads per-video JSON annotation files from a `combined_jsons/{split}/` directory (an intermediate annotation pipeline format, one JSON per video) and renders annotated videos. This script is a development tool used during the annotation creation process; it is not a loader for the final combined `road_waymo_trainval_v1.0.json` format.

---

## Key Characteristics

### Strengths

1. **Large scale with temporal depth:** 798 annotated videos (158,081 frames, 153,534 annotated), 41,935 agent tubes, and 3.3M bounding boxes provide a substantially larger training corpus than the original ROAD dataset (22 videos).

2. **Five-dimensional label system:** Every bounding box simultaneously carries agent type, action (multi-label), location (multi-label), duplex (compositional), and triplet (compositional) labels. This five-dimensional annotation directly enables situation-awareness benchmarking — a task distinct from simple object detection or single-label classification.

3. **Compositional event taxonomy:** Duplex and triplet labels are derived from finer primitives and fully documented through `duplex_childs` / `triplet_childs`, making the label space interpretable and extensible. The 49 duplex and 86 triplet final-task labels represent a carefully curated subset of all possible combinations.

4. **Temporal tube structure:** Agent UUID tracks (`tube_uid`) link annotations across frames, supporting spatiotemporal action tube detection tasks (3D detection, temporal action localization).

5. **AV action labels:** Per-frame ego-vehicle action labels (`av_action_ids`) are rare in driving datasets and support tasks such as predicting AV maneuvers conditioned on scene context.

6. **Traffic light as a first-class agent:** Traffic lights are annotated as trackable objects (`TL` agent class) with their own signal-state actions (`Red`, `Amber`, `Green`) and spatial tubes, enabling signal-state detection alongside road agent detection.

7. **Consistent schema with original ROAD:** The annotation format is deliberately identical to the original ROAD dataset, enabling joint training and direct transfer learning between Oxford (UK) and Waymo (US) footage.

8. **US geographic diversity:** Waymo data covers multiple US cities with varied weather, lighting, and traffic conditions, contrasting with the UK-only original ROAD dataset.

9. **Single-file distribution:** All train and validation annotations fit in one 1 GB JSON file, simplifying ingestion compared to per-video formats.

### Limitations

1. **Annotation JSON covers only 798 videos, not 1,000:** The README claims 1,000 videos; the actual released JSON contains 798 annotated videos (train+val). The 202 test-split videos in `test_videos/` have no annotation entries in the released file.

2. **Front camera only:** Only the Waymo front camera is used. The four other Waymo camera views (front-left, front-right, side-left, side-right) are discarded, sacrificing 360-degree scene coverage.

3. **No 3D annotations:** Despite Waymo's LiDAR and multi-camera rig, ROAD++ uses only 2D bounding boxes from the front RGB camera.

4. **No ego-vehicle telemetry:** Vehicle speed, steering, GPS, and IMU data from the Waymo platform are not included in the annotation JSON, though available in raw TFRecords.

5. **Access requires registration:** Videos must be downloaded through the Waymo portal (Community Contributions section) after accepting the non-commercial license.

6. **Low frame rate:** Clips are assembled at 10 FPS, lower than dashcam datasets recorded at 25–30 FPS.

7. **No pedestrian intent labels:** Unlike PIE or JAAD, there are no per-pedestrian crossing-intent probability scores, crossing-point markers, or demographic attributes.

8. **Severe class imbalance:** Car dominates at 66.5% of boxes; Emergency Vehicle (0.1%), Motorbike (0.2%), and Small Vehicle (0.1%) are extremely rare. Action class `Stop` represents 49.4% of all action labels, with `Overtaking` appearing only 329 times.

9. **Confusing naming conventions:** The JSON database uses `train_NNNNN` as the key for all 798 videos; the test videos in `test_videos/` are named `val_NNNNN.mp4`. This is inconsistent with the README's `Train_NNNNN.mp4` / `Val_NNNNN.mp4` / `Test_NNNNN.mp4` description.

10. **4 videos missing agent tubes:** Four train/val videos (`train_00453`, `train_00163`, `train_00443`, `train_00636`) have no `agent_tubes` entries, suggesting annotation gaps.

### Intended Use Cases

- **Spatiotemporal action tube detection:** Detecting what road agents are doing across time using 3D convolutional or transformer-based models
- **Multi-label scene understanding:** Jointly predicting agent type, action, and location from video clips
- **Traffic situation recognition:** Predicting high-level duplex/triplet event labels for situation-aware driving
- **AV maneuver prediction:** Using per-frame `av_action_ids` labels to classify or predict the ego vehicle's actions
- **Large-scale pretraining:** The 798-video scale makes ROAD++ suitable for pretraining models before fine-tuning on smaller ROAD-domain tasks
- **Domain generalization:** Cross-domain training (Waymo US footage → ROAD UK footage) and vice versa

---

## Setup and Download Instructions

### Prerequisites

```bash
pip install opencv-python numpy matplotlib
# ffmpeg required for frame extraction (not available as pip package)
sudo apt install ffmpeg   # Ubuntu/Debian
```

### Step 1 — Clone the Repository

```bash
git clone https://github.com/salmank255/Road-waymo-dataset.git /repos/ROAD_plusplus
```

### Step 2 — Register and Download Data

The videos and annotation JSON are hosted in the **Community Contributions** section of the Waymo Open Dataset:

1. Register at https://waymo.com/open/ (free for academic use)
2. Accept the [Waymo Open Dataset License Agreement for Non-Commercial Use](https://waymo.com/open/terms/)
3. Navigate to the [Waymo Download page](https://waymo.com/open/download/)
4. Locate **Community Contributions** and find the ROAD++ / Road-Waymo entry
5. Download `road_waymo_trainval_v1.0.json` (~1 GB) and all `train_*.mp4` video files

### Step 3 — Directory Setup

```bash
mkdir -p road-waymo/videos
mv road_waymo_trainval_v1.0.json road-waymo/
mv train_*.mp4 road-waymo/videos/
```

### Step 4 — Extract Frames

```bash
python extract_videos2jpgs.py road-waymo/
# Creates road-waymo/rgb-images/train_00000/00001.jpg ... for all 798 videos
```

### Step 5 — Load Annotations

```python
import json
with open('road-waymo/road_waymo_trainval_v1.0.json') as f:
    data = json.load(f)

db = data['db']
# Iterate train split
for vname, vdata in db.items():
    if 'train' in vdata['split_ids']:
        # Process training video
        pass
```

### Step 6 — Train / Evaluate

Use the [ROAD++ Baseline](https://github.com/salmank255/ROAD_plus_plus_Baseline) (3D-RetinaNet):

```bash
git clone https://github.com/salmank255/ROAD_plus_plus_Baseline.git
cd ROAD_plus_plus_Baseline
# Follow the baseline README
```

---

## Compare and Contrast: ROAD++ vs. JAAD vs. PIE

### Side-by-Side Comparison Table

| Dimension | ROAD++ (Road-Waymo) | JAAD | PIE |
|-----------|---------------------|------|-----|
| **Year** | 2023–2024 | 2016 (orig.); 2019 (v2.0) | 2019 |
| **Primary task** | Multi-agent spatiotemporal action + situation detection | Pedestrian crossing-intent prediction | Pedestrian intent + trajectory prediction |
| **Agent scope** | All road agents (10 classes incl. traffic lights) | Pedestrians + bystanders | Pedestrians + bystanders |
| **Videos / Clips** | 798 annotated (+ 202 unannotated test) | 346 clips | 53 clips |
| **Total frames** | 158,081 | 82,032 | 740,901 |
| **Video resolution** | 1920 × 1280 | 1920 × 1080 | 1920 × 1080 |
| **Frame rate** | 10 FPS | ~30 FPS | 30 FPS |
| **Bounding box annotations** | 3,304,353 | 391,038 | ~740,901 frame-level |
| **Annotated agent tracks** | 41,935 agent tubes | 2,786 ped tracks (686 behavioral) | 1,842 ped tracks |
| **Per-box behavioral labels** | Agent type, action (22-class, multi-label), location (16-class, multi-label) | action, look, cross, occlusion, nod, hand_gesture, reaction | action, look, cross, gesture, occlusion |
| **Intent / crossing label** | None | Binary crossing outcome per pedestrian | `intention_prob` (0–1 continuous) + binary `crossing` |
| **Compositional event labels** | Duplex (49 task classes), Triplet (86 task classes) | None | None |
| **Ego-vehicle data** | AV action labels (9 classes, per-frame) | 5-class action label only | OBD speed, GPS, accelerometer (x/y/z), gyroscope, heading |
| **Crossing scene context** | Agent location labels (16 classes per box) | road_type, ped_crossing, ped_sign, stop_sign, traffic_light | intersection type, signalization, num_lanes, traffic_direction |
| **Per-pedestrian demographics** | None | age (4 levels), gender, group_size | age, gender |
| **Crossing key-point labels** | None | crossing_point, decision_point | critical_point, crossing_point, exp_start_point |
| **Appearance annotations** | None | 24 binary per-frame attrs (pose, clothing, accessories) | None |
| **Driver reaction labels** | None | Yes (clear_path, speed_up, slow_down) | None |
| **Traffic light labels** | Yes (TL agent class with Red/Amber/Green actions) | Yes (traffic layer: red/green) | Scene attribute only |
| **Temporal structure** | Spatiotemporal agent tubes (UUID-linked) | Pedestrian tracks (bbox sequences) | Pedestrian tracks (bbox sequences) |
| **Sensor modality** | RGB video (front camera only) | RGB video only | RGB video + OBD / GPS telemetry |
| **Video source** | Waymo Open Dataset (US cities, sensor rig) | Mixed dashcam (N. America + Europe) | Dashcam (Toronto, Canada) |
| **Annotation format** | Single combined JSON | XML — 5 separate files per video | XML — single hierarchical file per set |
| **Geographic diversity** | Multiple US cities | N. America + Europe | Toronto only |
| **Python API / loader** | No official API; baseline code provides DataLoader | Yes — `JAAD` class in `jaad_data.py` | Yes — `PIE` class in `pie_data.py` |
| **License** | CC BY-NC-SA 4.0 + Waymo non-commercial | MIT (code) + CC BY 4.0 (video) | MIT (code and annotations) |
| **Access** | Registration required (Waymo portal) | Free download | Registration required (YorkU) |
| **Approximate size** | ~1 GB JSON + ~20 GB videos (est.) | ~3.1 GB video | ~74 GB video |

### Narrative Analysis

**Scale and annotation philosophy.** ROAD++ and PIE are the two largest datasets in this group, but they measure scale very differently. PIE has 740,901 frames in 53 long continuous clips with 1,842 carefully annotated pedestrian tracks. ROAD++ has 158,081 frames in 798 short (~20 s) clips with 41,935 agent tubes across 10 object classes. ROAD++ has a much higher annotation density per frame — with an average of 21.5 boxes per annotated frame and up to 80+ per frame in busy intersection scenes — while PIE focuses depth of annotation on a narrower set of pedestrian subjects with richer per-pedestrian context.

**Agent scope and label richness.** ROAD++ is unique in covering all road agent types simultaneously, annotating cars, pedestrians, cyclists, motorcycles, buses, large vehicles, emergency vehicles, and traffic lights in a single unified framework. JAAD and PIE are pedestrian-centric; other agents are labeled only as bystanders with minimal annotation. For general scene understanding, ROAD++ is the only dataset in this group that provides actionable labels for non-pedestrian agents.

**Intent and crossing prediction.** ROAD++ has no pedestrian-level intent annotations — it is not designed for this task. JAAD provides binary crossing outcomes with `crossing_point` and `decision_point` temporal markers. PIE adds a uniquely valuable `intention_prob` continuous score derived from human crowd-sourcing experiments, enabling uncertainty-aware intent modeling. For pedestrian crossing intent research, JAAD and PIE are the appropriate datasets; ROAD++ contributes only through its richer scene context (agent positions, AV action, traffic light state) that could be used as auxiliary input.

**Ego-vehicle information.** PIE's OBD/GPS telemetry (speed, accelerometer, gyroscope, GPS) is the richest ego-vehicle context of the three datasets, enabling physics-aware trajectory prediction and intent modeling conditioned on vehicle dynamics. ROAD++ provides categorical AV action labels (9 classes) that capture what the vehicle is doing but not the magnitude. JAAD provides only a 5-class action label with no magnitude. For models that require vehicle speed as input, PIE is the only viable option; ROAD++ can characterize vehicle maneuvers categorically.

**Temporal granularity.** JAAD and PIE are recorded at ~30 FPS, giving finer-grained temporal resolution of pedestrian motion. ROAD++ clips are at 10 FPS (Waymo's keyframe rate), which limits motion blur recovery and fine-grained trajectory analysis but reduces data volume by 3×.

**Compositional event labels.** ROAD++'s duplex (49 classes) and triplet (86 classes) labels are a unique capability with no equivalent in JAAD or PIE. They enable direct classification of composite situations (e.g., "bus stopped at junction") as single output labels, which is useful for symbolic reasoning and situation-aware planning.

**Complementary use.** The three datasets are complementary for different subtasks. A researcher building a general autonomous driving perception system would use ROAD++ for multi-agent detection and scene event classification. A researcher studying pedestrian behavior and intent at crosswalks would use JAAD for behavioral annotation richness and PIE for continuous intent probability and vehicle telemetry. Joint training across all three datasets is conceptually motivated but practically difficult due to different frame rates, annotation schemas, and the absence of overlapping label spaces — the primary bridge is bounding box detection, which all three provide.

---

## References

### Primary Citation

```bibtex
@article{salmank2024roadwaymo,
  title   = {ROAD-Waymo: Action Awareness at Scale for Autonomous Driving},
  author  = {Khan, Salman and Singh, Gurkirt and Cuzzolin, Fabio},
  journal = {arXiv preprint arXiv:2411.01683},
  year    = {2024}
}
```

### Original ROAD Dataset

```bibtex
@article{singh2022road,
  title     = {ROAD: The ROad event Awareness Dataset for Autonomous Driving},
  author    = {Singh, Gurkirt and Akrigg, Stephen and Di Maio, Manuele and
               Fontana, Valentina and Alitappeh, Reza Javanmard and Saha, Suman and
               Cuzzolin, Fabio},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year      = {2022},
  publisher = {IEEE}
}
```

### Waymo Open Dataset

```bibtex
@inproceedings{sun2020scalability,
  title     = {Scalability in Perception for Autonomous Driving: Waymo Open Dataset},
  author    = {Sun, Pei and Kretzschmar, Henrik and Dotiwalla, Xerxes and others},
  booktitle = {CVPR},
  year      = {2020}
}
```

### ROAD++ Challenge

- Challenge page: https://sites.google.com/view/road-plus-plus/home
- Baseline code (3D-RetinaNet): https://github.com/salmank255/ROAD_plus_plus_Baseline
- This repository: https://github.com/salmank255/Road-waymo-dataset

### Data Attribution

ROAD++ dataset was made using the [Waymo Open Dataset](https://waymo.com/open/), provided by Waymo LLC under the [Waymo Open Dataset License Agreement for Non-Commercial Use](https://waymo.com/open/terms/).

- Visual AI Lab (Oxford Brookes): https://cms.brookes.ac.uk/staff/FabioCuzzolin/
- Original ROAD GitHub: https://github.com/gurkirt/road-dataset

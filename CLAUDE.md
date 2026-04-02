# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PhD research on **logic-constrained scene understanding and reasoning for autonomous driving**, built on the ROAD-Waymo (ROAD++) dataset. The primary novel contribution (Approach 3) is fine-tuning a generative VLM (LLaVA/InstructBLIP) with t-norm constraint losses derived from ROAD-Waymo's compositional label structure, producing both structured labels and natural language reasoning.

See `APPROACHES.md` for the full research roadmap (6 approaches from baseline replication to JEPA+VLM hybrid). See `ROAD_plusplus_summary.md` for dataset documentation.

## Running Scripts

No build system. All scripts run directly with Python:

```bash
# Dataset analysis
python analysis/compute_stats.py        # Load annotation JSON, compute statistics → stats_full.json
python analysis/create_viz.py           # Generate frame grid PNGs and tube swimlane diagrams
python analysis/generate_real_frames.py # Extract pedestrian crossing keyframes

# Data utilities
python extract_videos2jpgs.py <data_dir>              # MP4 → JPEG frame sequences via ffmpeg
python frames_clips.py --MODE [all|train|val|test]    # JPEG frame dirs → MP4 videos
python convert_waymo_to_coco.py                        # Waymo TFRecord → COCO JSON
python plot_annotations.py                             # Draw bounding boxes on video frames
```

**Hardcoded data paths** (used across scripts):
- Annotation file: `/data/datasets/ROAD_plusplus/road_waymo_trainval_v1.0.json` (~1 GB)
- Videos: `/data/datasets/ROAD_plusplus/videos/`
- Viz output: `/data/repos/ROAD_plusplus/viz/` or `/data/repos/PedestrianIntent++/ROAD_plusplus/viz/`

**Dependencies** (no requirements.txt — inferred from imports): `tensorflow`, `waymo_open_dataset`, `opencv-python`, `numpy`, `matplotlib`

## Dataset Structure

The annotation JSON has this top-level structure:
```python
{
  "agent_labels": [...],      # 10 agent classes (Ped, Car, Cyc, LarVeh, ...)
  "action_labels": [...],     # 19 action classes
  "loc_labels": [...],        # 10 location classes
  "duplex_labels": [...],     # 39 valid agent+action combinations (out of 242 possible)
  "triplet_labels": [...],    # 68 valid agent+action+location combinations (out of 3,872 possible)
  "av_action_labels": [...],  # 6 ego-vehicle actions
  "duplex_childs": {...},     # validity mappings — which actions are valid per agent
  "triplet_childs": {...},    # validity mappings — which locations are valid per duplex
  "db": {
    "train_00000": {
      "numf": 198,            # frame count per video
      "agent_tubes": {...},   # per-agent bounding box tracks
      "action_tubes": {...},
      "location_tubes": {...},
      ...
    },
    ...
  }
}
```

Dataset stats: 798 videos (600 train / 198 val / 202 test), 153,534 annotated frames, 3.3M bounding boxes.

## Architecture Context

The `duplex_childs` and `triplet_childs` mappings encode the **neuro-symbolic constraints**: they define which label combinations are semantically valid. For Approaches 2–6, these validity mappings are used to construct t-norm constraint loss terms that penalize co-prediction of invalid label combinations (e.g., Łukasiewicz t-norm: `violation = max(0, p(LarVeh) + p(Xing) - 1)`).

The baseline code for Approaches 1 & 2 is at: https://github.com/salmank255/ROAD_plus_plus_Baseline

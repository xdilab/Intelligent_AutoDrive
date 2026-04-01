#!/usr/bin/env python3
"""
ROAD++ (Road-Waymo) Dataset Statistics Extractor
Loads the 1 GB annotation JSON once and computes all statistics.
"""
import json
import sys
import os
import statistics
from collections import defaultdict, Counter
import time

ANNO_FILE = "/data/datasets/ROAD_plusplus/road_waymo_trainval_v1.0.json"
OUT_FILE = "/data/repos/ROAD_plusplus/analysis/stats.json"

print(f"Loading {ANNO_FILE} ...", flush=True)
t0 = time.time()
with open(ANNO_FILE, "r") as f:
    data = json.load(f)
t1 = time.time()
print(f"Loaded in {t1-t0:.1f}s", flush=True)

# ──────────────────────────────────────────────
# TOP-LEVEL KEYS
# ──────────────────────────────────────────────
top_keys = list(data.keys())
print("Top-level keys:", top_keys)

label_types      = data.get("label_types", [])
agent_labels     = data.get("agent_labels", [])
action_labels    = data.get("action_labels", [])
loc_labels       = data.get("loc_labels", [])
duplex_labels    = data.get("duplex_labels", [])
triplet_labels   = data.get("triplet_labels", [])
av_action_labels = data.get("av_action_labels", [])

all_agent_labels   = data.get("all_agent_labels", [])
all_action_labels  = data.get("all_action_labels", [])
all_loc_labels     = data.get("all_loc_labels", [])
all_duplex_labels  = data.get("all_duplex_labels", [])
all_triplet_labels = data.get("all_triplet_labels", [])
all_av_action_labels = data.get("all_av_action_labels", [])
all_input_labels   = data.get("all_input_labels", [])

duplex_childs  = data.get("duplex_childs", {})
triplet_childs = data.get("triplet_childs", {})

db = data["db"]

print(f"label_types: {label_types}")
print(f"agent_labels ({len(agent_labels)}): {agent_labels}")
print(f"action_labels ({len(action_labels)}): {action_labels}")
print(f"loc_labels ({len(loc_labels)}): {loc_labels}")
print(f"duplex_labels ({len(duplex_labels)}): {duplex_labels}")
print(f"triplet_labels ({len(triplet_labels)}): {triplet_labels}")
print(f"av_action_labels ({len(av_action_labels)}): {av_action_labels}")
print()
print(f"all_agent_labels ({len(all_agent_labels)}): {all_agent_labels}")
print(f"all_action_labels ({len(all_action_labels)}): {all_action_labels}")
print(f"all_loc_labels ({len(all_loc_labels)}): {all_loc_labels}")
print(f"all_duplex_labels ({len(all_duplex_labels)}): {all_duplex_labels}")
print(f"all_triplet_labels ({len(all_triplet_labels)}): {all_triplet_labels}")
print(f"all_av_action_labels ({len(all_av_action_labels)}): {all_av_action_labels}")
print(f"all_input_labels ({len(all_input_labels)}): {all_input_labels}")
print()
print(f"duplex_childs sample (first 3): {dict(list(duplex_childs.items())[:3])}")
print(f"triplet_childs sample (first 3): {dict(list(triplet_childs.items())[:3])}")

# ──────────────────────────────────────────────
# PER-VIDEO / SPLIT STATISTICS
# ──────────────────────────────────────────────
print("\n--- Per-video statistics ---", flush=True)

video_names  = list(db.keys())
total_videos = len(video_names)
print(f"Total videos in db: {total_videos}")

# Print first few video keys to understand naming convention
print(f"First 10 video keys: {video_names[:10]}")
print(f"Last 10 video keys: {video_names[-10:]}")

split_counts = Counter()
video_splits = {}  # video_name -> list of splits
numf_list = []
split_numf = defaultdict(list)

for vname, vdata in db.items():
    split_ids = vdata.get("split_ids", [])
    # split_ids could be a list or a string
    if isinstance(split_ids, str):
        split_ids = [split_ids]
    video_splits[vname] = split_ids
    for s in split_ids:
        split_counts[s] += 1

    numf = vdata.get("numf", 0)
    numf_list.append(numf)
    for s in split_ids:
        split_numf[s].append(numf)

print(f"Split counts: {dict(split_counts)}")
print(f"Total numf (frames): {sum(numf_list)}")
print(f"numf min={min(numf_list)}, max={max(numf_list)}, mean={statistics.mean(numf_list):.1f}, median={statistics.median(numf_list)}")
for s, nlist in split_numf.items():
    print(f"  Split '{s}': {len(nlist)} videos, total frames={sum(nlist)}, mean numf={statistics.mean(nlist):.1f}")

# ──────────────────────────────────────────────
# ANNOTATION VOLUMES
# ──────────────────────────────────────────────
print("\n--- Annotation volumes ---", flush=True)

total_annotated_frames = 0
total_boxes = 0
agent_tube_count = 0
action_tube_count = 0
loc_tube_count = 0
duplex_tube_count = 0
triplet_tube_count = 0
av_action_tube_count = 0

# Per-class counts
agent_class_counts  = Counter()
action_class_counts = Counter()
loc_class_counts    = Counter()
duplex_class_counts = Counter()
triplet_class_counts = Counter()
av_action_frame_counts = Counter()

# Tube lengths per class
agent_tube_lengths_by_class = defaultdict(list)
action_tube_lengths_by_class = defaultdict(list)

# Video-level tube count for reporting
agent_tubes_per_video = []

# Also track frame-level AV action
total_frames_seen = 0

for vname, vdata in db.items():
    # Tube counts
    agent_tubes  = vdata.get("agent_tubes", {})
    action_tubes = vdata.get("action_tubes", {})
    loc_tubes    = vdata.get("loc_tubes", {})
    duplex_tubes = vdata.get("duplex_tubes", {})
    triplet_tubes = vdata.get("triplet_tubes", {})
    av_action_tubes = vdata.get("av_action_tubes", {})

    agent_tube_count  += len(agent_tubes)
    action_tube_count += len(action_tubes)
    loc_tube_count    += len(loc_tubes)
    duplex_tube_count += len(duplex_tubes)
    triplet_tube_count += len(triplet_tubes)
    av_action_tube_count += len(av_action_tubes)

    agent_tubes_per_video.append(len(agent_tubes))

    # Agent tube lengths and class distribution
    for tube_id, tube_data in agent_tubes.items():
        label_id = tube_data.get("label_id", -1)
        tube_annos = tube_data.get("annos", {})
        tube_len = len(tube_annos)
        agent_tube_lengths_by_class[label_id].append(tube_len)

    # Action tube lengths
    for tube_id, tube_data in action_tubes.items():
        label_id = tube_data.get("label_id", -1)
        tube_annos = tube_data.get("annos", {})
        tube_len = len(tube_annos)
        action_tube_lengths_by_class[label_id].append(tube_len)

    # Frames
    frames_dict = vdata.get("frames", {})
    for frame_id, frame_data in frames_dict.items():
        total_frames_seen += 1
        annotated = frame_data.get("annotated", 0)
        if annotated == 1:
            total_annotated_frames += 1

        # AV action per frame
        av_action_ids = frame_data.get("av_action_ids", [])
        for aid in av_action_ids:
            av_action_frame_counts[aid] += 1

        # Box annotations
        annos = frame_data.get("annos", {})
        for anno_key, anno_val in annos.items():
            # anno_val is the annotation key string (e.g. 'b19111')
            # We need to look it up - but in this format, annos maps frame_unique_id -> anno_key
            # The actual box data is stored elsewhere - let's check the structure
            total_boxes += 1
            # anno_val might be a string key or a dict
            if isinstance(anno_val, dict):
                # Direct box data
                agent_ids  = anno_val.get("agent_ids", [])
                action_ids = anno_val.get("action_ids", [])
                loc_ids    = anno_val.get("loc_ids", [])
                duplex_ids = anno_val.get("duplex_ids", [])
                triplet_ids = anno_val.get("triplet_ids", [])
                for aid in agent_ids:
                    agent_class_counts[aid] += 1
                for aid in action_ids:
                    action_class_counts[aid] += 1
                for lid in loc_ids:
                    loc_class_counts[lid] += 1
                for did in duplex_ids:
                    duplex_class_counts[did] += 1
                for tid in triplet_ids:
                    triplet_class_counts[tid] += 1

print(f"Total frames (numf summed): {sum(numf_list)}")
print(f"Total frames seen in frames dict: {total_frames_seen}")
print(f"Total annotated frames (annotated==1): {total_annotated_frames}")
print(f"Total boxes in annos dicts: {total_boxes}")
print()
print(f"Agent tube count: {agent_tube_count}")
print(f"Action tube count: {action_tube_count}")
print(f"Loc tube count: {loc_tube_count}")
print(f"Duplex tube count: {duplex_tube_count}")
print(f"Triplet tube count: {triplet_tube_count}")
print(f"AV action tube count: {av_action_tube_count}")
print()
print(f"Agent class counts (by id): {dict(agent_class_counts)}")
print(f"Action class counts (by id): {dict(action_class_counts)}")
print(f"Loc class counts (by id): {dict(loc_class_counts)}")
print(f"Duplex class counts (by id): {dict(duplex_class_counts)}")
print(f"AV action frame counts (by id): {dict(av_action_frame_counts)}")

# Sample frame structure to understand annotation format
print("\n--- Inspecting first annotated frame structure ---")
for vname, vdata in list(db.items())[:5]:
    frames_dict = vdata.get("frames", {})
    for frame_id, frame_data in frames_dict.items():
        annos = frame_data.get("annos", {})
        if annos:
            print(f"Video: {vname}, Frame: {frame_id}")
            print(f"  Frame keys: {list(frame_data.keys())}")
            print(f"  annotated: {frame_data.get('annotated')}")
            print(f"  av_action_ids: {frame_data.get('av_action_ids')}")
            print(f"  annos type sample (first item): {list(annos.items())[:2]}")
            # Check what type the anno value is
            first_val = list(annos.values())[0]
            print(f"  First anno value type: {type(first_val)}, value: {first_val}")
            break
    break

# ──────────────────────────────────────────────
# Understand the actual annotation structure
# The annos in frames may store data differently
# ──────────────────────────────────────────────
print("\n--- Digging into annotation structure ---")
found = False
for vname, vdata in db.items():
    frames_dict = vdata.get("frames", {})
    for frame_id, frame_data in frames_dict.items():
        annos = frame_data.get("annos", {})
        if annos:
            for key, val in annos.items():
                print(f"  annos['{key}'] = type={type(val)}")
                if isinstance(val, dict):
                    print(f"    dict keys: {list(val.keys())}")
                    print(f"    full val: {val}")
                elif isinstance(val, str):
                    print(f"    string value: {val}")
                    # This is a reference key - need to find where the box data is stored
                    # Check if there's a separate 'annos_db' or similar
                    print(f"    Top-level keys: {top_keys}")
                found = True
                break
        if found:
            break
    if found:
        break

# ──────────────────────────────────────────────
# Tube length statistics
# ──────────────────────────────────────────────
print("\n--- Tube length statistics ---")
all_agent_lengths = [l for lengths in agent_tube_lengths_by_class.values() for l in lengths]
if all_agent_lengths:
    print(f"Agent tube lengths: min={min(all_agent_lengths)}, max={max(all_agent_lengths)}, mean={statistics.mean(all_agent_lengths):.1f}, median={statistics.median(all_agent_lengths)}")

print(f"Agent tubes per video: min={min(agent_tubes_per_video)}, max={max(agent_tubes_per_video)}, mean={statistics.mean(agent_tubes_per_video):.1f}")

# ──────────────────────────────────────────────
# SAVE STATS
# ──────────────────────────────────────────────
stats = {
    "top_keys": top_keys,
    "label_types": label_types,
    "agent_labels": agent_labels,
    "action_labels": action_labels,
    "loc_labels": loc_labels,
    "duplex_labels": duplex_labels,
    "triplet_labels": triplet_labels,
    "av_action_labels": av_action_labels,
    "all_agent_labels": all_agent_labels,
    "all_action_labels": all_action_labels,
    "all_loc_labels": all_loc_labels,
    "all_duplex_labels": all_duplex_labels,
    "all_triplet_labels": all_triplet_labels,
    "all_av_action_labels": all_av_action_labels,
    "all_input_labels": all_input_labels,
    "duplex_childs": duplex_childs,
    "triplet_childs": triplet_childs,
    "total_videos": total_videos,
    "split_counts": dict(split_counts),
    "total_frames_numf": sum(numf_list),
    "total_frames_in_frames_dict": total_frames_seen,
    "total_annotated_frames": total_annotated_frames,
    "total_boxes_in_annos": total_boxes,
    "numf_min": min(numf_list),
    "numf_max": max(numf_list),
    "numf_mean": statistics.mean(numf_list),
    "numf_median": statistics.median(numf_list),
    "agent_tube_count": agent_tube_count,
    "action_tube_count": action_tube_count,
    "loc_tube_count": loc_tube_count,
    "duplex_tube_count": duplex_tube_count,
    "triplet_tube_count": triplet_tube_count,
    "av_action_tube_count": av_action_tube_count,
    "agent_class_counts": {str(k): v for k, v in agent_class_counts.items()},
    "action_class_counts": {str(k): v for k, v in action_class_counts.items()},
    "loc_class_counts": {str(k): v for k, v in loc_class_counts.items()},
    "duplex_class_counts": {str(k): v for k, v in duplex_class_counts.items()},
    "triplet_class_counts": {str(k): v for k, v in triplet_class_counts.items()},
    "av_action_frame_counts": {str(k): v for k, v in av_action_frame_counts.items()},
    "split_numf": {s: {"count": len(nlist), "total_frames": sum(nlist), "mean_numf": statistics.mean(nlist)} for s, nlist in split_numf.items()},
    "agent_tube_lengths_by_class": {str(k): {"count": len(v), "mean": statistics.mean(v) if v else 0, "median": statistics.median(v) if v else 0, "min": min(v) if v else 0, "max": max(v) if v else 0} for k, v in agent_tube_lengths_by_class.items()},
    "all_agent_lengths_stats": {
        "min": min(all_agent_lengths) if all_agent_lengths else 0,
        "max": max(all_agent_lengths) if all_agent_lengths else 0,
        "mean": statistics.mean(all_agent_lengths) if all_agent_lengths else 0,
        "median": statistics.median(all_agent_lengths) if all_agent_lengths else 0,
    },
}

with open(OUT_FILE, "w") as f:
    json.dump(stats, f, indent=2)
print(f"\nStats saved to {OUT_FILE}")

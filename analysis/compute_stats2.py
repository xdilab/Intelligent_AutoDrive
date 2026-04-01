#!/usr/bin/env python3
"""
ROAD++ (Road-Waymo) Dataset Statistics Extractor - Phase 2
Loads the 1GB JSON and computes box-level and tube-level statistics.
"""
import json
import sys
import statistics
from collections import defaultdict, Counter
import time

ANNO_FILE = "/data/datasets/ROAD_plusplus/road_waymo_trainval_v1.0.json"
OUT_FILE = "/data/repos/ROAD_plusplus/analysis/stats_full.json"

print(f"Loading {ANNO_FILE} ...", flush=True)
t0 = time.time()
with open(ANNO_FILE, "r") as f:
    data = json.load(f)
t1 = time.time()
print(f"Loaded in {t1-t0:.1f}s", flush=True)

agent_labels     = data.get("agent_labels", [])
action_labels    = data.get("action_labels", [])
loc_labels       = data.get("loc_labels", [])
duplex_labels    = data.get("duplex_labels", [])
triplet_labels   = data.get("triplet_labels", [])
av_action_labels = data.get("av_action_labels", [])
all_agent_labels   = data.get("all_agent_labels", [])
all_action_labels  = data.get("all_action_labels", [])
all_loc_labels     = data.get("all_loc_labels", [])
all_av_action_labels = data.get("all_av_action_labels", [])
all_duplex_labels = data.get("all_duplex_labels", [])
all_triplet_labels = data.get("all_triplet_labels", [])
all_input_labels   = data.get("all_input_labels", [])
duplex_childs  = data.get("duplex_childs", {})
triplet_childs = data.get("triplet_childs", {})
label_types    = data.get("label_types", [])

db = data["db"]

# ──────────────────────────────────────────────
# Explore annotation structure for a few videos
# ──────────────────────────────────────────────
print("\n--- Exploring annotation structure ---")
for vidx, (vname, vdata) in enumerate(db.items()):
    if vidx >= 3:
        break
    frames_dict = vdata.get("frames", {})
    print(f"\nVideo: {vname}")
    print(f"  split_ids: {vdata.get('split_ids')}")
    print(f"  numf: {vdata.get('numf')}")
    print(f"  video keys: {list(vdata.keys())}")

    # Find first annotated frame
    for frame_id, fdata in frames_dict.items():
        if fdata.get("annotated") == 1 and fdata.get("annos"):
            print(f"  Frame {frame_id} (annotated):")
            print(f"    frame keys: {list(fdata.keys())}")
            print(f"    av_action_ids: {fdata.get('av_action_ids')}")
            annos = fdata.get("annos", {})
            print(f"    #annos: {len(annos)}")
            # Show first few anno entries
            for k, v in list(annos.items())[:2]:
                print(f"    annos['{k}'] = type:{type(v).__name__}, value:{v}")
                if isinstance(v, dict):
                    print(f"      dict keys: {list(v.keys())}")
                    print(f"      agent_ids: {v.get('agent_ids')}")
                    print(f"      action_ids: {v.get('action_ids')}")
                    print(f"      loc_ids: {v.get('loc_ids')}")
                    print(f"      duplex_ids: {v.get('duplex_ids')}")
                    print(f"      triplet_ids: {v.get('triplet_ids')}")
                    print(f"      box: {v.get('box')}")
                    print(f"      tube_uid: {v.get('tube_uid')}")
            break

# ──────────────────────────────────────────────
# Per-video statistics
# ──────────────────────────────────────────────
print("\n--- Computing per-video statistics ---", flush=True)

split_counts = Counter()
video_splits = {}
numf_list = []
split_numf = defaultdict(list)

for vname, vdata in db.items():
    split_ids = vdata.get("split_ids", [])
    if isinstance(split_ids, str):
        split_ids = [split_ids]
    video_splits[vname] = split_ids
    for s in split_ids:
        split_counts[s] += 1
    numf = vdata.get("numf", 0)
    numf_list.append(numf)
    for s in split_ids:
        split_numf[s].append(numf)

print(f"Total videos: {len(numf_list)}")
print(f"Split counts: {dict(split_counts)}")
print(f"Total frames (numf summed): {sum(numf_list)}")
print(f"numf stats: min={min(numf_list)}, max={max(numf_list)}, mean={statistics.mean(numf_list):.1f}, median={statistics.median(numf_list)}")
for s in sorted(split_numf.keys()):
    nlist = split_numf[s]
    print(f"  Split '{s}': {len(nlist)} videos, total_frames={sum(nlist)}")

# ──────────────────────────────────────────────
# Tube counts and tube-level stats
# ──────────────────────────────────────────────
print("\n--- Computing tube counts ---", flush=True)

agent_tube_count  = 0
action_tube_count = 0
loc_tube_count    = 0
duplex_tube_count = 0
triplet_tube_count = 0
av_action_tube_count = 0

agent_tube_lengths_by_class  = defaultdict(list)
action_tube_lengths_by_class = defaultdict(list)
loc_tube_lengths_by_class    = defaultdict(list)
duplex_tube_lengths_by_class = defaultdict(list)
triplet_tube_lengths_by_class = defaultdict(list)

agent_tube_class_counts = Counter()  # how many tubes per agent class

for vname, vdata in db.items():
    for tname, tdict in vdata.get("agent_tubes", {}).items():
        agent_tube_count += 1
        lid = tdict.get("label_id", -1)
        tlen = len(tdict.get("annos", {}))
        agent_tube_lengths_by_class[lid].append(tlen)
        agent_tube_class_counts[lid] += 1

    for tname, tdict in vdata.get("action_tubes", {}).items():
        action_tube_count += 1
        lid = tdict.get("label_id", -1)
        tlen = len(tdict.get("annos", {}))
        action_tube_lengths_by_class[lid].append(tlen)

    for tname, tdict in vdata.get("loc_tubes", {}).items():
        loc_tube_count += 1
        lid = tdict.get("label_id", -1)
        tlen = len(tdict.get("annos", {}))
        loc_tube_lengths_by_class[lid].append(tlen)

    for tname, tdict in vdata.get("duplex_tubes", {}).items():
        duplex_tube_count += 1
        lid = tdict.get("label_id", -1)
        tlen = len(tdict.get("annos", {}))
        duplex_tube_lengths_by_class[lid].append(tlen)

    for tname, tdict in vdata.get("triplet_tubes", {}).items():
        triplet_tube_count += 1
        lid = tdict.get("label_id", -1)
        tlen = len(tdict.get("annos", {}))
        triplet_tube_lengths_by_class[lid].append(tlen)

    av_action_tube_count += len(vdata.get("av_action_tubes", {}))

print(f"agent_tube_count: {agent_tube_count}")
print(f"action_tube_count: {action_tube_count}")
print(f"loc_tube_count: {loc_tube_count}")
print(f"duplex_tube_count: {duplex_tube_count}")
print(f"triplet_tube_count: {triplet_tube_count}")
print(f"av_action_tube_count: {av_action_tube_count}")

all_agent_len = [l for lens in agent_tube_lengths_by_class.values() for l in lens]
if all_agent_len:
    print(f"Agent tube length: min={min(all_agent_len)}, max={max(all_agent_len)}, mean={statistics.mean(all_agent_len):.1f}, median={statistics.median(all_agent_len)}")

print(f"\nAgent tubes by class ID:")
for k in sorted(agent_tube_class_counts.keys()):
    cname = agent_labels[k] if 0 <= k < len(agent_labels) else f"id_{k}"
    lens = agent_tube_lengths_by_class[k]
    mean_len = statistics.mean(lens) if lens else 0
    med_len = statistics.median(lens) if lens else 0
    print(f"  {k} ({cname}): {agent_tube_class_counts[k]} tubes, mean_len={mean_len:.1f}, med_len={med_len:.1f}")

# ──────────────────────────────────────────────
# Frame + Box level stats
# ──────────────────────────────────────────────
print("\n--- Computing frame + box stats ---", flush=True)

total_annotated_frames  = 0
total_boxes             = 0
agent_box_counts   = Counter()   # per agent class
action_box_counts  = Counter()   # per action class (multi-label)
loc_box_counts     = Counter()   # per loc class (multi-label)
duplex_box_counts  = Counter()
triplet_box_counts = Counter()
av_action_frame_counts = Counter()

# Also need to handle two possible annotation formats:
# Format A: annos[key] = dict with box data directly
# Format B: annos[key] = string reference to another dict
# From exploration above we'll detect which is present

anno_format = None  # will be 'direct' or 'reference'

for vname, vdata in db.items():
    frames_dict = vdata.get("frames", {})
    for frame_id, fdata in frames_dict.items():
        annotated = fdata.get("annotated", 0)
        if annotated == 1:
            total_annotated_frames += 1

        av_ids = fdata.get("av_action_ids", [])
        for aid in av_ids:
            av_action_frame_counts[aid] += 1

        annos = fdata.get("annos", {})
        for anno_key, anno_val in annos.items():
            if isinstance(anno_val, dict):
                anno_format = 'direct'
                total_boxes += 1
                for aid in anno_val.get("agent_ids", []):
                    agent_box_counts[aid] += 1
                for aid in anno_val.get("action_ids", []):
                    action_box_counts[aid] += 1
                for lid in anno_val.get("loc_ids", []):
                    loc_box_counts[lid] += 1
                for did in anno_val.get("duplex_ids", []):
                    duplex_box_counts[did] += 1
                for tid in anno_val.get("triplet_ids", []):
                    triplet_box_counts[tid] += 1
            elif isinstance(anno_val, str):
                anno_format = 'reference'
                # The value is a reference key - need different lookup
                total_boxes += 1

print(f"\nAnnotation format detected: {anno_format}")
print(f"Total annotated frames: {total_annotated_frames}")
print(f"Total boxes: {total_boxes}")
print(f"Total agent label instances: {sum(agent_box_counts.values())}")
print(f"Total action label instances: {sum(action_box_counts.values())}")
print(f"Total loc label instances: {sum(loc_box_counts.values())}")
print(f"Total duplex label instances: {sum(duplex_box_counts.values())}")
print(f"Total triplet label instances: {sum(triplet_box_counts.values())}")

print(f"\nAgent class distribution:")
for k in sorted(agent_box_counts.keys()):
    cname = agent_labels[k] if 0 <= k < len(agent_labels) else f"id_{k}"
    print(f"  {k} ({cname}): {agent_box_counts[k]}")

print(f"\nAction class distribution:")
for k in sorted(action_box_counts.keys()):
    cname = action_labels[k] if 0 <= k < len(action_labels) else f"id_{k}"
    print(f"  {k} ({cname}): {action_box_counts[k]}")

print(f"\nLocation class distribution:")
for k in sorted(loc_box_counts.keys()):
    cname = loc_labels[k] if 0 <= k < len(loc_labels) else f"id_{k}"
    print(f"  {k} ({cname}): {loc_box_counts[k]}")

print(f"\nAV action frame distribution:")
for k in sorted(av_action_frame_counts.keys()):
    cname = av_action_labels[k] if 0 <= k < len(av_action_labels) else f"id_{k}"
    print(f"  {k} ({cname}): {av_action_frame_counts[k]}")

# ──────────────────────────────────────────────
# If format is 'reference', we need to handle differently
# Let's check if there's a top-level 'annos' dict or similar
# ──────────────────────────────────────────────
if anno_format == 'reference':
    print("\nFormat is reference - searching for annotation lookup dict...")
    top_keys = list(data.keys())
    print(f"Top-level keys: {top_keys}")
    # Try to find global annos dict
    for k in top_keys:
        if k not in ('db', 'label_types', 'agent_labels', 'action_labels', 'loc_labels',
                     'duplex_labels', 'triplet_labels', 'av_action_labels', 'old_loc_labels',
                     'all_duplex_labels', 'all_triplet_labels', 'all_loc_labels', 'all_agent_labels',
                     'all_action_labels', 'duplex_childs', 'triplet_childs',
                     'all_input_labels', 'all_av_action_labels'):
            print(f"  Unexpected top-level key: {k}")

# ──────────────────────────────────────────────
# Frame width/height
# ──────────────────────────────────────────────
print("\n--- Checking frame dimensions ---")
widths = []
heights = []
checked = 0
for vname, vdata in db.items():
    for fid, fdata in vdata.get("frames", {}).items():
        w = fdata.get("width")
        h = fdata.get("height")
        if w and h:
            widths.append(w)
            heights.append(h)
            checked += 1
            if checked >= 100:
                break
    if checked >= 100:
        break

if widths:
    print(f"Width values (sample of {checked}): unique={set(widths)}")
    print(f"Height values (sample of {checked}): unique={set(heights)}")

# ──────────────────────────────────────────────
# Check test videos - do they have annotations?
# ──────────────────────────────────────────────
print("\n--- Checking split annotation completeness ---")
split_annotated_frames = defaultdict(int)
split_total_boxes = defaultdict(int)
for vname, vdata in db.items():
    split_ids = vdata.get("split_ids", [])
    if isinstance(split_ids, str):
        split_ids = [split_ids]
    frames_dict = vdata.get("frames", {})
    for fid, fdata in frames_dict.items():
        if fdata.get("annotated") == 1:
            annos = fdata.get("annos", {})
            for s in split_ids:
                split_annotated_frames[s] += 1
                split_total_boxes[s] += len(annos)

print("Per-split annotated frame counts:")
for s in sorted(split_annotated_frames.keys()):
    print(f"  {s}: {split_annotated_frames[s]} annotated frames, {split_total_boxes[s]} boxes")

# ──────────────────────────────────────────────
# Save full stats
# ──────────────────────────────────────────────

def tube_stats(lengths_by_class, label_list):
    result = {}
    all_len = [l for ls in lengths_by_class.values() for l in ls]
    result["total"] = sum(len(ls) for ls in lengths_by_class.values())
    if all_len:
        result["len_min"] = min(all_len)
        result["len_max"] = max(all_len)
        result["len_mean"] = round(statistics.mean(all_len), 2)
        result["len_median"] = statistics.median(all_len)
    result["by_class"] = {}
    for k, lens in lengths_by_class.items():
        cname = label_list[k] if 0 <= k < len(label_list) else f"id_{k}"
        result["by_class"][cname] = {
            "count": len(lens),
            "mean_len": round(statistics.mean(lens), 2) if lens else 0,
            "median_len": statistics.median(lens) if lens else 0,
            "min_len": min(lens) if lens else 0,
            "max_len": max(lens) if lens else 0,
        }
    return result

stats = {
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
    "total_videos": len(numf_list),
    "split_counts": dict(split_counts),
    "total_frames_numf": sum(numf_list),
    "numf_min": min(numf_list),
    "numf_max": max(numf_list),
    "numf_mean": round(statistics.mean(numf_list), 1),
    "numf_median": statistics.median(numf_list),
    "split_numf": {s: {"count": len(nlist), "total_frames": sum(nlist)} for s, nlist in split_numf.items()},
    "total_annotated_frames": total_annotated_frames,
    "split_annotated_frames": dict(split_annotated_frames),
    "split_total_boxes": dict(split_total_boxes),
    "total_boxes": total_boxes,
    "total_agent_label_instances": sum(agent_box_counts.values()),
    "total_action_label_instances": sum(action_box_counts.values()),
    "total_loc_label_instances": sum(loc_box_counts.values()),
    "total_duplex_label_instances": sum(duplex_box_counts.values()),
    "total_triplet_label_instances": sum(triplet_box_counts.values()),
    "agent_box_counts": {(agent_labels[k] if 0 <= k < len(agent_labels) else f"id_{k}"): v for k, v in sorted(agent_box_counts.items())},
    "action_box_counts": {(action_labels[k] if 0 <= k < len(action_labels) else f"id_{k}"): v for k, v in sorted(action_box_counts.items())},
    "loc_box_counts": {(loc_labels[k] if 0 <= k < len(loc_labels) else f"id_{k}"): v for k, v in sorted(loc_box_counts.items())},
    "duplex_box_counts": {(duplex_labels[k] if 0 <= k < len(duplex_labels) else f"id_{k}"): v for k, v in sorted(duplex_box_counts.items())},
    "av_action_frame_counts": {(av_action_labels[k] if 0 <= k < len(av_action_labels) else f"id_{k}"): v for k, v in sorted(av_action_frame_counts.items())},
    "agent_tube_stats": tube_stats(agent_tube_lengths_by_class, agent_labels),
    "action_tube_stats": tube_stats(action_tube_lengths_by_class, action_labels),
    "loc_tube_stats": tube_stats(loc_tube_lengths_by_class, loc_labels),
    "duplex_tube_stats": tube_stats(duplex_tube_lengths_by_class, duplex_labels),
    "triplet_tube_stats": tube_stats(triplet_tube_lengths_by_class, triplet_labels),
    "total_tube_counts": {
        "agent": agent_tube_count,
        "action": action_tube_count,
        "loc": loc_tube_count,
        "duplex": duplex_tube_count,
        "triplet": triplet_tube_count,
        "av_action": av_action_tube_count,
    },
    "anno_format": anno_format,
    "frame_width": list(set(widths))[0] if len(set(widths)) == 1 else widths[:5],
    "frame_height": list(set(heights))[0] if len(set(heights)) == 1 else heights[:5],
}

with open(OUT_FILE, "w") as f:
    json.dump(stats, f, indent=2)
print(f"\nFull stats saved to {OUT_FILE}")

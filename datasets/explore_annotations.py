"""
Annotation explorer for BDD-X and CoVLA datasets.
Outputs statistics to stdout and writes dataset_comparison.csv.

Usage:
    python explore_annotations.py
"""

import os
import csv
import json
from pathlib import Path
from collections import Counter, defaultdict

BDD_X_CSV   = "/data/datasets/BDD-X/BDD-X-Annotations_v1.csv"
BDD_X_TRAIN = "/data/datasets/BDD-X/train.txt"
BDD_X_VAL   = "/data/datasets/BDD-X/val.txt"
BDD_X_TEST  = "/data/datasets/BDD-X/test.txt"
COVLA_DIR   = "/data/datasets/CoVLA/mini"
OUTPUT_CSV  = "/data/repos/ROAD_Reason/datasets/dataset_comparison.csv"

# ─────────────────────────────────────────────────────────────────────────────
# BDD-X
# ─────────────────────────────────────────────────────────────────────────────

def explore_bddx():
    print("\n" + "="*60)
    print("BDD-X EXPLORATION")
    print("="*60)

    if not os.path.exists(BDD_X_CSV):
        print("ERROR: BDD-X CSV not found at", BDD_X_CSV)
        return {}

    rows = []
    with open(BDD_X_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        for row in reader:
            rows.append(row)

    print(f"\nCSV columns ({len(columns)}):")
    for c in columns:
        print(f"  {c}")

    print(f"\nTotal video rows: {len(rows)}")

    # Parse into flat action records
    actions = []
    for row in rows:
        video = row.get("Input.Video", "")
        for i in range(1, 16):
            action_text = row.get(f"Answer.{i}action", "").strip()
            just_text   = row.get(f"Answer.{i}justification", "").strip()
            start       = row.get(f"Answer.{i}start", "").strip()
            end         = row.get(f"Answer.{i}end", "").strip()
            if action_text:
                actions.append({
                    "video": video,
                    "action": action_text,
                    "justification": just_text,
                    "start": start,
                    "end": end,
                })

    print(f"Total annotated action segments: {len(actions)}")
    print(f"Average actions per video: {len(actions)/max(len(rows),1):.2f}")

    # Action word frequency
    action_words = Counter()
    for a in actions:
        words = a["action"].lower().split()
        for w in words:
            action_words[w] += 1

    print("\nTop 20 action words:")
    for word, count in action_words.most_common(20):
        print(f"  {word:20s} {count}")

    # Justification word count stats
    just_lengths = [len(a["justification"].split()) for a in actions if a["justification"]]
    if just_lengths:
        print(f"\nJustification word counts — min: {min(just_lengths)}  "
              f"max: {max(just_lengths)}  "
              f"mean: {sum(just_lengths)/len(just_lengths):.1f}")

    # Split sizes
    def count_lines(path):
        if not os.path.exists(path):
            return 0
        with open(path) as f:
            return sum(1 for line in f if line.strip())

    train_n = count_lines(BDD_X_TRAIN)
    val_n   = count_lines(BDD_X_VAL)
    test_n  = count_lines(BDD_X_TEST)
    print(f"\nSplits — train: {train_n}  val: {val_n}  test: {test_n}  total videos: {train_n+val_n+test_n}")

    # Sample rows
    print("\nSample action+justification pairs (first 5):")
    for a in actions[:5]:
        print(f"  ACTION:  {a['action']}")
        print(f"  JUSTIFY: {a['justification']}")
        print()

    # Segment duration stats
    durations = []
    for a in actions:
        try:
            durations.append(float(a["end"]) - float(a["start"]))
        except ValueError:
            pass
    if durations:
        print(f"Action segment duration — min: {min(durations):.1f}s  "
              f"max: {max(durations):.1f}s  "
              f"mean: {sum(durations)/len(durations):.1f}s")

    return {
        "total_videos": len(rows),
        "total_actions": len(actions),
        "avg_actions_per_video": f"{len(actions)/max(len(rows),1):.2f}",
        "train_videos": train_n,
        "val_videos": val_n,
        "test_videos": test_n,
        "annotation_format": "CSV (wide — 1 row per video, up to 15 actions)",
        "annotation_granularity": "Action-level (temporal segment)",
        "caption_types": "Action description + Justification (why)",
        "caption_generation": "Manual (human annotators, US driving rules)",
        "has_trajectory": "No",
        "has_ego_speed": "No (videos link to BDD100K which has speed)",
        "has_steering_angle": "No (videos link to BDD100K which has steering)",
        "has_brake_gas": "No",
        "has_traffic_lights": "No",
        "has_leading_vehicle": "No",
        "has_weather": "No (implicit in video)",
        "has_road_type": "No (implicit in video)",
        "camera_resolution": "~720p (BDD100K standard)",
        "collection_region": "USA (diverse)",
        "total_hours": "77+",
        "total_frames": "~8.4M",
        "license": "UC Berkeley — non-commercial research",
        "dsdag_y_relevance": "Action descriptions → Y node (what is happening)",
        "dsdag_w_relevance": "Justifications → W reason mode (why it happened)",
        "just_word_count_mean": f"{sum(just_lengths)/len(just_lengths):.1f}" if just_lengths else "N/A",
    }


# ─────────────────────────────────────────────────────────────────────────────
# CoVLA
# ─────────────────────────────────────────────────────────────────────────────

def parse_jsonl_concat(filepath, max_records=None):
    """Parse a file of concatenated JSON objects (CoVLA format — no newline delimiters)."""
    records = []
    with open(filepath) as f:
        content = f.read()
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(content):
        if max_records and len(records) >= max_records:
            break
        stripped = content[pos:].lstrip()
        if not stripped:
            break
        offset = len(content[pos:]) - len(stripped)
        try:
            obj, end = decoder.raw_decode(stripped)
            records.append(obj)
            pos += offset + end
        except json.JSONDecodeError:
            break
    return records


def explore_covla():
    print("\n" + "="*60)
    print("CoVLA EXPLORATION")
    print("="*60)

    if not os.path.exists(COVLA_DIR):
        print("ERROR: CoVLA mini dir not found at", COVLA_DIR)
        return {}

    # Directory layout
    subdirs = [d for d in os.listdir(COVLA_DIR)
               if os.path.isdir(os.path.join(COVLA_DIR, d))]
    print(f"\nSubdirectories: {sorted(subdirs)}")

    # Count scenes (one JSONL per scene per subdirectory)
    captions_dir = os.path.join(COVLA_DIR, "captions")
    states_dir   = os.path.join(COVLA_DIR, "states")
    fc_dir       = os.path.join(COVLA_DIR, "front_car")
    tl_dir       = os.path.join(COVLA_DIR, "traffic_lights")

    scene_files = sorted([f for f in os.listdir(captions_dir) if f.endswith(".jsonl")])
    n_scenes = len(scene_files)
    print(f"Scenes (JSONL files per subdir): {n_scenes}")

    # Count frames in first few scenes to estimate total
    sample_counts = []
    for fn in scene_files[:5]:
        recs = parse_jsonl_concat(os.path.join(captions_dir, fn))
        sample_counts.append(len(recs))
    frames_per_scene = sum(sample_counts) / len(sample_counts) if sample_counts else 0
    total_frames_est = int(frames_per_scene * n_scenes)
    print(f"Frames per scene (first 5): {sample_counts}")
    print(f"Estimated total frames in mini: {total_frames_est:,}")

    # ── captions schema ──
    print("\n--- captions/ schema (first record) ---")
    cap_record = parse_jsonl_concat(os.path.join(captions_dir, scene_files[0]), max_records=1)[0]
    for k, v in cap_record.items():
        print(f"  {k:30s} {type(v).__name__:10s}  {str(v)[:80]}")

    # Caption samples
    print("\nSample plain_caption (behavior → Y):")
    for fn in scene_files[:3]:
        r = parse_jsonl_concat(os.path.join(captions_dir, fn), max_records=1)[0]
        print(f"  {r.get('plain_caption','')[:100]}")

    print("\nSample risk (reason → W):")
    for fn in scene_files[:3]:
        r = parse_jsonl_concat(os.path.join(captions_dir, fn), max_records=1)[0]
        print(f"  {r.get('risk','')[:100]}")

    # ── states schema ──
    print("\n--- states/ schema (first record, ego_state keys) ---")
    state_record = parse_jsonl_concat(os.path.join(states_dir, scene_files[0]), max_records=1)[0]
    ego = state_record.get("ego_state", {})
    for k, v in ego.items():
        print(f"  ego_state.{k:30s} {type(v).__name__:10s}  {str(v)[:60]}")
    traj = state_record.get("trajectory", [])
    print(f"\n  trajectory: {len(traj)} points × {len(traj[0]) if traj else 0} dims "
          f"  (first: {traj[0] if traj else 'N/A'})")
    print(f"  trajectory_count: {state_record.get('trajectory_count')}")
    print(f"  image_path: {state_record.get('image_path','')}")
    for k in ["extrinsic_matrix", "intrinsic_matrix", "frame_id"]:
        v = state_record.get(k)
        if isinstance(v, list):
            print(f"  {k}: {len(v)}x{len(v[0]) if v and isinstance(v[0], list) else ''} matrix")
        else:
            print(f"  {k}: {v}")

    # ── front_car schema ──
    print("\n--- front_car/ schema (first record) ---")
    fc_records = parse_jsonl_concat(os.path.join(fc_dir, scene_files[0]), max_records=1)
    if fc_records:
        first = fc_records[0]
        # Each concat-JSON object is a list of per-frame dicts
        if isinstance(first, list) and first:
            first = first[0]
        if isinstance(first, dict):
            for k, v in first.items():
                print(f"  {k:30s} {type(v).__name__:10s}  {str(v)[:60]}")

    # ── traffic_lights schema ──
    print("\n--- traffic_lights/ schema (first record) ---")
    tl_records = parse_jsonl_concat(os.path.join(tl_dir, scene_files[0]), max_records=1)
    if tl_records:
        first = tl_records[0]
        if isinstance(first, list) and first:
            first = first[0]
        if isinstance(first, dict):
            for k, v in first.items():
                print(f"  {k:30s} {type(v).__name__:10s}  {str(v)[:60]}")

    # Weather / road distribution across scenes
    weather_counts = Counter()
    road_counts = Counter()
    for fn in scene_files:
        r = parse_jsonl_concat(os.path.join(captions_dir, fn), max_records=1)[0]
        weather_counts[r.get("weather", "unknown")] += 1
        road_counts[r.get("road", "unknown")] += 1
    print(f"\nWeather distribution (first frame per scene): {dict(weather_counts.most_common(5))}")
    print(f"Road type distribution (first frame per scene): {dict(road_counts.most_common(5))}")

    print(f"\nMetadata: {open(os.path.join(COVLA_DIR, 'metadata.json')).read().strip()}")

    return {
        "total_scenes_mini": n_scenes,
        "frames_per_scene": int(frames_per_scene),
        "total_frames_mini": total_frames_est,
        "annotation_format": "JSONL (concatenated JSON objects, one file per scene per modality)",
        "annotation_granularity": "Frame-level",
        "caption_types": "plain_caption (behavior) + rich_caption (extended) + risk (reason)",
        "caption_generation": "Auto (rule-based + VideoLLaMA2-7B; hallucination-mitigated)",
        "caption_field_behavior": "plain_caption",
        "caption_field_reasoning": "risk (extracted from rich_caption)",
        "has_trajectory": "Yes (60 pts × 3D in vehicle frame, ~3s at 20FPS)",
        "has_ego_speed": "Yes (vEgo, vEgoRaw in ego_state)",
        "has_steering_angle": "Yes (steeringAngleDeg in ego_state)",
        "has_brake_gas": "Yes (brake, brakePressed, gas, gasPressed in ego_state)",
        "has_traffic_lights": "Yes (class: green/red/amber, bbox per frame)",
        "has_leading_vehicle": "Yes (has_lead, lead_prob, lead_x/y, lead_speed_kmh)",
        "has_weather": "Yes (weather field in captions, with confidence rate)",
        "has_road_type": "Yes (road field in captions, with confidence rate)",
        "camera_resolution": "1928×1208 (H.265, 20FPS)",
        "collection_region": "Tokyo, Japan (urban, highway, residential, mountain)",
        "total_hours": "83.3",
        "total_frames": "6,000,000",
        "total_clips": "10,000 (30s each)",
        "train_split": "7,000 scenes (70%)",
        "val_split": "1,500 scenes (15%)",
        "test_split": "1,500 scenes (15%)",
        "license": "Turing Inc. — non-commercial, no redistribution",
        "dsdag_y_relevance": "plain_caption → Y node (what ego action is occurring)",
        "dsdag_w_relevance": "risk field → W reason mode (why driver should be careful)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Comparison CSV
# ─────────────────────────────────────────────────────────────────────────────

def write_comparison_csv(bddx_stats, covla_stats):
    print("\n" + "="*60)
    print("WRITING COMPARISON CSV")
    print("="*60)

    properties = [
        ("Dataset",                     "BDD-X",                        "CoVLA"),
        ("Full name",                   "Berkeley DeepDrive eXplanation","Comprehensive Vision-Language-Action"),
        ("Authors",                     "Kim et al. (UC Berkeley)",      "Arai et al. (Turing Inc.)"),
        ("Year",                        "2018",                          "2025"),
        ("Venue",                       "ECCV 2018",                     "arXiv 2408.10845"),
        ("Collection region",           bddx_stats.get("collection_region","USA (diverse)"), covla_stats.get("collection_region","Tokyo, Japan")),
        ("Total clips/scenes",          str(bddx_stats.get("total_videos","6,970")),          covla_stats.get("total_clips","10,000 (30s each)")),
        ("Total frames",                bddx_stats.get("total_frames","~8.4M"),               covla_stats.get("total_frames","6,000,000")),
        ("Total hours",                 bddx_stats.get("total_hours","77+"),                  covla_stats.get("total_hours","83.3")),
        ("Camera resolution",           bddx_stats.get("camera_resolution","~720p"),          covla_stats.get("camera_resolution","1928×1208 (H.265, 20FPS)")),
        ("Annotation format",           bddx_stats.get("annotation_format","CSV (wide)"),     covla_stats.get("annotation_format","Parquet (frame-level)")),
        ("Annotation granularity",      bddx_stats.get("annotation_granularity","Action-level (temporal segment)"), covla_stats.get("annotation_granularity","Frame-level")),
        ("Total annotated instances",   str(bddx_stats.get("total_actions","26,000+ actions")), covla_stats.get("total_frames","6,000,000 captions")),
        ("Caption types",               bddx_stats.get("caption_types","Action desc + Justification"), covla_stats.get("caption_types","Behavior + Reasoning")),
        ("Caption generation",          bddx_stats.get("caption_generation","Manual (human)"),  covla_stats.get("caption_generation","Auto (rule-based + VideoLLaMA2-7B)")),
        ("Action description field",    "Yes (Answer.Naction)",          "Yes (plain_caption)"),
        ("Reasoning/explanation field", "Yes (Answer.Njustification)",   "Yes (risk field from rich_caption)"),
        ("Mean justification length",   bddx_stats.get("just_word_count_mean","N/A") + " words", "frame-level (varies)"),
        ("Trajectory data",             bddx_stats.get("has_trajectory","No"),                covla_stats.get("has_trajectory","Yes (GPS/IMU, 3s horizon, 10 pts)")),
        ("Ego speed",                   bddx_stats.get("has_ego_speed","No"),                 covla_stats.get("has_ego_speed","Yes (vEgo, CAN bus)")),
        ("Steering angle",              bddx_stats.get("has_steering_angle","No"),            covla_stats.get("has_steering_angle","Yes (steeringAngleDeg, CAN bus)")),
        ("Brake / gas",                 bddx_stats.get("has_brake_gas","No"),                 covla_stats.get("has_brake_gas","Yes (brake, gas, CAN bus)")),
        ("Traffic light annotations",   bddx_stats.get("has_traffic_lights","No"),            covla_stats.get("has_traffic_lights","Yes (OpenLenda-s detector)")),
        ("Leading vehicle annotations", bddx_stats.get("has_leading_vehicle","No"),           covla_stats.get("has_leading_vehicle","Yes (radar+camera fusion)")),
        ("Weather in annotations",      bddx_stats.get("has_weather","No"),                   covla_stats.get("has_weather","Yes (in captions)")),
        ("Road type in annotations",    bddx_stats.get("has_road_type","No"),                 covla_stats.get("has_road_type","Yes (in captions)")),
        ("Train split",                 str(bddx_stats.get("train_videos","5,597")) + " videos", covla_stats.get("train_split","7,000 scenes (70%)")),
        ("Val split",                   str(bddx_stats.get("val_videos","717")) + " videos",    covla_stats.get("val_split","1,500 scenes (15%)")),
        ("Test split",                  str(bddx_stats.get("test_videos","656")) + " videos",   covla_stats.get("test_split","1,500 scenes (15%)")),
        ("License",                     bddx_stats.get("license","UC Berkeley non-commercial"), covla_stats.get("license","Turing Inc. non-commercial")),
        ("DSDAG Y-node relevance",      bddx_stats.get("dsdag_y_relevance","Action descriptions → Y"), covla_stats.get("dsdag_y_relevance","Behavior captions → Y")),
        ("DSDAG W reason mode relevance", bddx_stats.get("dsdag_w_relevance","Justifications → W"), covla_stats.get("dsdag_w_relevance","Reasoning captions → W")),
        ("Availability",                "GitHub clone + Google Drive (annotations only)",     "HuggingFace (gated, requires account)"),
        ("Video availability",          "S3 URLs in CSV (may be expired); BDD100K needed",    "Included in HuggingFace download"),
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Property", "BDD-X", "CoVLA"])
        for row in properties:
            writer.writerow(row)

    print(f"Written: {OUTPUT_CSV} ({len(properties)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bddx_stats  = explore_bddx()
    covla_stats = explore_covla()
    write_comparison_csv(bddx_stats, covla_stats)
    print("\nDone.")

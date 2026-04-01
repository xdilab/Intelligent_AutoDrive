#!/usr/bin/env python3
"""
ROAD++ Real Frame Visualizations
Generates one PNG per pedestrian tube at the Wait2X (or first crossing-action) keyframe.
Output: ROAD_plusplus/viz/frames/frame_XX_video_tube_wait2x.png

Pedestrian selection: finds ped tubes with Wait2X / XingFmLft / XingFmRht / Xing
across diverse videos, preferring tubes that have at least one Wait2X frame.
"""

import json
import os
import time
import cv2
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
ANNO_FILE = "/data/datasets/ROAD_plusplus/road_waymo_trainval_v1.0.json"
VID_DIR   = "/data/datasets/ROAD_plusplus/videos"
OUT_DIR   = "/data/repos/PedestrianIntent++/ROAD_plusplus/viz/frames"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Agent color map (BGR) – reused from create_viz.py ─────────────────────────
# agent_labels: Ped, Car, Cyc, Mobike, SmalVeh, MedVeh, LarVeh, Bus, EmVeh, TL
AGENT_COLORS_BGR = {
    0: ( 86, 180, 233),   # Ped      – sky blue
    1: (  0, 158, 115),   # Car      – green
    2: (230, 159,   0),   # Cyc      – orange
    3: (204, 121, 167),   # Mobike   – pink/mauve
    4: (  0, 114, 178),   # SmalVeh  – dark blue
    5: (213,  94,   0),   # MedVeh   – vermillion
    6: (240, 228,  66),   # LarVeh   – yellow
    7: (  0,   0, 255),   # Bus      – red (BGR)
    8: (255, 255,   0),   # EmVeh    – cyan
    9: (128, 128, 128),   # TL       – gray
}

# Chip colors (BGR) for ROAD++ annotation dimensions
CHIP_COLORS_ROAD = {
    "AV_action": (  0,   0, 180),   # red
    "agent":     ( 86, 180, 233),   # sky-blue (same as Ped agent color)
    "action":    (  0, 180,   0),   # green
    "location":  (  0, 215, 255),   # gold
}
CHIP_ORDER = ["AV_action", "agent", "action", "location"]

# Crossing-relevant action indices
WAIT2X_IDX           = 17
CROSSING_ACTION_IDS  = {17, 18, 19, 20}   # Wait2X, XingFmLft, XingFmRht, Xing
PED_LABEL_ID         = 0

# ── Drawing helpers ───────────────────────────────────────────────────────────
FONT     = cv2.FONT_HERSHEY_SIMPLEX
CHIP_H   = 22
CHIP_FS  = 0.42
CHIP_PAD = 5


def draw_chip_stack(img, x1, y1, chips):
    """Stacked annotation chips above the bounding-box top edge."""
    n = len(chips)
    if n == 0:
        return
    # Some chips may be multi-value; expand them first
    expanded = []
    for label, values, color in chips:
        if isinstance(values, list):
            expanded.append((label, " | ".join(values) if values else "--", color))
        else:
            expanded.append((label, values if values else "--", color))

    start_y = max(0, y1 - len(expanded) * CHIP_H)
    for i, (label, value, color) in enumerate(expanded):
        text = f"{label}: {value}"
        (tw, th), _ = cv2.getTextSize(text, FONT, CHIP_FS, 1)
        chip_w = tw + CHIP_PAD * 2
        cy  = start_y + i * CHIP_H
        cx2 = min(img.shape[1] - 1, x1 + chip_w)
        cy2 = min(img.shape[0] - 1, cy + CHIP_H - 2)
        cv2.rectangle(img, (x1, cy), (cx2, cy2), color, -1)
        text_y = cy + CHIP_H - 5
        cv2.putText(img, text, (x1 + CHIP_PAD, text_y),
                    FONT, CHIP_FS, (255, 255, 255), 1, cv2.LINE_AA)


def draw_banner(img, text, color):
    """Colored banner strip near the bottom of the image."""
    h, w = img.shape[:2]
    bh = 30
    y0 = h - bh - 4
    cv2.rectangle(img, (0, y0), (w, y0 + bh), color, -1)
    (tw, _), _ = cv2.getTextSize(text, FONT, 0.65, 2)
    tx = max(8, (w - tw) // 2)
    cv2.putText(img, text, (tx, y0 + bh - 7),
                FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def draw_title(img, text):
    """Dark title bar at the very top."""
    cv2.rectangle(img, (0, 0), (img.shape[1], 32), (25, 25, 25), -1)
    cv2.putText(img, text, (10, 23),
                FONT, 0.52, (210, 210, 210), 1, cv2.LINE_AA)


def draw_badge(img, text, color=(60, 60, 60)):
    """Small phase badge at top-right corner."""
    h, w = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.45, 1)
    bw, bh = tw + 12, th + 10
    x0 = w - bw - 6
    y0 = 36
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), color, -1)
    cv2.putText(img, text, (x0 + 6, y0 + bh - 5),
                FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def draw_all_boxes(img, frame_data, agent_labels, action_labels, selected_tube_uid=None):
    """Draw all bounding boxes in a frame; highlight the selected tube."""
    h, w = img.shape[:2]
    annos = frame_data.get("annos", {})
    for box_key, anno in annos.items():
        if not isinstance(anno, dict):
            continue
        box = anno.get("box", [])
        if len(box) != 4:
            continue
        xmin, ymin, xmax, ymax = box
        x1 = int(xmin * w)
        y1 = int(ymin * h)
        x2 = int(xmax * w)
        y2 = int(ymax * h)

        agent_ids = anno.get("agent_ids", [])
        aid = agent_ids[0] if agent_ids else 0
        color = AGENT_COLORS_BGR.get(aid, (200, 200, 200))

        is_selected = (anno.get("tube_uid") == selected_tube_uid)
        thickness   = 4 if is_selected else 1
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Simple label for non-selected boxes
        if not is_selected:
            agent_name = agent_labels[aid] if aid < len(agent_labels) else f"A{aid}"
            action_ids = anno.get("action_ids", [])
            act_str    = ""
            if action_ids:
                act_id = action_ids[0]
                act_str = action_labels[act_id] if act_id < len(action_labels) else ""
            label = f"{agent_name}" + (f"|{act_str}" if act_str else "")
            fs = 0.32
            (lw, lh), _ = cv2.getTextSize(label, FONT, fs, 1)
            text_y = max(y1 - 2, lh + 2)
            cv2.rectangle(img, (x1, text_y - lh - 2), (x1 + lw + 4, text_y + 2), color, -1)
            br = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
            tc = (0, 0, 0) if br > 128 else (255, 255, 255)
            cv2.putText(img, label, (x1 + 2, text_y), FONT, fs, tc, 1, cv2.LINE_AA)

    # AV action label is shown in title bar; nothing more to draw here


# ── Pedestrian tube selection ─────────────────────────────────────────────────
def find_ped_tubes(db, action_labels, agent_labels, n_max=12):
    """
    Find up to n_max pedestrian tubes with crossing actions across diverse videos.
    Prioritise tubes with Wait2X; fall back to other crossing actions.
    Returns list of dicts: {vid_name, tube_id, tube_uid, keyframe, actions, box_key}
    """
    results_wait2x = []
    results_other  = []
    videos_used    = set()

    for vid_name, vdata in db.items():
        if vid_name in videos_used:
            continue
        agent_tubes = vdata.get("agent_tubes", {})
        frames      = vdata.get("frames", {})

        for tube_id, tube_data in agent_tubes.items():
            if tube_data.get("label_id") != PED_LABEL_ID:
                continue
            tube_annos = tube_data.get("annos", {})
            tube_uid   = tube_id  # tube_id is the UUID

            # Walk frames in order looking for crossing actions
            first_wait2x_fid    = None
            first_crossing_fid  = None
            wait2x_acts         = []
            crossing_acts       = []
            first_wait2x_box    = None
            first_crossing_box  = None

            for fid_str in sorted(tube_annos.keys(), key=lambda x: int(x)):
                box_key    = tube_annos[fid_str]
                frame_data = frames.get(fid_str, {})
                if not isinstance(frame_data, dict):
                    continue
                anno = frame_data.get("annos", {}).get(box_key, {})
                if not isinstance(anno, dict):
                    continue
                action_ids = anno.get("action_ids", [])

                if first_wait2x_fid is None and WAIT2X_IDX in action_ids:
                    first_wait2x_fid = int(fid_str)
                    wait2x_acts      = action_ids
                    first_wait2x_box = box_key

                if first_crossing_fid is None and any(a in CROSSING_ACTION_IDS for a in action_ids):
                    first_crossing_fid = int(fid_str)
                    crossing_acts      = action_ids
                    first_crossing_box = box_key

                if first_wait2x_fid is not None and first_crossing_fid is not None:
                    break

            if first_wait2x_fid is not None:
                results_wait2x.append({
                    "vid_name": vid_name,
                    "tube_id":  tube_id,
                    "tube_uid": tube_uid,
                    "keyframe": first_wait2x_fid,
                    "action_ids": wait2x_acts,
                    "box_key":  first_wait2x_box,
                    "has_wait2x": True,
                })
                videos_used.add(vid_name)
                break   # one tube per video
            elif first_crossing_fid is not None:
                results_other.append({
                    "vid_name": vid_name,
                    "tube_id":  tube_id,
                    "tube_uid": tube_uid,
                    "keyframe": first_crossing_fid,
                    "action_ids": crossing_acts,
                    "box_key":  first_crossing_box,
                    "has_wait2x": False,
                })
                videos_used.add(vid_name)
                break

        if len(results_wait2x) + len(results_other) >= n_max * 2:
            break

    # Prefer Wait2X tubes; top up with other crossing tubes
    combined = results_wait2x[:n_max]
    if len(combined) < n_max:
        combined += results_other[: n_max - len(combined)]
    return combined[:n_max]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading ROAD++ annotations...", flush=True)
    t0 = time.time()
    with open(ANNO_FILE) as f:
        data = json.load(f)
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

    agent_labels     = data["agent_labels"]
    action_labels    = data["action_labels"]
    loc_labels       = data["loc_labels"]
    av_action_labels = data["av_action_labels"]
    db               = data["db"]

    print("Finding pedestrian tubes with crossing actions...", flush=True)
    tubes = find_ped_tubes(db, action_labels, agent_labels, n_max=12)
    print(f"Selected {len(tubes)} tubes.", flush=True)

    for idx, tube_info in enumerate(tubes, 1):
        vid_name  = tube_info["vid_name"]
        tube_id   = tube_info["tube_id"]
        keyframe  = tube_info["keyframe"]
        action_ids = tube_info["action_ids"]
        box_key   = tube_info["box_key"]
        has_wait2x = tube_info["has_wait2x"]

        vdata  = db[vid_name]
        frames = vdata.get("frames", {})

        fid_str    = str(keyframe)
        frame_data = frames.get(fid_str, {})
        anno       = frame_data.get("annos", {}).get(box_key, {}) if isinstance(frame_data, dict) else {}

        if not isinstance(anno, dict):
            print(f"  [{idx:02d}] WARNING: annotation not found for frame {keyframe}")
            continue

        # Geometry
        h_vid = frame_data.get("height", 1)
        w_vid = frame_data.get("width", 1)
        box   = anno.get("box", [])

        # Read real video frame
        vid_path = os.path.join(VID_DIR, f"{vid_name}.mp4")
        if not os.path.exists(vid_path):
            print(f"  [{idx:02d}] WARNING: video not found: {vid_path}")
            continue

        cap = cv2.VideoCapture(vid_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, keyframe - 1)   # create_viz.py pattern (1-indexed)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"  [{idx:02d}] WARNING: could not read frame {keyframe} from {vid_path}")
            continue

        h, w = frame.shape[:2]

        # Draw all boxes in frame (background context)
        # We pass tube_uid so the selected ped gets thicker outline
        draw_all_boxes(frame, frame_data, agent_labels, action_labels,
                       selected_tube_uid=tube_id)

        # Selected ped box
        if len(box) == 4:
            xmin, ymin, xmax, ymax = box
            x1 = max(0, int(xmin * w))
            y1 = max(0, int(ymin * h))
            x2 = min(w - 1, int(xmax * w))
            y2 = min(h - 1, int(ymax * h))
        else:
            # Fall back: just skip chip drawing
            x1 = y1 = x2 = y2 = 0

        # Build chip list for selected ped
        agent_ids_ped  = anno.get("agent_ids", [])
        action_ids_ped = anno.get("action_ids", [])
        loc_ids_ped    = anno.get("loc_ids", [])
        av_ids         = frame_data.get("av_action_ids", []) if isinstance(frame_data, dict) else []

        agent_names  = [agent_labels[a]  for a in agent_ids_ped  if a < len(agent_labels)]
        action_names = [action_labels[a] for a in action_ids_ped if a < len(action_labels)]
        loc_names    = [loc_labels[a]    for a in loc_ids_ped    if a < len(loc_labels)]
        av_names     = [av_action_labels[a] for a in av_ids      if a < len(av_action_labels)]

        chips = [
            ("AV_action", av_names,    CHIP_COLORS_ROAD["AV_action"]),
            ("agent",     agent_names, CHIP_COLORS_ROAD["agent"]),
            ("action",    action_names, CHIP_COLORS_ROAD["action"]),
            ("location",  loc_names,   CHIP_COLORS_ROAD["location"]),
        ]

        if x1 < x2 and y1 < y2:
            draw_chip_stack(frame, x1, y1, chips)

        # Banner
        if has_wait2x:
            banner_text  = "WAIT2X -> XING ONSET"
            banner_color = (128, 128,   0)   # teal BGR
        else:
            act_str = " | ".join(action_names) if action_names else "CROSSING"
            banner_text  = f"> {act_str}"
            banner_color = (  0, 160,   0)   # green

        draw_banner(frame, banner_text, banner_color)

        # Title
        short_tube = tube_id[:8] if len(tube_id) > 8 else tube_id
        av_str     = av_names[0] if av_names else "?"
        title = (f"{vid_name}  |  tube: {short_tube}...  "
                 f"|  Frame: {keyframe}  |  AV: {av_str}")
        draw_title(frame, title)

        # Phase badge – "cross onset" if Wait2X, "crossing" if XingFm*/Xing
        if has_wait2x:
            badge_text, badge_color = "cross onset", (0, 80, 160)
        else:
            badge_text, badge_color = "crossing",    (0, 130, 0)
        draw_badge(frame, badge_text, badge_color)

        # Save
        action_tag = "wait2x" if has_wait2x else "xing"
        out_fname  = f"frame_{idx:02d}_{vid_name}_{short_tube}_{action_tag}.png"
        out_path   = os.path.join(OUT_DIR, out_fname)
        cv2.imwrite(out_path, frame)
        act_str = " | ".join(action_names)
        print(f"  [{idx:02d}] {out_fname}  (f={keyframe}, actions={act_str})")

    print(f"\nDone. Output in: {OUT_DIR}")


if __name__ == "__main__":
    main()

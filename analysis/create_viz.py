#!/usr/bin/env python3
"""
ROAD++ Visualization Script
Creates:
1. Composite grid of annotated frames from 4 train videos (4 frames each = 4x4 grid)
2. Tube timeline swimlane diagram for 2 videos
"""
import json
import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import defaultdict
import time

ANNO_FILE = "/data/datasets/ROAD_plusplus/road_waymo_trainval_v1.0.json"
VID_DIR   = "/data/datasets/ROAD_plusplus/videos"
OUT_DIR   = "/data/repos/ROAD_plusplus/viz"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading annotations...", flush=True)
t0 = time.time()
with open(ANNO_FILE) as f:
    data = json.load(f)
print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

agent_labels     = data["agent_labels"]
action_labels    = data["action_labels"]
loc_labels       = data["loc_labels"]
duplex_labels    = data["duplex_labels"]
triplet_labels   = data["triplet_labels"]
av_action_labels = data["av_action_labels"]
db = data["db"]

# ──────────────────────────────────────────────
# Color map: one color per agent class
# ──────────────────────────────────────────────
# agent_labels: Ped, Car, Cyc, Mobike, SmalVeh, MedVeh, LarVeh, Bus, EmVeh, TL
AGENT_COLORS_BGR = {
    0:  (86,  180, 233),   # Ped        - sky blue
    1:  (0,   158,  115),  # Car        - green
    2:  (230, 159,   0),   # Cyc        - orange
    3:  (204, 121,  167),  # Mobike     - pink/mauve
    4:  (0,   114,  178),  # SmalVeh    - dark blue
    5:  (213,  94,   0),   # MedVeh     - vermillion
    6:  (240, 228,  66),   # LarVeh     - yellow
    7:  (0,    0,  255),   # Bus        - red (BGR)
    8:  (255, 255,   0),   # EmVeh      - cyan
    9:  (128, 128, 128),   # TL         - gray
}
AGENT_COLORS_MPL = {
    0: '#56B4E9',
    1: '#009E73',
    2: '#E69F00',
    3: '#CC79A7',
    4: '#0072B2',
    5: '#D55E00',
    6: '#F0E442',
    7: '#0000FF',
    8: '#FFFF00',
    9: '#808080',
}

def draw_anno_frame(frame_rgb, frame_data, agent_labels, action_labels, loc_labels, av_action_labels):
    """Draw bounding boxes with labels on an RGB frame."""
    img = frame_rgb.copy()
    h, w = img.shape[:2]

    annos = frame_data.get("annos", {})
    for anno_key, anno_val in annos.items():
        if not isinstance(anno_val, dict):
            continue
        box = anno_val.get("box", [])
        if len(box) != 4:
            continue
        xmin, ymin, xmax, ymax = box
        x1 = int(xmin * w)
        y1 = int(ymin * h)
        x2 = int(xmax * w)
        y2 = int(ymax * h)

        agent_ids  = anno_val.get("agent_ids", [])
        action_ids = anno_val.get("action_ids", [])
        loc_ids    = anno_val.get("loc_ids", [])

        # Agent class for color
        aid = agent_ids[0] if agent_ids else 0
        color_bgr = AGENT_COLORS_BGR.get(aid, (200, 200, 200))
        # Convert BGR for cv2
        color = (color_bgr[2], color_bgr[1], color_bgr[0])

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Build label text: agent + first action
        agent_name = agent_labels[aid] if aid < len(agent_labels) else f"A{aid}"
        action_name = ""
        if action_ids:
            act_id = action_ids[0]
            action_name = action_labels[act_id] if act_id < len(action_labels) else f"Act{act_id}"
        label_text = f"{agent_name}"
        if action_name:
            label_text += f"|{action_name}"

        # Draw text background + text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        text_y = max(y1 - 2, th + 2)
        cv2.rectangle(img, (x1, text_y - th - baseline), (x1 + tw, text_y + baseline), color, -1)
        # Text color: black for bright colors, white for dark
        brightness = 0.299*color[2] + 0.587*color[1] + 0.114*color[0]
        txt_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
        cv2.putText(img, label_text, (x1, text_y), font, font_scale, txt_color, thickness, cv2.LINE_AA)

    # AV action in corner
    av_ids = frame_data.get("av_action_ids", [])
    if av_ids:
        av_name = av_action_labels[av_ids[0]] if av_ids[0] < len(av_action_labels) else f"AV{av_ids[0]}"
        cv2.rectangle(img, (5, 5), (5 + len(av_name)*9 + 10, 28), (40, 40, 40), -1)
        cv2.putText(img, f"AV: {av_name}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def get_annotated_frames_indices(frames_dict, n=4):
    """Get n evenly-spaced annotated frame indices."""
    ann_frames = sorted([int(k) for k, v in frames_dict.items() if v.get("annotated") == 1])
    if not ann_frames:
        return []
    if len(ann_frames) <= n:
        return ann_frames
    step = len(ann_frames) // n
    indices = [ann_frames[i * step] for i in range(n)]
    return indices


# ──────────────────────────────────────────────
# 1. COMPOSITE GRID: 5 videos × 4 frames
# ──────────────────────────────────────────────
# Videos chosen for high annotation density and visual variety
VIZ_VIDEOS = [
    "train_00331",  # Very high density (82.5 boxes/frame, 147 agent tubes)
    "train_00117",  # High density (69.7 boxes/frame)
    "train_00240",  # High tube count (177 agent tubes)
    "train_00355",  # High tube count (156 agent tubes)
    "train_00064",  # Good density
]

print("\nCreating composite annotated frame grid...", flush=True)

n_videos = len(VIZ_VIDEOS)
n_frames = 4
thumb_w, thumb_h = 480, 320  # thumbnail size

# Create figure: n_videos rows x n_frames columns
fig_w = thumb_w * n_frames / 96
fig_h = (thumb_h * n_videos + 80) / 96  # extra for title/legend
fig, axes = plt.subplots(n_videos, n_frames, figsize=(thumb_w * n_frames / 96, thumb_h * n_videos / 96 + 1.5))
fig.patch.set_facecolor('#1a1a1a')

for vi, vname in enumerate(VIZ_VIDEOS):
    if vname not in db:
        print(f"  WARNING: {vname} not in db, skipping")
        continue

    vdata = db[vname]
    frames_dict = vdata.get("frames", {})
    av_action_labels_local = av_action_labels

    # Get frame indices to visualize
    frame_indices = get_annotated_frames_indices(frames_dict, n_frames)
    if not frame_indices:
        print(f"  No annotated frames found for {vname}")
        continue

    # Open video
    vid_path = os.path.join(VID_DIR, f"{vname}.mp4")
    if not os.path.exists(vid_path):
        print(f"  WARNING: {vid_path} not found")
        continue

    cap = cv2.VideoCapture(vid_path)
    total_boxes = sum(len(fd.get("annos", {})) for fd in frames_dict.values() if fd.get("annotated") == 1)
    n_tubes = len(vdata.get("agent_tubes", {}))

    print(f"  Processing {vname}: {len(frame_indices)} frames, {total_boxes} boxes, {n_tubes} agent tubes")

    for fi, frame_idx in enumerate(frame_indices):
        # Read frame from video (frame_idx is 1-indexed based on JSON)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)  # 0-indexed
        ret, frame_bgr = cap.read()
        if not ret:
            print(f"    WARNING: could not read frame {frame_idx}")
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Get frame annotation
        fdata = frames_dict.get(str(frame_idx), {})
        if not fdata:
            # Try without leading zeros
            fdata = {}

        # Draw annotations
        annotated_frame = draw_anno_frame(frame_rgb, fdata, agent_labels, action_labels, loc_labels, av_action_labels)

        # Resize to thumbnail
        thumb = cv2.resize(annotated_frame, (thumb_w, thumb_h))

        ax = axes[vi][fi] if n_videos > 1 else axes[fi]
        ax.imshow(thumb)
        ax.set_xticks([])
        ax.set_yticks([])

        n_boxes = len(fdata.get("annos", {}))
        av_ids = fdata.get("av_action_ids", [])
        av_name = av_action_labels[av_ids[0]] if av_ids else "?"
        ax.set_title(f"f{frame_idx} | {n_boxes} boxes | {av_name}",
                     fontsize=7, color='white', pad=2)

        # Row label on first column
        if fi == 0:
            ax.set_ylabel(f"{vname}\n({n_tubes} tubes)",
                         fontsize=7, color='#aaaaaa', rotation=90, labelpad=4)

    cap.release()

# Legend
legend_patches = []
for i, name in enumerate(agent_labels):
    color = AGENT_COLORS_MPL.get(i, '#888888')
    legend_patches.append(mpatches.Patch(color=color, label=name))

fig.legend(handles=legend_patches,
           loc='lower center',
           ncol=len(agent_labels),
           fontsize=8,
           framealpha=0.3,
           facecolor='#333333',
           edgecolor='gray',
           labelcolor='white',
           bbox_to_anchor=(0.5, 0.0))

fig.suptitle("ROAD++ (Road-Waymo): Annotated Frame Samples\nColored boxes = agent type. Label = agent|action. AV: = ego-vehicle action.",
             fontsize=10, color='white', y=1.0)
fig.tight_layout(rect=[0, 0.06, 1, 0.98])
plt.subplots_adjust(hspace=0.15, wspace=0.04)

out_path = os.path.join(OUT_DIR, "ROAD_real_annotated_frames.png")
fig.savefig(out_path, dpi=96, bbox_inches='tight', facecolor='#1a1a1a')
plt.close(fig)
print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────
# 2. TUBE TIMELINE SWIMLANE DIAGRAM
# ──────────────────────────────────────────────
print("\nCreating tube timeline diagrams...", flush=True)

TIMELINE_VIDEOS = ["train_00240", "train_00355"]

fig, axes = plt.subplots(len(TIMELINE_VIDEOS), 1,
                          figsize=(16, 6 * len(TIMELINE_VIDEOS)),
                          facecolor='#1a1a1a')
if len(TIMELINE_VIDEOS) == 1:
    axes = [axes]

for ax_idx, vname in enumerate(TIMELINE_VIDEOS):
    ax = axes[ax_idx]
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#555555')
    ax.spines['left'].set_color('#555555')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if vname not in db:
        continue
    vdata = db[vname]
    agent_tubes = vdata.get("agent_tubes", {})
    numf = vdata.get("numf", 0)
    n_tubes = len(agent_tubes)
    n_tubes_to_show = min(n_tubes, 60)

    print(f"  {vname}: {n_tubes} agent tubes total, showing {n_tubes_to_show}")

    # Sort tubes by label_id then by start frame
    tube_list = []
    for tube_id, tube_data in agent_tubes.items():
        label_id = tube_data.get("label_id", -1)
        tube_annos = tube_data.get("annos", {})
        frame_ids = sorted([int(k) for k in tube_annos.keys()])
        if frame_ids:
            tube_list.append((label_id, min(frame_ids), max(frame_ids), tube_id, len(frame_ids)))

    # Sort: first by label_id, then by start frame
    tube_list.sort(key=lambda x: (x[0], x[1]))
    tube_list = tube_list[:n_tubes_to_show]

    # Draw swimlanes
    bar_height = 0.7
    for yi, (label_id, start_f, end_f, tube_id, tube_len) in enumerate(tube_list):
        color = AGENT_COLORS_MPL.get(label_id, '#888888')
        agent_name = agent_labels[label_id] if 0 <= label_id < len(agent_labels) else f"A{label_id}"
        ax.barh(yi, end_f - start_f + 1, left=start_f, height=bar_height,
                color=color, alpha=0.85, edgecolor='none')
        # Label inside if wide enough
        if end_f - start_f > 15:
            brightness = sum(int(color[i:i+2], 16) for i in (1, 3, 5)) / 3
            txt_color = '#000000' if brightness > 128 else '#ffffff'
            ax.text(start_f + 1, yi, agent_name, va='center', ha='left',
                    fontsize=5.5, color=txt_color, clip_on=True)

    ax.set_xlim(0, numf + 1)
    ax.set_ylim(-0.5, n_tubes_to_show - 0.5)
    ax.set_xlabel("Frame number", color='white', fontsize=10)
    ax.set_ylabel(f"Agent tube index (top {n_tubes_to_show} of {n_tubes})", color='white', fontsize=9)
    ax.set_title(f"{vname} — Agent Tube Timeline ({numf} frames, {n_tubes} total agent tubes)",
                 color='white', fontsize=11, pad=8)
    ax.yaxis.set_visible(False)

    # Legend for this video
    seen_labels = set(t[0] for t in tube_list)
    leg_patches = [mpatches.Patch(color=AGENT_COLORS_MPL.get(lid, '#888'),
                                   label=agent_labels[lid] if 0 <= lid < len(agent_labels) else f"A{lid}")
                   for lid in sorted(seen_labels)]
    ax.legend(handles=leg_patches, loc='upper right', fontsize=8,
              framealpha=0.4, facecolor='#333', edgecolor='gray', labelcolor='white')

plt.suptitle("ROAD++ (Road-Waymo): Agent Tube Timelines\nEach row = one agent tube. Bar color = agent class. Bar span = frames where tube is active.",
             color='white', fontsize=12, y=1.01)
plt.tight_layout()

out_path = os.path.join(OUT_DIR, "ROAD_tube_timeline.png")
fig.savefig(out_path, dpi=96, bbox_inches='tight', facecolor='#1a1a1a')
plt.close(fig)
print(f"  Saved: {out_path}")

print("\nDone.")

#!/usr/bin/env python3
"""
SmolVLM zero-shot inference baseline on ROAD-Waymo (ROAD++).

Samples frames from validation videos, runs SmolVLM with a structured
prompt listing valid ROAD-Waymo label classes, and saves predictions
alongside ground truth annotations for offline evaluation.

Usage:
    python experiments/exp0_smolvlm_baselines/smolvlm_inference.py
    python experiments/exp0_smolvlm_baselines/smolvlm_inference.py --n_videos 20 --frames_per_video 10
    python experiments/exp0_smolvlm_baselines/smolvlm_inference.py --model HuggingFaceTB/SmolVLM-256M-Instruct
    python experiments/exp0_smolvlm_baselines/smolvlm_inference.py --split train --output results/train_preds.json
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# ── Paths ──────────────────────────────────────────────────────────────────────
ANNO_FILE   = "/data/datasets/ROAD_plusplus/road_waymo_trainval_v1.0.json"
FRAMES_DIR  = "/data/datasets/ROAD_plusplus/rgb-images"
DEFAULT_OUT = "/data/repos/ROAD_Reason/experiments/exp0_smolvlm_baselines/results/smolvlm_preds.json"

# ── Prompt ─────────────────────────────────────────────────────────────────────
# Label sets are injected at runtime from the annotation JSON so the prompt
# always matches the dataset exactly.
PROMPT_TEMPLATE = """\
You are analysing a frame from a dashcam video recorded by an autonomous vehicle.

Identify every visible road agent and describe it using ONLY the labels below.

Agent types: {agent_labels}
Actions:     {action_labels}
Locations:   {loc_labels}

For each agent, output one JSON object with these exact keys:
  "agent"    – one value from Agent types
  "action"   – one or more values from Actions (list)
  "location" – one value from Locations

Wrap all detections in a top-level JSON object:
{{
  "detections": [
    {{"agent": "...", "action": ["..."], "location": "..."}},
    ...
  ],
  "scene": "one sentence describing the overall scene"
}}

Respond with the JSON object only — no markdown fences, no extra text.\
"""


def build_prompt(agent_labels, action_labels, loc_labels):
    return PROMPT_TEMPLATE.format(
        agent_labels=", ".join(agent_labels),
        action_labels=", ".join(action_labels),
        loc_labels=", ".join(loc_labels),
    )


def load_annotations(anno_file):
    print(f"Loading annotations from {anno_file} …", flush=True)
    t0 = time.time()
    with open(anno_file) as f:
        data = json.load(f)
    print(f"  Loaded in {time.time() - t0:.1f}s", flush=True)
    return data


def get_split_videos(db, split):
    """Return list of video names that belong to the given split."""
    videos = []
    for vname, vdata in db.items():
        split_ids = vdata.get("split_ids", [])
        if isinstance(split_ids, str):
            split_ids = [split_ids]
        if split in split_ids:
            videos.append(vname)
    return videos


def get_annotated_frames(vdata):
    """Return list of (frame_id_str, frame_data) for annotated frames that have boxes."""
    result = []
    for fid, fdata in vdata.get("frames", {}).items():
        if fdata.get("annotated", 0) == 1 and fdata.get("annos"):
            result.append((fid, fdata))
    return result


def frame_id_to_path(frames_dir, video_name, frame_id):
    """Map annotation frame_id (e.g. "1") to JPEG path (e.g. rgb-images/train_00000/00001.jpg)."""
    fname = f"{int(frame_id):05d}.jpg"
    return Path(frames_dir) / video_name / fname


def parse_gt(frame_data, agent_labels, action_labels, loc_labels):
    """Return list of dicts {agent, actions, locations} from ground-truth annotation."""
    detections = []
    for anno in frame_data.get("annos", {}).values():
        if not isinstance(anno, dict):
            continue
        agents    = [agent_labels[i]  for i in anno.get("agent_ids",  []) if i < len(agent_labels)]
        actions   = [action_labels[i] for i in anno.get("action_ids", []) if i < len(action_labels)]
        locations = [loc_labels[i]    for i in anno.get("loc_ids",    []) if i < len(loc_labels)]
        if agents:
            detections.append({"agent": agents, "actions": actions, "locations": locations})
    return detections


def parse_model_output(raw_text):
    """Extract JSON from model output, handling common failure modes."""
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?\s*", "", raw_text).strip().rstrip("`").strip()
    # Find first { … } block
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None, "no_json_found"
    try:
        return json.loads(text[start:end]), None
    except json.JSONDecodeError as e:
        return None, str(e)


def run_inference(args):
    # ── Load data ───────────────────────────────────────────────────────────
    data = load_annotations(args.anno_file)
    db              = data["db"]
    agent_labels    = data["agent_labels"]
    action_labels   = data["action_labels"]
    loc_labels      = data["loc_labels"]

    prompt_text = build_prompt(agent_labels, action_labels, loc_labels)

    # ── Sample frames ────────────────────────────────────────────────────────
    split_videos = get_split_videos(db, args.split)
    if not split_videos:
        sys.exit(f"No videos found for split '{args.split}'")
    print(f"Found {len(split_videos)} videos in split '{args.split}'")

    rng = random.Random(args.seed)
    sampled_videos = rng.sample(split_videos, min(args.n_videos, len(split_videos)))

    samples = []  # list of (video_name, frame_id, img_path, gt)
    for vname in sampled_videos:
        vdata = db[vname]
        annotated = get_annotated_frames(vdata)
        if not annotated:
            continue
        chosen = rng.sample(annotated, min(args.frames_per_video, len(annotated)))
        for fid, fdata in chosen:
            img_path = frame_id_to_path(args.frames_dir, vname, fid)
            if not img_path.exists():
                print(f"  WARNING: frame not found: {img_path}", flush=True)
                continue
            gt = parse_gt(fdata, agent_labels, action_labels, loc_labels)
            samples.append((vname, fid, str(img_path), gt))

    print(f"Sampled {len(samples)} frames across {len(sampled_videos)} videos")

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"\nLoading model {args.model} …", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"  Device: {device}, dtype: {dtype}", flush=True)

    processor = AutoProcessor.from_pretrained(args.model)
    model     = AutoModelForVision2Seq.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()
    print("  Model loaded.", flush=True)

    # ── Inference loop ────────────────────────────────────────────────────────
    results = []
    for i, (vname, fid, img_path, gt) in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] {vname} frame {fid}", end=" … ", flush=True)

        image = Image.open(img_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

        t0 = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        elapsed = time.time() - t0

        # Decode only the newly generated tokens
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_text  = processor.decode(generated, skip_special_tokens=True)

        parsed, parse_err = parse_model_output(raw_text)
        status = "ok" if parsed else f"parse_error: {parse_err}"
        print(f"{status} ({elapsed:.1f}s)", flush=True)

        results.append({
            "video":     vname,
            "frame_id":  fid,
            "img_path":  img_path,
            "gt":        gt,
            "raw":       raw_text,
            "parsed":    parsed,
            "parse_err": parse_err,
            "elapsed_s": round(elapsed, 2),
        })

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "model":            args.model,
        "split":            args.split,
        "n_videos":         args.n_videos,
        "frames_per_video": args.frames_per_video,
        "seed":             args.seed,
        "agent_labels":     agent_labels,
        "action_labels":    action_labels,
        "loc_labels":       loc_labels,
        "total_frames":     len(results),
        "parse_ok":         sum(1 for r in results if r["parsed"] is not None),
    }
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    print(f"\nSaved {len(results)} results → {out_path}")
    print(f"Parse success: {meta['parse_ok']}/{meta['total_frames']}")


def main():
    parser = argparse.ArgumentParser(description="SmolVLM zero-shot baseline on ROAD-Waymo")
    parser.add_argument("--anno_file",        default=ANNO_FILE)
    parser.add_argument("--frames_dir",       default=FRAMES_DIR)
    parser.add_argument("--output",           default=DEFAULT_OUT)
    parser.add_argument("--model",            default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--split",            default="val")
    parser.add_argument("--n_videos",         type=int, default=10)
    parser.add_argument("--frames_per_video", type=int, default=5)
    parser.add_argument("--max_new_tokens",   type=int, default=512)
    parser.add_argument("--seed",             type=int, default=42)
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Qwen2.5-VL native grounding baseline on ROAD-Waymo.

This baseline uses Qwen2.5-VL in its native prompted grounding mode rather than
adding a custom detector head. The model is asked to return visible road-agent
detections directly as JSON with 2D bounding boxes and ROAD-Waymo labels.

Purpose
-------
- Test Qwen2.5-VL's built-in grounding ability directly.
- Separate "Qwen native grounding" from the custom FCOS-on-token design used in
  `experiments/exp1b_road_r`.
- Provide a bridge baseline before moving to OpenMixer.

Output schema
-------------
{
  "detections": [
    {
      "bbox_2d": [x1, y1, x2, y2],
      "agent": "Car",
      "action": ["Stop"],
      "location": "VehLane",
      "confidence": "high"
    }
  ],
  "scene": "one sentence"
}
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
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


ANNO_FILE = "/data/datasets/road_waymo/road_waymo_trainval_v1.1.json"
FRAMES_DIR = "/data/datasets/road_waymo/rgb-images"
DEFAULT_OUT = "/data/repos/ROAD_Reason/experiments/exp1c_qwen_native_grounding/results/qwen_native_grounding.json"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


PROMPT_TEMPLATE = """\
You are analyzing a dashcam frame from the ROAD-Waymo autonomous-driving dataset.

Detect every clearly visible road agent and return a JSON object only.

Use only these ROAD-Waymo labels:
- Agent types: {agent_labels}
- Actions: {action_labels}
- Locations: {loc_labels}

Return this exact format:
{{
  "detections": [
    {{
      "bbox_2d": [x1, y1, x2, y2],
      "agent": "<one agent label>",
      "action": ["<one or more action labels>"],
      "location": "<one location label>",
      "confidence": "high" | "medium" | "low"
    }}
  ],
  "scene": "one sentence describing the scene"
}}

Rules:
- `bbox_2d` must use absolute pixel coordinates in the current image.
- Use only labels from the allowed ROAD-Waymo lists above.
- Output one detection per visible agent instance.
- If an object is too uncertain to localize, omit it rather than guessing.
- Respond with JSON only. No markdown fences, no extra commentary.
"""


def build_prompt(agent_labels, action_labels, loc_labels):
    return PROMPT_TEMPLATE.format(
        agent_labels=", ".join(agent_labels),
        action_labels=", ".join(action_labels),
        loc_labels=", ".join(loc_labels),
    )


def load_annotations(anno_file):
    print(f"Loading annotations from {anno_file} ...", flush=True)
    with open(anno_file) as f:
        return json.load(f)


def get_split_videos(db, split):
    videos = []
    for vname, vdata in db.items():
        split_ids = vdata.get("split_ids", [])
        if isinstance(split_ids, str):
            split_ids = [split_ids]
        if split == "train":
            is_train = "train" in split_ids or ("all" in split_ids and "val" not in split_ids)
            if is_train:
                videos.append(vname)
        elif split in split_ids:
            videos.append(vname)
    return videos


def get_annotated_frames(vdata):
    result = []
    for fid, fdata in vdata.get("frames", {}).items():
        if fdata.get("annotated", 0) == 1 and fdata.get("annos"):
            result.append((fid, fdata))
    return result


def frame_id_to_path(frames_dir, video_name, frame_id):
    return Path(frames_dir) / video_name / f"{int(frame_id):05d}.jpg"


def parse_gt(frame_data, agent_labels, action_labels, loc_labels):
    detections = []
    for anno in frame_data.get("annos", {}).values():
        if not isinstance(anno, dict):
            continue
        box = anno.get("box")
        if box is None or len(box) != 4:
            continue
        detections.append(
            {
                "bbox_2d_norm": box,
                "agent": [agent_labels[i] for i in anno.get("agent_ids", []) if i < len(agent_labels)],
                "actions": [action_labels[i] for i in anno.get("action_ids", []) if i < len(action_labels)],
                "locations": [loc_labels[i] for i in anno.get("loc_ids", []) if i < len(loc_labels)],
            }
        )
    return detections


def parse_model_output(raw_text):
    text = re.sub(r"```(?:json)?\s*", "", raw_text).strip().rstrip("`").strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None, "no_json_found"
    try:
        return json.loads(text[start:end]), None
    except json.JSONDecodeError as e:
        return None, str(e)


def run_inference(args):
    data = load_annotations(args.anno_file)
    db = data["db"]
    agent_labels = data["agent_labels"]
    action_labels = data["action_labels"]
    loc_labels = data["loc_labels"]

    prompt_text = build_prompt(agent_labels, action_labels, loc_labels)

    split_videos = get_split_videos(db, args.split)
    if not split_videos:
        sys.exit(f"No videos found for split '{args.split}'")
    print(f"Found {len(split_videos)} videos in split '{args.split}'", flush=True)

    rng = random.Random(args.seed)
    sampled_videos = rng.sample(split_videos, min(args.n_videos, len(split_videos)))

    samples = []
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

    print(f"Sampled {len(samples)} frames across {len(sampled_videos)} videos", flush=True)

    print(f"\nLoading model {args.model} ...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"  Device: {device}, dtype: {dtype}", flush=True)

    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()
    print("  Model loaded.", flush=True)

    results = []
    for i, (vname, fid, img_path, gt) in enumerate(samples):
        print(f"[{i + 1}/{len(samples)}] {vname} frame {fid} ...", end=" ", flush=True)
        image = Image.open(img_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

        t0 = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        elapsed = time.time() - t0

        generated = generated_ids[0][inputs["input_ids"].shape[1]:]
        raw_text = processor.decode(generated, skip_special_tokens=True)

        parsed, parse_err = parse_model_output(raw_text)
        status = "ok" if parsed else f"parse_error: {parse_err}"
        print(f"{status} ({elapsed:.1f}s)", flush=True)

        results.append(
            {
                "video": vname,
                "frame_id": fid,
                "img_path": img_path,
                "gt": gt,
                "raw": raw_text,
                "parsed": parsed,
                "parse_err": parse_err,
                "elapsed_s": round(elapsed, 2),
            }
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "model": args.model,
            "split": args.split,
            "n_videos": args.n_videos,
            "frames_per_video": args.frames_per_video,
            "seed": args.seed,
            "total_frames": len(results),
            "parse_ok": sum(r["parsed"] is not None for r in results),
            "prompt_style": "qwen_native_grounding",
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved predictions to {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_file", default=ANNO_FILE)
    parser.add_argument("--frames_dir", default=FRAMES_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--split", default="val")
    parser.add_argument("--n_videos", type=int, default=10)
    parser.add_argument("--frames_per_video", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=DEFAULT_OUT)
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()

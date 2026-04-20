#!/usr/bin/env python3
"""
SmolVLM GT-conditioned reasoning baseline on ROAD-Waymo (ROAD++).

Feeds ground-truth structured labels (triplets) alongside the image and asks
the model to produce natural language reasoning about the scene. This isolates
the reasoning capability from detection — the model is told what is present and
must explain why and what it implies about agent intent.

This is the most direct proxy for the Approach 3 thesis contribution:
constrained natural language scene reasoning conditioned on structured labels.

Output format per frame:
  {
    "triplets_given":  ["Ped-Wait2X-Jun", "Car-MovAway-VehLane", "TL-Red-Jun"],
    "reasoning":       "A pedestrian is waiting at the junction ...",
    "intent_summary":  "Pedestrian likely to cross once signal changes."
  }

Usage:
    python experiments/exp0_smolvlm_baselines/smolvlm_gt_reasoning.py
    python experiments/exp0_smolvlm_baselines/smolvlm_gt_reasoning.py --n_videos 20 --frames_per_video 10
    python experiments/exp0_smolvlm_baselines/smolvlm_gt_reasoning.py --model HuggingFaceTB/SmolVLM-Instruct
    python experiments/exp0_smolvlm_baselines/smolvlm_gt_reasoning.py --output results/gt_reasoning_preds.json
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
DEFAULT_OUT = "/data/repos/ROAD_Reason/experiments/exp0_smolvlm_baselines/results/gt_reasoning_preds.json"

# ── Label glossary injected once into the system prompt ────────────────────────
# Gives the model enough domain knowledge to reason meaningfully about the
# ROAD-Waymo shorthand labels without needing to infer their meaning.
LABEL_GLOSSARY = """
Agent types:
  Ped=Pedestrian, Car=Car, Cyc=Cyclist, Mobike=Motorbike, SmalVeh=Small vehicle,
  MedVeh=Medium vehicle, LarVeh=Large vehicle, Bus=Bus, EmVeh=Emergency vehicle,
  TL=Traffic light

Actions:
  Red/Amber/Green = traffic light state
  MovAway=Moving away from camera, MovTow=Moving toward camera, Mov=Moving (lateral),
  Rev=Reversing, Brake=Braking, Stop=Stationary, IncatLft/IncatRht=Indicating left/right,
  HazLit=Hazard lights, TurLft/TurRht=Turning left/right, MovRht/MovLft=Moving right/left,
  Ovtak=Overtaking, Wait2X=Waiting to cross, XingFmLft=Crossing from left,
  XingFmRht=Crossing from right, Xing=Crossing, PushObj=Pushing an object

Locations:
  VehLane=Vehicle lane, OutgoLane=Outgoing lane, IncomLane=Incoming lane,
  OutgoCycLane/IncomCycLane=Cycle lane, OutgoBusLane/IncomBusLane=Bus lane,
  Pav=Pavement, LftPav=Left pavement, RhtPav=Right pavement,
  Jun=Junction, xing=Pedestrian crossing, BusStop=Bus stop, parking=Parking area
"""

SYSTEM_PROMPT = (
    "You are an expert autonomous-driving scene analyst. "
    "You are given a dashcam image along with the verified ground-truth labels "
    "for every road agent visible in the scene. "
    "Your task is to reason about the scene — explaining what is happening, "
    "why, and what each agent is likely to do next — using the labels as your "
    "factual anchor. Do not contradict or ignore the provided labels.\n\n"
    "Label glossary:\n" + LABEL_GLOSSARY
)

USER_TEMPLATE = """\
The following agent labels have been verified for this frame:

{triplet_lines}

Using these labels and the image, provide:
1. A scene description grounded in the labels (2–3 sentences).
2. An intent summary for the most safety-relevant agent (1 sentence).

Respond with a JSON object only — no markdown fences:
{{
  "scene_description": "...",
  "intent_summary": "..."
}}\
"""


def build_user_prompt(triplets):
    lines = "\n".join(f"  - {t}" for t in triplets)
    return USER_TEMPLATE.format(triplet_lines=lines)


def load_annotations(anno_file):
    print(f"Loading annotations from {anno_file} …", flush=True)
    t0 = time.time()
    with open(anno_file) as f:
        data = json.load(f)
    print(f"  Loaded in {time.time() - t0:.1f}s", flush=True)
    return data


def get_split_videos(db, split):
    videos = []
    for vname, vdata in db.items():
        split_ids = vdata.get("split_ids", [])
        if isinstance(split_ids, str):
            split_ids = [split_ids]
        if split in split_ids:
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


def gt_triplets(frame_data, agent_labels, action_labels, loc_labels):
    """Build all agent-action-location triplet strings present in a frame."""
    triplets = []
    for anno in frame_data.get("annos", {}).values():
        if not isinstance(anno, dict):
            continue
        agents    = [agent_labels[i]  for i in anno.get("agent_ids",  []) if i < len(agent_labels)]
        actions   = [action_labels[i] for i in anno.get("action_ids", []) if i < len(action_labels)]
        locations = [loc_labels[i]    for i in anno.get("loc_ids",    []) if i < len(loc_labels)]
        for ag in agents:
            for ac in actions:
                for lo in locations:
                    t = f"{ag}-{ac}-{lo}"
                    if t not in triplets:
                        triplets.append(t)
    return triplets


def parse_model_output(raw_text):
    text = re.sub(r"```(?:json)?\s*", "", raw_text).strip().rstrip("`").strip()
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
    db             = data["db"]
    agent_labels   = data["agent_labels"]
    action_labels  = data["action_labels"]
    loc_labels     = data["loc_labels"]

    # ── Sample frames ────────────────────────────────────────────────────────
    split_videos = get_split_videos(db, args.split)
    if not split_videos:
        sys.exit(f"No videos found for split '{args.split}'")
    print(f"Found {len(split_videos)} videos in split '{args.split}'")

    rng = random.Random(args.seed)
    sampled_videos = rng.sample(split_videos, min(args.n_videos, len(split_videos)))

    samples = []
    for vname in sampled_videos:
        vdata = db[vname]
        annotated = get_annotated_frames(vdata)
        if not annotated:
            continue
        # Prefer frames with pedestrian labels for more interesting reasoning
        ped_idx = data["agent_labels"].index("Ped") if "Ped" in data["agent_labels"] else -1
        if args.prefer_ped and ped_idx >= 0:
            ped_frames = [
                (fid, fd) for fid, fd in annotated
                if any(
                    ped_idx in anno.get("agent_ids", [])
                    for anno in fd.get("annos", {}).values()
                    if isinstance(anno, dict)
                )
            ]
            pool = ped_frames if ped_frames else annotated
        else:
            pool = annotated

        chosen = rng.sample(pool, min(args.frames_per_video, len(pool)))
        for fid, fdata in chosen:
            img_path = frame_id_to_path(args.frames_dir, vname, fid)
            if not img_path.exists():
                print(f"  WARNING: frame not found: {img_path}", flush=True)
                continue
            triplets = gt_triplets(fdata, agent_labels, action_labels, loc_labels)
            if not triplets:
                continue
            samples.append((vname, fid, str(img_path), triplets))

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
    for i, (vname, fid, img_path, triplets) in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] {vname} frame {fid}  ({len(triplets)} triplets)", end=" … ", flush=True)

        image       = Image.open(img_path).convert("RGB")
        user_prompt = build_user_prompt(triplets)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        try:
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        except Exception:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": SYSTEM_PROMPT + "\n\n" + user_prompt},
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

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_text  = processor.decode(generated, skip_special_tokens=True)
        parsed, parse_err = parse_model_output(raw_text)

        status = "ok" if parsed else "parse_error"
        print(f"{status} ({elapsed:.1f}s)", flush=True)
        if parsed:
            # Print a preview so you can eyeball quality as it runs
            scene = parsed.get("scene_description", "")
            print(f"    > {scene[:120]}", flush=True)

        results.append({
            "video":           vname,
            "frame_id":        fid,
            "img_path":        img_path,
            "triplets_given":  triplets,
            "raw":             raw_text,
            "parsed":          parsed,
            "parse_err":       parse_err,
            "elapsed_s":       round(elapsed, 2),
        })

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "model":            args.model,
        "split":            args.split,
        "n_videos":         args.n_videos,
        "frames_per_video": args.frames_per_video,
        "prefer_ped":       args.prefer_ped,
        "seed":             args.seed,
        "total_frames":     len(results),
        "parse_ok":         sum(1 for r in results if r["parsed"] is not None),
    }
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)

    print(f"\nSaved {len(results)} results → {out_path}")
    print(f"Parse success: {meta['parse_ok']}/{meta['total_frames']}")


def main():
    parser = argparse.ArgumentParser(
        description="SmolVLM GT-conditioned reasoning baseline on ROAD-Waymo"
    )
    parser.add_argument("--anno_file",        default=ANNO_FILE)
    parser.add_argument("--frames_dir",       default=FRAMES_DIR)
    parser.add_argument("--output",           default=DEFAULT_OUT)
    parser.add_argument("--model",            default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--split",            default="val")
    parser.add_argument("--n_videos",         type=int, default=10)
    parser.add_argument("--frames_per_video", type=int, default=5)
    parser.add_argument("--max_new_tokens",   type=int, default=256)
    parser.add_argument("--seed",             type=int, default=42)
    parser.add_argument("--prefer_ped",       action="store_true", default=True,
                        help="Prefer frames containing pedestrians (more interesting reasoning)")
    parser.add_argument("--no_prefer_ped",    dest="prefer_ped", action="store_false")
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()

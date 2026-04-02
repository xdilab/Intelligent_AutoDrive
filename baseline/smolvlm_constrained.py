#!/usr/bin/env python3
"""
SmolVLM constrained inference baseline on ROAD-Waymo (ROAD++).

Bakes all 135 valid ROAD-Waymo constraint labels (49 duplexes + 86 triplets)
directly into the system prompt so the model can only produce combinations that
are semantically legal according to the dataset ontology.

Contrast with smolvlm_inference.py (zero-shot, flat label lists).

Usage:
    python baseline/smolvlm_constrained.py
    python baseline/smolvlm_constrained.py --n_videos 20 --frames_per_video 10
    python baseline/smolvlm_constrained.py --model HuggingFaceTB/SmolVLM-Instruct
    python baseline/smolvlm_constrained.py --output results/constrained_preds.json
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
DEFAULT_OUT = "/data/repos/ROAD_Reason/baseline/results/constrained_preds.json"

# ── Prompt ─────────────────────────────────────────────────────────────────────
# SYSTEM_TEMPLATE: presents the constraint ontology once at the top.
# USER_TEMPLATE: the per-frame instruction with the valid-triplet list injected.
#
# Design rationale:
#   - Valid duplexes tell the model which agent+action pairs are possible at all.
#   - Valid triplets are the hard constraint: every detection must be drawn from
#     this list, no other combination is legal.
#   - Listing the triplets as a numbered menu gives the model a concrete reference
#     to quote from, reducing hallucination of impossible combos.

SYSTEM_PROMPT = (
    "You are an expert autonomous-driving perception system. "
    "You label road agents in dashcam frames using a fixed ontology. "
    "Every detection you output MUST be drawn exactly from the valid triplet list "
    "provided in each query — no other agent/action/location combinations exist in "
    "this ontology."
)

USER_TEMPLATE = """\
Analyse this dashcam frame and identify every visible road agent.

## Valid agent–action pairs (duplexes)
Only these agent+action combinations are legal:
{duplex_block}

## Valid agent–action–location triplets
Your detections MUST come from this list only — do not invent combinations:
{triplet_block}

## Output format
Return a single JSON object with no markdown fences:
{{
  "detections": [
    {{
      "triplet": "<Agent>-<Action>-<Location>",
      "confidence": "high" | "medium" | "low"
    }}
  ],
  "av_context": "one sentence about what the ego-vehicle is doing",
  "scene": "one sentence describing the overall scene"
}}

Rules:
- Each "triplet" value must be copied verbatim from the valid triplets list above.
- List one entry per visible agent instance (multiple of the same triplet is fine).
- If you are uncertain, use confidence "low" rather than guessing a different triplet.\
"""


def build_prompt(duplex_labels, triplet_labels):
    duplex_block  = "\n".join(f"  {d}" for d in duplex_labels)
    triplet_block = "\n".join(f"  {t}" for t in triplet_labels)
    return USER_TEMPLATE.format(duplex_block=duplex_block, triplet_block=triplet_block)


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


def parse_gt(frame_data, agent_labels, action_labels, loc_labels):
    detections = []
    for anno in frame_data.get("annos", {}).values():
        if not isinstance(anno, dict):
            continue
        agents    = [agent_labels[i]  for i in anno.get("agent_ids",  []) if i < len(agent_labels)]
        actions   = [action_labels[i] for i in anno.get("action_ids", []) if i < len(action_labels)]
        locations = [loc_labels[i]    for i in anno.get("loc_ids",    []) if i < len(loc_labels)]
        triplets  = []
        for ag in agents:
            for ac in actions:
                for lo in locations:
                    triplets.append(f"{ag}-{ac}-{lo}")
        detections.append({"agent": agents, "actions": actions, "locations": locations, "triplets": triplets})
    return detections


def parse_model_output(raw_text, valid_triplet_set):
    text = re.sub(r"```(?:json)?\s*", "", raw_text).strip().rstrip("`").strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None, "no_json_found"
    try:
        parsed = json.loads(text[start:end])
    except json.JSONDecodeError as e:
        return None, str(e)

    # Annotate each detection with whether its triplet is in the valid set
    for det in parsed.get("detections", []):
        t = det.get("triplet", "")
        det["_valid"] = t in valid_triplet_set

    return parsed, None


def constraint_violation_rate(parsed, valid_triplet_set):
    """Fraction of predicted triplets that are not in the valid set."""
    if not parsed:
        return None
    dets = parsed.get("detections", [])
    if not dets:
        return 0.0
    violations = sum(1 for d in dets if d.get("triplet", "") not in valid_triplet_set)
    return violations / len(dets)


def run_inference(args):
    # ── Load data ───────────────────────────────────────────────────────────
    data = load_annotations(args.anno_file)
    db             = data["db"]
    agent_labels   = data["agent_labels"]
    action_labels  = data["action_labels"]
    loc_labels     = data["loc_labels"]
    duplex_labels  = data["duplex_labels"]
    triplet_labels = data["triplet_labels"]
    valid_triplets = set(triplet_labels)

    print(f"Constraint set: {len(duplex_labels)} valid duplexes, {len(triplet_labels)} valid triplets")

    user_prompt = build_prompt(duplex_labels, triplet_labels)

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
    total_violation_rate = 0.0
    n_with_dets = 0

    for i, (vname, fid, img_path, gt) in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] {vname} frame {fid}", end=" … ", flush=True)

        image = Image.open(img_path).convert("RGB")

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

        # SmolVLM may not support system role — fall back to prepending to user content
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

        parsed, parse_err = parse_model_output(raw_text, valid_triplets)
        viol_rate = constraint_violation_rate(parsed, valid_triplets)

        if viol_rate is not None:
            total_violation_rate += viol_rate
            n_with_dets += 1

        status = "ok" if parsed else f"parse_error"
        viol_str = f"  violations={viol_rate:.0%}" if viol_rate is not None else ""
        print(f"{status}{viol_str} ({elapsed:.1f}s)", flush=True)

        results.append({
            "video":          vname,
            "frame_id":       fid,
            "img_path":       img_path,
            "gt":             gt,
            "raw":            raw_text,
            "parsed":         parsed,
            "parse_err":      parse_err,
            "violation_rate": viol_rate,
            "elapsed_s":      round(elapsed, 2),
        })

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mean_viol = total_violation_rate / n_with_dets if n_with_dets else None
    meta = {
        "model":                   args.model,
        "split":                   args.split,
        "n_videos":                args.n_videos,
        "frames_per_video":        args.frames_per_video,
        "seed":                    args.seed,
        "n_duplex_constraints":    len(duplex_labels),
        "n_triplet_constraints":   len(triplet_labels),
        "agent_labels":            agent_labels,
        "action_labels":           action_labels,
        "loc_labels":              loc_labels,
        "duplex_labels":           duplex_labels,
        "triplet_labels":          triplet_labels,
        "total_frames":            len(results),
        "parse_ok":                sum(1 for r in results if r["parsed"] is not None),
        "mean_constraint_violation_rate": mean_viol,
    }
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)

    print(f"\nSaved {len(results)} results → {out_path}")
    print(f"Parse success:           {meta['parse_ok']}/{meta['total_frames']}")
    if mean_viol is not None:
        print(f"Mean constraint violations: {mean_viol:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="SmolVLM constraint-aware inference baseline on ROAD-Waymo"
    )
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

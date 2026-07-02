"""Exp5 stage 2/3 — vanilla Qwen2.5-VL reasoning over RetinaNet boxes.

For each cached frame: load the native RGB image, draw the top-K RetinaNet
*predicted* boxes (pseudo-labels) as numbered boxes, prompt frozen
Qwen2.5-VL-7B-Instruct to classify each box, parse the JSON, and cache it.

One JSON file per frame (cache/qwen/<video>/<fid>.json) → fully resumable; a
crash loses only the in-flight frame. Per Moradi: cache VLM outputs once.

--prompt steered (detection-steered, update email 2026-07-02): the detector's
per-box class predictions (top-1 agent/action/location + confidence, read off
the 184-dim logits) are added to the prompt as priors and Qwen verifies/
refines instead of classifying from scratch. Requires a detections pkl that
carries logits (exp6_detection_steered/dump_detections.py). Cached separately
under cache/qwen_steered/ so the plain zero-shot cache stays the control.

Usage:
  CUDA_VISIBLE_DEVICES=0 python -u qwen_infer.py [--split val] [--limit N]
  CUDA_VISIBLE_DEVICES=0 python -u qwen_infer.py --prompt steered \
      --detections ../exp6_detection_steered/cache/detections_val.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import torch
from PIL import Image

import config as C
import vlm_io


def _frame_path(key: str) -> Path:
    vname, fid = key.rsplit("/", 1)
    return Path(C.FRAMES_DIR) / vname / f"{int(fid):05d}.jpg"


def _cache_path(root: Path, key: str) -> Path:
    vname, fid = key.rsplit("/", 1)
    return root / vname / f"{int(fid):05d}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="First N frames (debug). 0 = all.")
    ap.add_argument("--detections", default=None)
    ap.add_argument("--overwrite", action="store_true", help="Re-run frames already cached.")
    ap.add_argument("--num-shards", type=int, default=1, help="Split frames into N disjoint shards.")
    ap.add_argument("--shard", type=int, default=0, help="Which shard (0..num_shards-1) this job runs.")
    ap.add_argument("--prompt", choices=("plain", "steered"), default="plain",
                    help="steered = detector class predictions in the prompt as priors.")
    args = ap.parse_args()

    cache_root = C.QWEN_STEERED_CACHE_DIR if args.prompt == "steered" else C.QWEN_CACHE_DIR
    det_path = Path(args.detections) if args.detections else C.CACHE_DIR / f"detections_{args.split}.pkl"
    print(f"[qwen] prompt mode: {args.prompt}  cache: {cache_root}", flush=True)
    print(f"[qwen] loading detections: {det_path}", flush=True)
    with open(det_path, "rb") as f:
        payload = pickle.load(f)
    records = payload["records"]
    labels = payload["labels"]
    if args.prompt == "steered":
        assert payload["meta"].get("has_logits"), (
            f"{det_path} has no per-box logits — steered prompting needs a dump "
            "from exp6_detection_steered/dump_detections.py"
        )
    agent_set = set(labels["agent"])
    action_set = set(labels["action"])
    loc_set = set(labels["loc"])
    keys = sorted(records.keys())
    print(f"[qwen] {len(keys):,} frames in detection cache", flush=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[qwen] loading {C.QWEN_MODEL} on {device} ...", flush=True)
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        C.QWEN_MODEL, torch_dtype=torch.bfloat16, device_map={"": device},
    ).eval()
    processor = AutoProcessor.from_pretrained(C.QWEN_MODEL)
    print("[qwen] model ready", flush=True)

    n_done = n_run = 0
    t0 = time.time()
    for ki, key in enumerate(keys):
        if args.limit and n_run >= args.limit:
            break
        if args.num_shards > 1 and ki % args.num_shards != args.shard:
            continue                                    # another shard owns this frame
        cpath = _cache_path(cache_root, key)
        if cpath.exists() and not args.overwrite:
            n_done += 1
            continue

        rec = records[key]
        boxes_600 = rec["boxes"][: C.MAX_BOXES_PER_FRAME]            # [n,4] in 600x840
        n_boxes = boxes_600.shape[0]
        fpath = _frame_path(key)
        if not fpath.exists() or n_boxes == 0:
            cpath.parent.mkdir(parents=True, exist_ok=True)
            cpath.write_text(json.dumps({"key": key, "n_boxes": int(n_boxes),
                                         "parsed": [], "raw": "", "skipped": True}))
            n_run += 1
            continue

        img = Image.open(fpath).convert("RGB")
        nw, nh = img.size
        sx, sy = nw / C.VAL_MAX_SIZE, nh / C.VAL_SHORT_SIDE         # 600x840 → native
        boxes_native = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in boxes_600]
        drawn = vlm_io.draw_numbered_boxes(img, boxes_native)
        priors = None
        if args.prompt == "steered":
            priors = [vlm_io.format_priors(rec["logits"][i], labels["agent"],
                                           labels["action"], labels["loc"])
                      for i in range(n_boxes)]
        prompt = vlm_io.build_prompt(n_boxes, labels["agent"], labels["action"],
                                     labels["loc"], priors=priors)

        messages = [{"role": "user", "content": [
            {"type": "image", "image": drawn},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=C.QWEN_MAX_NEW_TOKENS, do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        raw = processor.batch_decode(trimmed, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)[0]
        parsed = vlm_io.parse_qwen_response(raw, n_boxes, agent_set, action_set, loc_set)

        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps({"key": key, "n_boxes": int(n_boxes),
                                     "parsed": parsed, "raw": raw}))
        n_run += 1
        if n_run % 20 == 0:
            el = time.time() - t0
            rate = el / n_run
            print(f"[qwen] ran {n_run} (skipped {n_done} cached) | {rate:.1f}s/frame | "
                  f"frame {ki+1}/{len(keys)}", flush=True)

    print(f"[qwen] done. ran={n_run}, already-cached={n_done}, "
          f"total-elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

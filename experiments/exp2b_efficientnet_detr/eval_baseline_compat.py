#!/usr/bin/env python3
"""
Baseline-compatible evaluation for Exp2b.

Two modes:
    --mode frame  : frame-level mAP (f-mAP) at IoU=0.5 using the official
                    ROAD++ evaluate_frames() function. Comparable to 3D-RetinaNet's
                    published numbers (e.g., agent f-mAP=17.0%).
    --mode video  : approximate tube-level evaluation using mean best-tube IoU.
                    A true video-mAP requires the official tube evaluation protocol;
                    this is a proxy for early iteration.

Why a separate eval file:
    Training's validate() uses "matched mAP" — AP only on the queries that
    successfully matched to GT tubes. This is a useful training signal but not
    comparable to the baseline. The baseline measures every detection against
    every GT box, penalising missed detections and false positives equally.
    This file implements that full evaluation protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]
EXP1_DIR = REPO_ROOT / "experiments" / "exp1_road_r"
# The baseline evaluation code lives in the PedestrianIntent++ research repo
BASELINE_ROOT = Path("/data/repos/PedestrianIntent++/ROAD_plus_plus_Baseline")

if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXP1_DIR) not in sys.path:
    sys.path.append(str(EXP1_DIR))

import config as C
from losses import greedy_group_tubes
from matcher import box_iou
from model import EfficientNetFPNDETRModel


def _load_module(name: str, path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def preprocess_clip(pil_frames, processor, device, dtype):
    inputs = processor.image_processor(
        images=pil_frames,
        return_tensors="pt",
        min_pixels=C.MIN_PIXELS,
        max_pixels=C.MAX_PIXELS,
    )
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    image_grid_thw = inputs["image_grid_thw"].to(device=device)
    return pixel_values, image_grid_thw


# Import baseline evaluation code AFTER adding its root to sys.path
sys.path.insert(0, str(BASELINE_ROOT))
from modules.evaluation import evaluate_frames  # noqa: E402
import data.datasets as baseline_datasets       # noqa: E402


def is_in_subset(split_ids, subset: str) -> bool:
    """Check whether a video belongs to the requested split (train/val/test)."""
    if isinstance(split_ids, str):
        split_ids = [split_ids]
    return subset in split_ids


def to_xyxy(boxes_cxcywh: torch.Tensor) -> torch.Tensor:
    """
    Convert DETR's [cx,cy,w,h] → [x1,y1,x2,y2] for the baseline evaluator.
    The baseline expects corner-format boxes.
    """
    cx, cy, w, h = boxes_cxcywh.unbind(-1)
    x1 = (cx - 0.5 * w).clamp(0.0, 1.0)
    y1 = (cy - 0.5 * h).clamp(0.0, 1.0)
    x2 = (cx + 0.5 * w).clamp(0.0, 1.0)
    y2 = (cy + 0.5 * h).clamp(0.0, 1.0)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def frame_to_detection_lists(
    boxes: torch.Tensor,
    probs: dict,
    width: int,
    height: int,
    class_threshold: float,
):
    """
    Convert DETR's per-query predictions into the format expected by evaluate_frames().

    evaluate_frames() expects a dict mapping head name → list of C arrays, where each
    array is [N_dets, 5] (x1, y1, x2, y2, score) in pixel coordinates.

    Key design decisions:

    Score = raw per-class sigmoid:
        The baseline (val.py) scores each class independently using the raw
        sigmoid output — confidence[b, s, :, class_idx]. We match this: each
        detection's score is its per-class sigmoid probability, with no
        agentness multiplication. Agentness is only used for the agent_ness
        label type itself.

    class_threshold filtering:
        Detections below class_threshold are discarded. This reduces false positives
        at the cost of recall. The baseline evaluator uses IoU=0.5 and sweeps over
        confidence thresholds internally for AP computation, so this threshold just
        limits how many low-confidence detections we pass in.

    Pixel scaling:
        Our box coordinates are in [0,1] normalised. The baseline evaluator needs
        pixel coordinates — multiply x by width and y by height.
    """

    def empty(n: int):
        return [np.zeros((0, 5), dtype=np.float32) for _ in range(n)]

    if boxes.shape[0] == 0:
        return {
            "agent_ness": [np.zeros((0, 5), dtype=np.float32)],
            "agent": empty(C.N_AGENTS),
            "action": empty(C.N_ACTIONS),
            "loc": empty(C.N_LOCS),
            "duplex": empty(C.N_DUPLEXES),
            "triplet": empty(C.N_TRIPLETS),
        }

    # Scale normalised [0,1] boxes to pixel coordinates
    boxes_np = boxes.detach().cpu().numpy().copy()
    boxes_np[:, 0] *= width    # x1
    boxes_np[:, 2] *= width    # x2
    boxes_np[:, 1] *= height   # y1
    boxes_np[:, 3] *= height   # y2

    # agentness as detection confidence — [N_kept, 1] for hstack
    conf = probs["agentness"].squeeze(1).detach().cpu().numpy()[:, None]
    out = {"agent_ness": [np.hstack([boxes_np, conf]).astype(np.float32)]}

    for head, n_classes in C.HEAD_SIZES.items():
        # Raw per-class sigmoid — matches baseline's scoring (no agentness multiplication).
        # The baseline (val.py) uses confidence[b, s, :, cc] directly per class.
        scores = probs[head].detach().cpu().numpy()  # [N_kept, C]
        per_class = []
        for cid in range(n_classes):
            cls_scores = scores[:, cid]
            keep = cls_scores > class_threshold
            if keep.any():
                per_class.append(
                    np.hstack([boxes_np[keep], cls_scores[keep, None]]).astype(np.float32)
                )
            else:
                per_class.append(np.zeros((0, 5), dtype=np.float32))
        out[head] = per_class
    return out


def summarize_results(results: dict) -> dict:
    """Extract mAP and mR (mean recall) from the baseline evaluator's output."""
    summary = {}
    for label_type in ("agent_ness", "agent", "action", "loc", "duplex", "triplet"):
        if label_type in results:
            summary[label_type] = {
                "mAP": round(float(results[label_type]["mAP"]), 6),
                "mR":  round(float(results[label_type]["mR"]), 6),
            }
    return summary


RESULTS_CSV = Path(__file__).resolve().parents[2] / "results" / "val_metrics.csv"
CSV_FIELDS = [
    "model", "source", "status", "epoch", "metric", "split", "iou",
    "agent_ness", "agent", "action", "loc", "duplex", "triplet",
]


def write_to_csv(
    csv_path: Path,
    model_name: str,
    source: str,
    status: str,
    epoch: int,
    metric: str,
    split: str,
    iou: float | str,
    summary: dict,
) -> None:
    """
    Append (or update) one row in the shared results CSV.

    Uniqueness key: (model, epoch, metric, split). If a row with the same key
    already exists it is replaced in-place; otherwise the row is appended.
    This lets you re-run eval without duplicating entries.
    """
    new_row = {
        "model":  model_name,
        "source": source,
        "status": status,
        "epoch":  epoch,
        "metric": metric,
        "split":  split,
        "iou":    iou,
        "agent_ness": summary.get("agent_ness", {}).get("mAP", ""),
        "agent":      summary.get("agent",      {}).get("mAP", ""),
        "action":     summary.get("action",     {}).get("mAP", ""),
        "loc":        summary.get("loc",        {}).get("mAP", ""),
        "duplex":     summary.get("duplex",     {}).get("mAP", ""),
        "triplet":    summary.get("triplet",    {}).get("mAP", ""),
    }

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    key = lambda r: (r["model"], str(r["epoch"]), r["metric"], r["split"])
    new_key = key(new_row)

    if csv_path.exists():
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

    replaced = False
    for i, r in enumerate(rows):
        if key(r) == new_key:
            rows[i] = new_row
            replaced = True
            break
    if not replaced:
        rows.append(new_row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    action = "Updated" if replaced else "Appended"
    print(f"{action} results in {csv_path}")


def tube_iou(
    pred_boxes: torch.Tensor,   # [T, 4] in [x1,y1,x2,y2]
    gt_boxes: torch.Tensor,     # [T, 4] in [x1,y1,x2,y2]
    pred_mask: torch.Tensor,    # [T] bool
    gt_mask: torch.Tensor,      # [T] bool
) -> float:
    """
    Temporal IoU between predicted and GT tubes: average spatial IoU over frames
    where both tubes have boxes.

    Frames where only one tube has a box contribute zero IoU (no overlap possible).
    This is the standard ROAD++ tube IoU definition used in the ECCV challenge.
    """
    overlap = pred_mask & gt_mask  # frames where both are present
    if not overlap.any():
        return 0.0
    ious = []
    for t in torch.where(overlap)[0]:
        # box_iou is [N,M]; with 1 pred and 1 gt it returns [[iou]]
        ious.append(float(box_iou(pred_boxes[t : t + 1], gt_boxes[t : t + 1]).item()))
    return float(sum(ious) / max(len(ious), 1))


def approximate_video_eval(
    model, processor, anno_file: Path, frames_dir: Path,
    subset: str, device, dtype, threshold: float
):
    """
    Video-level evaluation: for each GT tube, find the best-matching predicted tube
    and record its tube IoU. Average over all GT tubes = mean best tube IoU.

    This is an approximation of video-mAP because:
        True video-mAP requires scoring predicted tubes against ALL GT tubes and
        computing AP over the confidence ranking. Here we just measure "how well
        do we cover each GT tube at all?" — a recall-like metric.

    Runs one clip per video (first CLIP_LEN annotated frames) rather than the
    full video. This is fast enough for iteration but misses agents that only
    appear late in the video.
    """
    with open(anno_file) as f:
        data = json.load(f)

    ap_like_scores = []
    for videoname, vdata in data["db"].items():
        if not is_in_subset(vdata.get("split_ids", []), subset):
            continue
        frame_ids = sorted(
            [fid for fid, fd in vdata.get("frames", {}).items() if fd.get("annotated", 0) == 1],
            key=lambda x: int(x),
        )
        if len(frame_ids) < C.CLIP_LEN:
            continue

        clip_ids = frame_ids[: C.CLIP_LEN]
        pil_frames = []
        frame_targets = []
        for fid in clip_ids:
            img_path = frames_dir / videoname / f"{int(fid):05d}.jpg"
            pil_frames.append(Image.open(img_path).convert("RGB"))
            annos = vdata["frames"][fid].get("annos", {})
            parsed = [anno for anno in annos.values()
                      if isinstance(anno, dict) and anno.get("box") is not None]
            frame_targets.append(parsed if parsed else None)

        pixel_values, image_grid_thw = preprocess_clip(pil_frames, processor, device, dtype)
        outputs = model(pil_frames, pixel_values, image_grid_thw)
        probs = {k: v.sigmoid() for k, v in outputs["pred_logits"].items()}

        # Filter to confident detections
        keep = probs["agentness"].squeeze(1) > threshold
        if not keep.any():
            continue

        pred_boxes = to_xyxy(outputs["pred_boxes"][keep])          # [N_kept, T, 4]
        pred_mask = torch.ones(pred_boxes.shape[:2], dtype=torch.bool, device=pred_boxes.device)

        # Parse frame_targets into GT format for tube IoU
        parsed_targets = []
        for annos in frame_targets:
            if annos is None:
                parsed_targets.append(None)
                continue
            boxes = torch.tensor([anno["box"] for anno in annos], dtype=torch.float32, device=device)
            # Labels not needed for tube IoU — just box positions
            agent  = torch.zeros(len(annos), C.N_AGENTS, device=device)
            action = torch.zeros(len(annos), C.N_ACTIONS, device=device)
            loc    = torch.zeros(len(annos), C.N_LOCS, device=device)
            duplex  = torch.zeros(len(annos), C.N_DUPLEXES, device=device)
            triplet = torch.zeros(len(annos), C.N_TRIPLETS, device=device)
            parsed_targets.append({"boxes": boxes, "agent": agent, "action": action,
                                    "loc": loc, "duplex": duplex, "triplet": triplet})

        gt_tubes = greedy_group_tubes(parsed_targets, iou_thresh=C.TUBE_LINK_IOU)
        for tube in gt_tubes:
            # For each GT tube, find the predicted tube with best overlap
            best = 0.0
            for q in range(pred_boxes.shape[0]):
                best = max(best, tube_iou(pred_boxes[q], tube["boxes"], pred_mask[q], tube["box_mask"]))
            ap_like_scores.append(best)

    return {
        "mean_best_tube_iou": round(float(sum(ap_like_scores) / max(len(ap_like_scores), 1)), 6),
        "n_tubes": len(ap_like_scores),
        "note": "Approximate tube metric for early experiment iteration; replace with official video AP when available.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=C.MODEL_ID)
    parser.add_argument("--ckpt",       default=str(Path(C.CKPT_DIR) / "best.pt"))
    parser.add_argument("--anno",       default=C.ANNO_FILE)
    parser.add_argument("--frames",     default=C.FRAMES_DIR)
    parser.add_argument("--subset",     default="val")
    parser.add_argument("--mode",       choices=["frame", "video"], default="frame")
    parser.add_argument("--out",        default=None)
    parser.add_argument("--det-pkl",    default=str(Path(C.LOG_DIR) / "baseline_compat_dets.pkl"))
    parser.add_argument("--csv",        default=str(RESULTS_CSV),
                        help="Path to shared results CSV (default: results/val_metrics.csv)")
    parser.add_argument("--model-name", default=None,
                        help="Row label for the CSV (default: auto-generated from checkpoint)")
    parser.add_argument("--source",          default="novel",
                        help="Source column value in CSV (default: novel)")
    parser.add_argument("--status",          default="training",
                        help="Status column value in CSV (default: training)")
    parser.add_argument("--no-csv",          action="store_true",
                        help="Skip writing to the results CSV")
    parser.add_argument("--sync-sharepoint", action="store_true",
                        help="Push results/val_metrics.csv to OneDrive after writing CSV")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(args.model)

    # Rebuild the model architecture exactly as in training
    model = EfficientNetFPNDETRModel(
        model_id=args.model,
        vit_dim=C.VIT_DIM,
        d_model=C.D_MODEL,
        head_sizes=C.HEAD_SIZES,
        clip_len=C.CLIP_LEN,
        num_queries=C.NUM_QUERIES,
        num_decoder_layers=C.NUM_DECODER_LAYERS,
        nhead=C.NHEAD,
        dim_ffn=C.DIM_FFN,
        dropout=C.DROPOUT,
        n_deform_points=C.N_DEFORM_POINTS,
        backbone_name=C.BACKBONE,
        backbone_freeze_blocks=C.BACKBONE_FREEZE_BLOCKS,
        fpn_in_channels=C.FPN_IN_CHANNELS,
        gate_init=C.VLM_GATE_INIT,
        lora_r=C.LORA_R,
        lora_alpha=C.LORA_ALPHA,
        lora_dropout=C.LORA_DROPOUT,
        lora_target_modules=C.LORA_TARGET_MODULES,
        lora_n_layers=C.LORA_N_LAYERS,
    )
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(device)
    model.eval()

    if args.mode == "video":
        summary = approximate_video_eval(
            model, processor, Path(args.anno), Path(args.frames),
            args.subset, device, dtype, C.CONFIDENCE_THRESHOLD
        )
        out_path = Path(args.out or (Path(C.LOG_DIR) / "eval_vmap.json"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))
        return

    # --- Frame-level mAP ---
    # Strategy: run the model on 8-frame clips (matching training), extract per-frame
    # boxes from the tube predictions, and collect into the detection dict for
    # evaluate_frames().
    #
    # Key design decisions matching the baseline (val.py):
    #   - Full clip inference: the DETR decoder was trained on 8-frame clips with
    #     2048 spatiotemporal tokens. Running single-frame (256 tokens) is a huge
    #     distribution shift. We run proper 8-frame clips.
    #   - No pre-filtering: the baseline passes ALL detections to the AP evaluator,
    #     which sweeps thresholds internally. Pre-filtering removes detections the
    #     sweep needs.
    #   - Raw per-class scores: the baseline scores each class independently via
    #     raw sigmoid, not objectness × class. (See frame_to_detection_lists.)

    with open(args.anno) as f:
        anno_data = json.load(f)

    # Initialise the detection accumulator — one entry per head, one key per frame
    detections = {k: {} for k in ["agent_ness", "agent", "action", "loc", "duplex", "triplet"]}
    frame_size = None

    # Build non-overlapping 8-frame clips covering all annotated frames
    clips = []  # list of (videoname, [frame_id_str, ...])
    for videoname, vdata in anno_data["db"].items():
        if not is_in_subset(vdata.get("split_ids", []), args.subset):
            continue
        ann_fids = sorted(
            [fid for fid, fd in vdata.get("frames", {}).items()
             if fd.get("annotated", 0) == 1],
            key=lambda x: int(x),
        )
        if not ann_fids:
            continue
        for start in range(0, len(ann_fids), C.CLIP_LEN):
            clip = ann_fids[start : start + C.CLIP_LEN]
            if len(clip) < C.CLIP_LEN:
                # Pad short tail by repeating last frame
                clip = clip + [clip[-1]] * (C.CLIP_LEN - len(clip))
            clips.append((videoname, clip))

    seen_frames = set()  # avoid duplicate detections from padded frames
    n_clips = len(clips)

    with torch.no_grad():
        for clip_idx, (videoname, clip_fids) in enumerate(clips):
            pil_frames = []
            for fid in clip_fids:
                img_path = Path(args.frames) / videoname / f"{int(fid):05d}.jpg"
                pil_frames.append(Image.open(img_path).convert("RGB"))

            width, height = pil_frames[0].size
            frame_size = (height, width)

            pixel_values, image_grid_thw = preprocess_clip(
                pil_frames, processor, device, dtype
            )
            outputs = model(pil_frames, pixel_values, image_grid_thw)
            probs = {k: v.sigmoid() for k, v in outputs["pred_logits"].items()}

            # Filter queries by agentness confidence. AP computation
            # sweeps thresholds internally, so dropping near-zero confidence
            # queries doesn't affect mAP — they'd rank last anyway.
            # This cuts ~90% of detections and dramatically speeds up
            # evaluate_frames().
            keep = probs["agentness"].squeeze(1) > 0.01
            kept_probs = {k: v[keep] for k, v in probs.items()}

            for t, fid in enumerate(clip_fids):
                frame_key = videoname + f"{int(fid):05d}"
                if frame_key in seen_frames:
                    continue  # skip duplicates from padding
                seen_frames.add(frame_key)

                # pred_boxes[:, t, :] = frame t's boxes from each query's tube
                boxes_t = to_xyxy(outputs["pred_boxes"][keep, t, :])  # [N_kept, 4] in [0,1]

                det = frame_to_detection_lists(
                    boxes_t, kept_probs, width, height, class_threshold=0.0
                )
                for head, value in det.items():
                    detections[head][frame_key] = value

            if (clip_idx + 1) % 50 == 0 or clip_idx == n_clips - 1:
                print(f"  clip {clip_idx + 1}/{n_clips}", flush=True)

    # Serialise detections to disk — evaluate_frames() reads from a pickle file
    det_pkl = Path(args.det_pkl)
    det_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(det_pkl, "wb") as f:
        pickle.dump(detections, f)

    assert frame_size is not None, "No annotated frames found for baseline-compatible eval"
    height, width = frame_size

    # The baseline evaluator uses global variables for image dimensions (legacy API)
    baseline_datasets.g_w = height
    baseline_datasets.g_h = width

    results = evaluate_frames(
        args.anno,
        str(det_pkl),
        args.subset,
        wh=[height, width],
        iou_thresh=0.5,        # standard IoU threshold — a box must overlap GT by ≥50%
        dataset="road_waymo",
    )
    summary = summarize_results(results)

    out_path = Path(args.out or (Path(C.LOG_DIR) / "eval_fmap.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    if not args.no_csv:
        ckpt_data = torch.load(args.ckpt, map_location="cpu", weights_only=True) \
            if Path(args.ckpt).exists() else {}
        epoch = ckpt_data.get("epoch", "?")
        model_name = args.model_name or "Exp2b-EfficientNet-DeformDETR-Godel"
        write_to_csv(
            csv_path=Path(args.csv),
            model_name=model_name,
            source=args.source,
            status=args.status,
            epoch=epoch,
            metric="f-mAP",
            split=args.subset,
            iou=0.5,
            summary=summary,
        )

    if args.sync_sharepoint:
        sync_script = Path(__file__).resolve().parents[2] / "results" / "sync_to_sharepoint.py"
        import subprocess
        subprocess.run([sys.executable, str(sync_script)], check=True)


if __name__ == "__main__":
    main()

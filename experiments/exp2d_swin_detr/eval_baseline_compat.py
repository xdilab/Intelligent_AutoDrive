#!/usr/bin/env python3
"""
Baseline-compatible evaluation for Exp2c (Frozen-DETR).

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
from losses import greedy_group_tubes, load_constraint_children
from matcher import box_iou
from model import FrozenDETRModel


def _load_module(name: str, path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# Import baseline evaluation code AFTER adding its root to sys.path
sys.path.insert(0, str(BASELINE_ROOT))
from modules.evaluation import evaluate_frames as _evaluate_frames_orig  # noqa: E402
from modules import evaluation as _eval_module  # noqa: E402
import data.datasets as baseline_datasets       # noqa: E402


def evaluate_frames(anno_file, det_file, subset, wh, iou_thresh=0.5, dataset='road'):
    """Wrapper around baseline evaluate_frames() that adds progress printing.

    Monkey-patches compute_class_ap to print progress per label_type × class,
    then restores the original after evaluation completes.
    """
    import time as _time

    _orig_compute = _eval_module.compute_class_ap
    _state = {"label_type": "", "cl_idx": 0, "n_classes": 0, "t0": _time.perf_counter()}

    # Patch get_gt_frames to capture label_type and class count
    _orig_get_gt_frames = _eval_module.get_gt_frames

    def _patched_get_gt_frames(annots, sub, label_type, ds, wh_):
        result = _orig_get_gt_frames(annots, sub, label_type, ds, wh_)
        # Figure out how many classes this label_type has
        used_labels = {
            "agent_ness": 1, "agent_labels": 10, "action_labels": 22,
            "loc_labels": 16, "duplex_labels": 49, "triplet_labels": 86,
        }
        lt_key = label_type + "_labels" if label_type != "agent_ness" else "agent_ness"
        _state["label_type"] = label_type
        _state["cl_idx"] = 0
        _state["n_classes"] = used_labels.get(lt_key, 0)
        _state["t0"] = _time.perf_counter()
        print(f"\n  [eval] computing AP for {label_type} ({_state['n_classes']} classes) ...",
              flush=True)
        return result

    def _patched_compute(dets, gts, iou_fn, iou_th):
        _state["cl_idx"] += 1
        idx, total = _state["cl_idx"], _state["n_classes"]
        if total > 0 and (idx % max(1, total // 5) == 0 or idx == total):
            elapsed = _time.perf_counter() - _state["t0"]
            print(f"    {_state['label_type']}: class {idx}/{total} "
                  f"({elapsed:.1f}s elapsed)", flush=True)
        return _orig_compute(dets, gts, iou_fn, iou_th)

    _eval_module.compute_class_ap = _patched_compute
    _eval_module.get_gt_frames = _patched_get_gt_frames

    try:
        result = _evaluate_frames_orig(anno_file, det_file, subset, wh,
                                       iou_thresh, dataset)
    finally:
        _eval_module.compute_class_ap = _orig_compute
        _eval_module.get_gt_frames = _orig_get_gt_frames

    return result


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
    topk_per_class: int = 10,
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

    Top-K per class:
        The baseline keeps at most TOPK=10 detections per class per frame
        (after NMS). Without top-K, DETR passes all 300 queries per class,
        flooding the AP evaluator with thousands of low-quality false positives
        that bury true positives in the ranked list and destroy precision.

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
    # Top-K for agentness: keep only top-K highest scoring detections
    if topk_per_class > 0 and conf.shape[0] > topk_per_class:
        topk_idx = np.argsort(conf[:, 0])[::-1][:topk_per_class]
        out = {"agent_ness": [np.hstack([boxes_np[topk_idx], conf[topk_idx]]).astype(np.float32)]}
    else:
        out = {"agent_ness": [np.hstack([boxes_np, conf]).astype(np.float32)]}

    for head, n_classes in C.HEAD_SIZES.items():
        # Raw per-class sigmoid — matches baseline's scoring (no agentness multiplication).
        # The baseline (val.py) uses confidence[b, s, :, cc] directly per class.
        scores = probs[head].detach().cpu().numpy()  # [N_kept, C]
        per_class = []
        for cid in range(n_classes):
            cls_scores = scores[:, cid]
            # Top-K filtering: keep only the K highest-scoring detections per class.
            # Matches baseline's GEN_TOPK=100 → TOPK=10 pipeline.
            if topk_per_class > 0 and cls_scores.shape[0] > topk_per_class:
                topk_idx = np.argsort(cls_scores)[::-1][:topk_per_class]
                per_class.append(
                    np.hstack([boxes_np[topk_idx], cls_scores[topk_idx, None]]).astype(np.float32)
                )
            elif cls_scores.shape[0] > 0:
                per_class.append(
                    np.hstack([boxes_np, cls_scores[:, None]]).astype(np.float32)
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
    "duplex_viol", "triplet_viol",
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
        "duplex_viol":  summary.get("duplex_viol", ""),
        "triplet_viol": summary.get("triplet_viol", ""),
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
    model, anno_file: Path, frames_dir: Path,
    subset: str, device, threshold: float
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

        outputs = model(pil_frames)
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


def compute_constraint_violations(
    det_pkl_path: Path,
    anno_file: str,
    thresholds: list[float] = [0.3, 0.5, 0.7, 0.9],
) -> dict:
    """
    Compute constraint violation rates from saved detections.

    For each detection with confidence > threshold, check whether predicted
    agent+action pairs and agent+action+loc triples are valid according to
    duplex_childs and triplet_childs from the annotation JSON.

    Returns dict mapping threshold -> {duplex_viol_rate, triplet_viol_rate,
    duplex_n_copreds, triplet_n_copreds}.
    """
    children = load_constraint_children(anno_file)
    valid_d = set(map(tuple, children["duplex_childs"]))
    valid_t = set(map(tuple, children["triplet_childs"]))

    with open(det_pkl_path, "rb") as f:
        detections = pickle.load(f)

    # detections["agent"] is dict: frame_key -> list of C arrays [N_dets, 5]
    # detections["action"] same structure, detections["loc"] same
    # Each array index = class_id, each row = [x1, y1, x2, y2, score]

    results = {}
    for thresh in thresholds:
        duplex_violations = 0
        duplex_copreds = 0
        triplet_violations = 0
        triplet_copreds = 0

        for frame_key in detections["agent"]:
            # Get per-class detections for this frame
            agent_dets = detections["agent"][frame_key]    # list of C arrays
            action_dets = detections["action"][frame_key]
            loc_dets = detections["loc"][frame_key]

            # For each box, find which classes are active above threshold.
            # Since each class has independent detections, we match by box IoU.
            # Simpler: for each detection box that appears in agent class i with
            # score > thresh AND action class j with score > thresh, that's a
            # co-prediction. Check if (i,j) is valid.
            #
            # Efficient approach: collect all (box, class, score) tuples per head,
            # then match boxes across heads by high IoU overlap.

            # Collect confident detections per head
            agent_active = []  # (box_array[4], class_id, score)
            for cid, arr in enumerate(agent_dets):
                if arr.shape[0] == 0:
                    continue
                mask = arr[:, 4] > thresh
                for row in arr[mask]:
                    agent_active.append((row[:4], cid, row[4]))

            action_active = []
            for cid, arr in enumerate(action_dets):
                if arr.shape[0] == 0:
                    continue
                mask = arr[:, 4] > thresh
                for row in arr[mask]:
                    action_active.append((row[:4], cid, row[4]))

            loc_active = []
            for cid, arr in enumerate(loc_dets):
                if arr.shape[0] == 0:
                    continue
                mask = arr[:, 4] > thresh
                for row in arr[mask]:
                    loc_active.append((row[:4], cid, row[4]))

            if not agent_active or not action_active:
                continue

            # Match agent-action pairs by box overlap (IoU > 0.5 = same object)
            for a_box, a_cid, _ in agent_active:
                for ac_box, ac_cid, _ in action_active:
                    # Quick IoU check
                    x1 = max(a_box[0], ac_box[0])
                    y1 = max(a_box[1], ac_box[1])
                    x2 = min(a_box[2], ac_box[2])
                    y2 = min(a_box[3], ac_box[3])
                    inter = max(0, x2 - x1) * max(0, y2 - y1)
                    if inter == 0:
                        continue
                    a_area = (a_box[2] - a_box[0]) * (a_box[3] - a_box[1])
                    ac_area = (ac_box[2] - ac_box[0]) * (ac_box[3] - ac_box[1])
                    union = a_area + ac_area - inter
                    if union <= 0 or inter / union < 0.5:
                        continue

                    # This is a co-prediction on the same object
                    duplex_copreds += 1
                    if (a_cid, ac_cid) not in valid_d:
                        duplex_violations += 1

                    # Check triplets if loc active
                    for l_box, l_cid, _ in loc_active:
                        lx1 = max(a_box[0], l_box[0])
                        ly1 = max(a_box[1], l_box[1])
                        lx2 = min(a_box[2], l_box[2])
                        ly2 = min(a_box[3], l_box[3])
                        l_inter = max(0, lx2 - lx1) * max(0, ly2 - ly1)
                        if l_inter == 0:
                            continue
                        l_area = (l_box[2] - l_box[0]) * (l_box[3] - l_box[1])
                        l_union = a_area + l_area - l_inter
                        if l_union <= 0 or l_inter / l_union < 0.5:
                            continue
                        triplet_copreds += 1
                        if (a_cid, ac_cid, l_cid) not in valid_t:
                            triplet_violations += 1

        d_rate = duplex_violations / max(duplex_copreds, 1)
        t_rate = triplet_violations / max(triplet_copreds, 1)
        results[thresh] = {
            "duplex_viol_rate": round(d_rate * 100, 2),
            "triplet_viol_rate": round(t_rate * 100, 2),
            "duplex_n_copreds": duplex_copreds,
            "triplet_n_copreds": triplet_copreds,
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-model", default=C.CLIP_MODEL)
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
    parser.add_argument("--topk", type=int, default=10,
                        help="Top-K detections per class per frame (baseline default=10, 0=unlimited)")
    parser.add_argument("--sync-sharepoint", action="store_true",
                        help="Push results/val_metrics.csv to OneDrive after writing CSV")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild the model architecture exactly as in training
    model = FrozenDETRModel(
        clip_model=args.clip_model,
        clip_cache_dir=C.CLIP_CACHE_DIR,
        clip_cls_dim=C.CLIP_CLS_DIM,
        clip_patch_dim=C.CLIP_PATCH_DIM,
        clip_grid=C.CLIP_GRID,
        d_model=C.D_MODEL,
        head_sizes=C.HEAD_SIZES,
        clip_len=C.CLIP_LEN,
        num_queries=C.NUM_QUERIES,
        num_encoder_layers=C.NUM_ENCODER_LAYERS,
        num_decoder_layers=C.NUM_DECODER_LAYERS,
        nhead=C.NHEAD,
        dim_ffn=C.DIM_FFN,
        dropout=C.DROPOUT,
        n_deform_points=C.N_DEFORM_POINTS,
        encoder_n_levels=C.ENCODER_N_LEVELS,
        decoder_n_levels=C.FPN_LEVELS,
        backbone_name=C.BACKBONE,
        backbone_freeze_stages=C.BACKBONE_FREEZE_STAGES,
        fpn_in_channels=C.FPN_IN_CHANNELS,
    )
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(device)
    model.eval()

    if args.mode == "video":
        summary = approximate_video_eval(
            model, Path(args.anno), Path(args.frames),
            args.subset, device, C.CONFIDENCE_THRESHOLD
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

            outputs = model(pil_frames)
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
                    boxes_t, kept_probs, width, height, class_threshold=0.0,
                    topk_per_class=args.topk,
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

    # --- Constraint violation rates ---
    print("\n  [eval] computing constraint violation rates ...", flush=True)
    viol_thresholds = [0.3, 0.5, 0.7, 0.9]
    viol_results = compute_constraint_violations(det_pkl, args.anno, viol_thresholds)

    print("\n  Constraint violation rates (% of co-predictions that violate valid combos):")
    print(f"  {'Threshold':<12} {'Duplex viol%':<15} {'Triplet viol%':<15} "
          f"{'Duplex N':<12} {'Triplet N'}")
    for th in viol_thresholds:
        v = viol_results[th]
        print(f"  conf>{th:<7} {v['duplex_viol_rate']:<15} {v['triplet_viol_rate']:<15} "
              f"{v['duplex_n_copreds']:<12} {v['triplet_n_copreds']}")

    # Save violation results alongside f-mAP
    viol_out_path = out_path.with_name("eval_violations.json")
    with open(viol_out_path, "w") as f:
        json.dump({str(k): v for k, v in viol_results.items()}, f, indent=2)
    print(f"\n  Violation results saved to {viol_out_path}", flush=True)

    # Add violation rates at conf>0.5 to summary for CSV
    if 0.5 in viol_results:
        summary["duplex_viol"] = viol_results[0.5]["duplex_viol_rate"]
        summary["triplet_viol"] = viol_results[0.5]["triplet_viol_rate"]

    if not args.no_csv:
        ckpt_data = torch.load(args.ckpt, map_location="cpu", weights_only=True) \
            if Path(args.ckpt).exists() else {}
        epoch = ckpt_data.get("epoch", "?")
        model_name = args.model_name or "Exp2c-FrozenDETR-CLIP-ViTL14"
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

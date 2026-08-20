#!/usr/bin/env python3
"""
Baseline-compatible evaluation for Exp2g (R50 + Frozen-DETR + CLIP + MS-DETR).

Unpacks the flat 184-dim vector into per-head dicts for the baseline evaluator.
Agent dims use softmax (single-label), rest use sigmoid.
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
import torch.nn.functional as F
from PIL import Image

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]
EXP1_DIR = REPO_ROOT / "experiments" / "exp1_road_r"
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


sys.path.insert(0, str(BASELINE_ROOT))
from modules.evaluation import evaluate_frames as _evaluate_frames_orig  # noqa: E402
from modules import evaluation as _eval_module  # noqa: E402
import data.datasets as baseline_datasets       # noqa: E402


def evaluate_frames(anno_file, det_file, subset, wh, iou_thresh=0.5, dataset='road'):
    """Wrapper around baseline evaluate_frames() with progress printing."""
    import time as _time

    _orig_compute = _eval_module.compute_class_ap
    _state = {"label_type": "", "cl_idx": 0, "n_classes": 0, "t0": _time.perf_counter()}
    _orig_get_gt_frames = _eval_module.get_gt_frames

    def _patched_get_gt_frames(annots, sub, label_type, ds, wh_):
        result = _orig_get_gt_frames(annots, sub, label_type, ds, wh_)
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
    if isinstance(split_ids, str):
        split_ids = [split_ids]
    return subset in split_ids


def to_xyxy(boxes_cxcywh: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes_cxcywh.unbind(-1)
    x1 = (cx - 0.5 * w).clamp(0.0, 1.0)
    y1 = (cy - 0.5 * h).clamp(0.0, 1.0)
    x2 = (cx + 0.5 * w).clamp(0.0, 1.0)
    y2 = (cy + 0.5 * h).clamp(0.0, 1.0)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def unpack_flat_probs(flat_logits: torch.Tensor) -> dict:
    """
    Unpack flat 184-dim logits into per-head probability dict.

    Agent dims [1:11] use softmax (single-label) when USE_SOFTMAX_AGENT is set.
    All other dims use sigmoid.

    Layout: [agentness(1), agent(10), action(22), loc(16), duplex(49), triplet(86)]
    """
    off = C.CLS_OFFSETS
    agent_start = off["agent"]
    agent_end = agent_start + C.N_AGENTS

    if getattr(C, "USE_SOFTMAX_AGENT", False):
        agent_probs = F.softmax(flat_logits[:, agent_start:agent_end], dim=-1)
    else:
        agent_probs = flat_logits[:, agent_start:agent_end].sigmoid()

    return {
        "agentness": flat_logits[:, 0:1].sigmoid(),
        "agent":     agent_probs,
        "action":    flat_logits[:, off["action"]:off["action"] + C.N_ACTIONS].sigmoid(),
        "loc":       flat_logits[:, off["loc"]:off["loc"] + C.N_LOCS].sigmoid(),
        "duplex":    flat_logits[:, off["duplex"]:off["duplex"] + C.N_DUPLEXES].sigmoid(),
        "triplet":   flat_logits[:, off["triplet"]:off["triplet"] + C.N_TRIPLETS].sigmoid(),
    }


def frame_to_detection_lists(
    boxes: torch.Tensor,
    probs: dict,
    width: int,
    height: int,
    class_threshold: float,
    topk_per_class: int = 10,
):
    """Convert DETR predictions to baseline evaluator format."""

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

    boxes_np = boxes.detach().cpu().numpy().copy()
    boxes_np[:, 0] *= width
    boxes_np[:, 2] *= width
    boxes_np[:, 1] *= height
    boxes_np[:, 3] *= height

    conf = probs["agentness"].squeeze(1).detach().cpu().numpy()[:, None]
    if topk_per_class > 0 and conf.shape[0] > topk_per_class:
        topk_idx = np.argsort(conf[:, 0])[::-1][:topk_per_class]
        out = {"agent_ness": [np.hstack([boxes_np[topk_idx], conf[topk_idx]]).astype(np.float32)]}
    else:
        out = {"agent_ness": [np.hstack([boxes_np, conf]).astype(np.float32)]}

    for head, n_classes in C.HEAD_SIZES.items():
        scores = probs[head].detach().cpu().numpy()
        per_class = []
        for cid in range(n_classes):
            cls_scores = scores[:, cid]
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


def compute_constraint_violations(
    det_pkl_path: Path,
    anno_file: str,
    thresholds: list[float] = [0.3, 0.5, 0.7, 0.9],
) -> dict:
    children = load_constraint_children(anno_file)
    valid_d = set(map(tuple, children["duplex_childs"]))
    valid_t = set(map(tuple, children["triplet_childs"]))

    with open(det_pkl_path, "rb") as f:
        detections = pickle.load(f)

    results = {}
    for thresh in thresholds:
        duplex_violations = 0
        duplex_copreds = 0
        triplet_violations = 0
        triplet_copreds = 0

        for frame_key in detections["agent"]:
            agent_dets = detections["agent"][frame_key]
            action_dets = detections["action"][frame_key]
            loc_dets = detections["loc"][frame_key]

            agent_active = []
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

            for a_box, a_cid, _ in agent_active:
                for ac_box, ac_cid, _ in action_active:
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

                    duplex_copreds += 1
                    if (a_cid, ac_cid) not in valid_d:
                        duplex_violations += 1

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
    parser.add_argument("--csv",        default=str(RESULTS_CSV))
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--source",          default="novel")
    parser.add_argument("--status",          default="training")
    parser.add_argument("--no-csv",          action="store_true")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--sync-sharepoint", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model — MS-DETR (two-stage + O2M + softmax agent)
    model = FrozenDETRModel(
        clip_model=args.clip_model,
        clip_cache_dir=C.CLIP_CACHE_DIR,
        clip_cls_dim=C.CLIP_CLS_DIM,
        clip_patch_dim=C.CLIP_PATCH_DIM,
        clip_grid=C.CLIP_GRID,
        d_model=C.D_MODEL,
        num_classes=C.NUM_CLASSES,
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
        fpn_in_channels=C.FPN_IN_CHANNELS,
        val_short_side=C.VAL_SHORT_SIDE,
        val_max_size=C.VAL_MAX_SIZE,
        two_stage=C.TWO_STAGE,
        mixed_selection=C.MIXED_SELECTION,
        use_ms_detr=C.USE_MS_DETR,
        use_aux_ffn=C.USE_AUX_FFN,
    )
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(device)
    model.eval()

    # --- Frame-level mAP ---
    with open(args.anno) as f:
        anno_data = json.load(f)

    detections = {k: {} for k in ["agent_ness", "agent", "action", "loc", "duplex", "triplet"]}
    frame_size = None

    clips = []
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
                clip = clip + [clip[-1]] * (C.CLIP_LEN - len(clip))
            clips.append((videoname, clip))

    seen_frames = set()
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

            # Unpack flat 184-dim logits → per-head probs (softmax on agent, sigmoid on rest)
            probs = unpack_flat_probs(outputs["pred_logits"])

            keep = probs["agentness"].squeeze(1) > 0.01
            kept_probs = {k: v[keep] for k, v in probs.items()}

            for t, fid in enumerate(clip_fids):
                frame_key = videoname + f"{int(fid):05d}"
                if frame_key in seen_frames:
                    continue
                seen_frames.add(frame_key)

                boxes_t = to_xyxy(outputs["pred_boxes"][keep, t, :])

                det = frame_to_detection_lists(
                    boxes_t, kept_probs, width, height, class_threshold=0.0,
                    topk_per_class=args.topk,
                )
                for head, value in det.items():
                    detections[head][frame_key] = value

            if (clip_idx + 1) % 50 == 0 or clip_idx == n_clips - 1:
                print(f"  clip {clip_idx + 1}/{n_clips}", flush=True)

    det_pkl = Path(args.det_pkl)
    det_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(det_pkl, "wb") as f:
        pickle.dump(detections, f)

    assert frame_size is not None, "No annotated frames found"
    height, width = frame_size

    baseline_datasets.g_w = height
    baseline_datasets.g_h = width

    results = evaluate_frames(
        args.anno,
        str(det_pkl),
        args.subset,
        wh=[height, width],
        iou_thresh=0.5,
        dataset="road_waymo",
    )
    summary = summarize_results(results)

    out_path = Path(args.out or (Path(C.LOG_DIR) / "eval_fmap.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # Constraint violations
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

    viol_out_path = out_path.with_name("eval_violations.json")
    with open(viol_out_path, "w") as f:
        json.dump({str(k): v for k, v in viol_results.items()}, f, indent=2)
    print(f"\n  Violation results saved to {viol_out_path}", flush=True)

    if 0.5 in viol_results:
        summary["duplex_viol"] = viol_results[0.5]["duplex_viol_rate"]
        summary["triplet_viol"] = viol_results[0.5]["triplet_viol_rate"]

    if not args.no_csv:
        ckpt_data = torch.load(args.ckpt, map_location="cpu", weights_only=True) \
            if Path(args.ckpt).exists() else {}
        epoch = ckpt_data.get("epoch", "?")
        model_name = args.model_name or "Exp2g-R50-FrozenDETR-CLIP-MSDETR"
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

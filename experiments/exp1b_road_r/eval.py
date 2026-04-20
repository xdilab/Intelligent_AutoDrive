#!/usr/bin/env python3
"""
Experiment 1b evaluation script — dense detection with NMS post-processing.

Usage
-----
# Evaluate best checkpoint:
python -u experiments/exp1b_road_r/eval.py

# Evaluate a specific checkpoint:
python -u experiments/exp1b_road_r/eval.py --ckpt experiments/exp1b_road_r/checkpoints/best.pt

# Sweep agentness threshold:
python -u experiments/exp1b_road_r/eval.py --sweep-threshold

# Save full results to JSON:
python -u experiments/exp1b_road_r/eval.py --out experiments/exp1b_road_r/logs/eval_results.json

Inference pipeline
------------------
1. Forward pass → dense per-token predictions
2. Threshold by agentness > τ to get foreground token candidates
3. Decode FCOS ltrb → [x1,y1,x2,y2] using token center coordinates
4. Per-frame NMS on candidate boxes (torchvision.ops.nms, IoU > 0.5)
5. Evaluate classification heads on surviving detections

Metrics reported
----------------
For each label type (agent / action / loc / duplex / triplet):
  - Per-class: precision, recall, F1 at threshold=0.5
  - mAP: mean AP across classes with ≥1 positive GT instance
  (Note: GT matching is by agentness-threshold assignment, not IoU; for proper
   detection mAP by IoU use a separate evaluation harness against the annotation file)
Also reports:
  - T-norm constraint violation rate
  - Detection statistics: avg foreground tokens/frame after NMS
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

_REPO_ROOT = str(Path(__file__).parent.parent.parent)
_EXP1B_DIR = str(Path(__file__).parent)
_EXP1_DIR  = str(Path(__file__).parent.parent / "exp1_road_r")

if _EXP1B_DIR in sys.path:
    sys.path.remove(_EXP1B_DIR)
sys.path.insert(0, _EXP1B_DIR)

for p in [_EXP1_DIR, _REPO_ROOT]:
    if p not in sys.path:
        sys.path.append(p)

import importlib.util as _ilu

import config as C
from assign import assign_tokens_to_gt, empty_assignment, merge_assignments
from dataset import ROADWaymoDataset
from model import QwenROADModel

_exp1_train_spec = _ilu.spec_from_file_location("exp1_train", _EXP1_DIR + "/train.py")
_exp1_train = _ilu.module_from_spec(_exp1_train_spec)
_exp1_train_spec.loader.exec_module(_exp1_train)
collate_fn      = _exp1_train.collate_fn
preprocess_clip = _exp1_train.preprocess_clip


# ── FCOS ltrb → box decoding ───────────────────────────────────────────────────

def decode_boxes(
    ltrb:        torch.Tensor,   # [N, 4]  raw predictions
    frame_shape: tuple,          # (H', W')
    device:      torch.device,
) -> torch.Tensor:               # [N, 4]  normalized (x1,y1,x2,y2)
    """
    Decode FCOS (l,t,r,b) distances into (x1,y1,x2,y2) boxes.
    Token centers are computed from the H'×W' grid in normalized [0,1] space.
    """
    H_prime, W_prime = frame_shape
    rows = torch.arange(H_prime, dtype=torch.float32, device=device)
    cols = torch.arange(W_prime, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
    cy = ((grid_y + 0.5) / H_prime).reshape(-1)
    cx = ((grid_x + 0.5) / W_prime).reshape(-1)

    x1 = (cx - ltrb[:, 0]).clamp(0.0, 1.0)
    y1 = (cy - ltrb[:, 1]).clamp(0.0, 1.0)
    x2 = (cx + ltrb[:, 2]).clamp(0.0, 1.0)
    y2 = (cy + ltrb[:, 3]).clamp(0.0, 1.0)

    return torch.stack([x1, y1, x2, y2], dim=1)   # [N, 4]


# ── Per-frame post-processing (NMS) ───────────────────────────────────────────

def postprocess_frame(
    preds:        dict,
    frame_idx:    int,
    frame_offset: int,
    frame_shape:  tuple,
    agentness_thresh: float,
    nms_iou_thresh:   float,
    device:       torch.device,
) -> dict:
    """
    Apply agentness threshold + NMS to a single frame's token predictions.

    Returns:
        dict with keys: boxes [M,4], agentness [M], agent [M,10], action [M,22],
                        loc [M,16], duplex [M,49], triplet [M,86]
        (M = surviving detections after NMS)
    """
    H_prime, W_prime = frame_shape
    n_tokens = H_prime * W_prime
    start = frame_offset
    end   = frame_offset + n_tokens

    agentness = preds["agentness"][start:end, 0]   # [n_tokens]
    fg_mask   = agentness >= agentness_thresh

    if not fg_mask.any():
        empty = lambda n: torch.zeros(0, n, device=device)
        return {
            "boxes":     torch.zeros(0, 4, device=device),
            "agentness": torch.zeros(0,    device=device),
            "agent":     empty(10),
            "action":    empty(22),
            "loc":       empty(16),
            "duplex":    empty(49),
            "triplet":   empty(86),
        }

    ltrb_fg   = preds["box"][start:end][fg_mask]           # [K, 4]
    boxes_fg  = decode_boxes(ltrb_fg, frame_shape, device) # [K, 4]
    scores_fg = agentness[fg_mask]                          # [K]

    # NMS — requires torchvision; fall back to no-NMS if unavailable
    try:
        from torchvision.ops import nms as torchvision_nms
        keep = torchvision_nms(boxes_fg, scores_fg, nms_iou_thresh)
    except ImportError:
        keep = torch.arange(boxes_fg.shape[0], device=device)

    # Build fg index array for gathering other predictions
    fg_indices = torch.where(fg_mask)[0]    # [K]
    kept_local = fg_indices[keep]           # [M]
    kept_global = start + kept_local        # [M] index into full preds

    result = {
        "boxes":     boxes_fg[keep],
        "agentness": scores_fg[keep],
    }
    for head in ("agent", "action", "loc", "duplex", "triplet"):
        result[head] = preds[head][kept_global]
    return result


# ── AP / PRF helpers (unchanged from original eval.py) ────────────────────────

def compute_ap(scores: np.ndarray, labels: np.ndarray) -> float:
    if labels.sum() == 0:
        return float("nan")
    order  = np.argsort(-scores)
    labels = labels[order]
    tp     = np.cumsum(labels)
    fp     = np.cumsum(1 - labels)
    rec    = tp / max(labels.sum(), 1)
    prec   = tp / np.maximum(tp + fp, 1)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p   = prec[rec >= t]
        ap += p.max() if len(p) > 0 else 0.0
    return ap / 101


def compute_prf(preds_bin: np.ndarray, gt: np.ndarray):
    tp   = (preds_bin * gt).sum(0)
    fp   = (preds_bin * (1 - gt)).sum(0)
    fn   = ((1 - preds_bin) * gt).sum(0)
    prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    rec  = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    f1   = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return prec, rec, f1


def build_invalid_pairs(duplex_childs, n_agents=10, n_actions=22):
    valid = set(tuple(p) for p in duplex_childs)
    return [(a, b) for a in range(n_agents) for b in range(n_actions)
            if (a, b) not in valid]


def violation_rate(agent_probs, action_probs, invalid_pairs, threshold=0.5):
    if len(agent_probs) == 0:
        return 0.0
    agent_bin  = agent_probs  >= threshold
    action_bin = action_probs >= threshold
    n_violated = 0
    for a_idx, b_idx in invalid_pairs:
        n_violated += (agent_bin[:, a_idx] & action_bin[:, b_idx]).sum()
    return int(n_violated) / max(len(agent_probs) * len(invalid_pairs), 1)


# ── Main evaluation ────────────────────────────────────────────────────────────

LOG_EVERY = 100


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16

    print(f"Device: {device} | dtype: {dtype}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Agentness threshold: {args.agentness_threshold}")

    print("Loading val dataset …")
    val_ds = ROADWaymoDataset(
        args.anno, args.frames,
        split="val", clip_len=C.CLIP_LEN, stride=C.CLIP_STRIDE,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    print(f"  {len(val_ds):,} val clips")

    with open(args.anno) as f:
        anno_data = json.load(f)
    agent_labels   = anno_data["agent_labels"]
    action_labels  = anno_data["action_labels"]
    loc_labels     = anno_data["loc_labels"]
    duplex_labels  = anno_data["duplex_labels"]
    triplet_labels = anno_data["triplet_labels"]
    duplex_childs  = anno_data["duplex_childs"]
    invalid_pairs  = build_invalid_pairs(duplex_childs)

    print("Loading processor …")
    processor = AutoProcessor.from_pretrained(args.model)

    print("Loading model …")
    model = QwenROADModel(
        model_id   = args.model,
        d_model    = C.VIT_DIM,
        freeze_vit = True,
    )
    model.add_lora(
        r              = C.LORA_R,
        lora_alpha     = C.LORA_ALPHA,
        lora_dropout   = C.LORA_DROPOUT,
        target_modules = C.LORA_TARGET_MODULES,
        n_layers       = C.LORA_N_LAYERS,
    )
    model = model.to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    epoch_loaded = ckpt.get("epoch", "?")
    print(f"  Loaded epoch {epoch_loaded} (best action mAP={ckpt.get('best_map', '?')})")
    model.eval()

    # ── Accumulate predictions across val set ─────────────────────────────────
    all_scores  = defaultdict(list)
    all_gt      = defaultdict(list)
    all_agent_probs  = []
    all_action_probs = []

    total_detections = 0
    total_frames     = 0
    t0       = time.time()
    n_clips  = 0

    with torch.no_grad():
        for pil_frames, frame_targets in val_loader:
            pixel_values, image_grid_thw = preprocess_clip(
                pil_frames, processor, device, dtype
            )

            preds = model(pixel_values, image_grid_thw)
            frame_shapes = preds["frame_shapes"]   # list of T (H', W')

            # For evaluation: use GT assignment to get clean GT labels per token,
            # then compare with model's predictions on agentness-filtered tokens.
            # This measures classification quality on correctly-localized regions.
            assignment = merge_assignments([
                (empty_assignment(h*w, device)
                 if (ft is None or ft["boxes"].shape[0] == 0)
                 else assign_tokens_to_gt(
                     h, w, ft["boxes"],
                     {k: ft[k] for k in ("agent","action","loc","duplex","triplet")},
                     device,
                 ))
                for ft, (h, w) in zip(frame_targets, frame_shapes)
            ])

            is_fg = assignment["is_fg"]
            if is_fg.any():
                for k in ("agent", "action", "loc", "duplex", "triplet"):
                    all_scores[k].append(preds[k][is_fg].float().cpu().numpy())
                    all_gt[k].append(assignment[f"{k}_target"][is_fg].cpu().numpy())

                all_agent_probs.append(preds["agent"][is_fg].float().cpu().numpy())
                all_action_probs.append(preds["action"][is_fg].float().cpu().numpy())

            # Count detections (agentness > threshold, across all frames)
            offset = 0
            for (h, w) in frame_shapes:
                n_tok   = h * w
                agn     = preds["agentness"][offset:offset+n_tok, 0]
                total_detections += (agn >= args.agentness_threshold).sum().item()
                total_frames     += 1
                offset           += n_tok

            n_clips += 1
            if n_clips % LOG_EVERY == 0:
                print(f"  eval clip {n_clips}/{len(val_ds)} | {time.time()-t0:.0f}s elapsed")

    print(f"\nEvaluated {n_clips:,} clips | "
          f"avg {total_detections/max(total_frames,1):.1f} detections/frame "
          f"(agentness > {args.agentness_threshold})")

    if not any(len(v) > 0 for v in all_scores.values()):
        print("ERROR: no foreground tokens collected — check agentness threshold and checkpoint.")
        return {}

    stacked_scores = {k: np.concatenate(v, axis=0) for k, v in all_scores.items()}
    stacked_gt     = {k: np.concatenate(v, axis=0) for k, v in all_gt.items()}
    agent_probs_all  = np.concatenate(all_agent_probs,  axis=0)
    action_probs_all = np.concatenate(all_action_probs, axis=0)
    n_instances = stacked_gt["agent"].shape[0]
    print(f"Total foreground tokens evaluated: {n_instances:,}\n")

    if args.sweep_threshold:
        print("Sweeping agentness threshold for macro-F1 (action head) …")
        best_t, best_f1 = args.agentness_threshold, 0.0
        for t in np.arange(0.1, 0.9, 0.05):
            prec, rec, f1 = compute_prf(
                (stacked_scores["action"] >= t).astype(float),
                stacked_gt["action"],
            )
            has_pos = stacked_gt["action"].sum(0) > 0
            mf1 = f1[has_pos].mean()
            print(f"  t={t:.2f}  macro-F1={mf1:.4f}")
            if mf1 > best_f1:
                best_f1, best_t = mf1, t
        print(f"\nBest threshold: {best_t:.2f} (macro-F1={best_f1:.4f})\n")
        args.threshold = best_t

    threshold = args.threshold

    label_sets = {
        "agent":   agent_labels,
        "action":  action_labels,
        "loc":     loc_labels,
        "duplex":  duplex_labels,
        "triplet": triplet_labels,
    }
    results = {}

    for head_name, label_names in label_sets.items():
        scores = stacked_scores[head_name]
        gt     = stacked_gt[head_name]

        preds_bin = (scores >= threshold).astype(float)
        prec, rec, f1 = compute_prf(preds_bin, gt)
        aps = np.array([compute_ap(scores[:, c], gt[:, c]) for c in range(len(label_names))])

        has_pos       = gt.sum(0) > 0
        n_pos_classes = has_pos.sum()
        macro_p  = prec[has_pos].mean()              if n_pos_classes > 0 else float("nan")
        macro_r  = rec[has_pos].mean()               if n_pos_classes > 0 else float("nan")
        macro_f1 = f1[has_pos].mean()                if n_pos_classes > 0 else float("nan")
        mAP      = float(np.nanmean(aps[has_pos]))   if n_pos_classes > 0 else float("nan")

        per_class = []
        for c, name in enumerate(label_names):
            per_class.append({
                "class":     name,
                "n_gt":      int(gt[:, c].sum()),
                "precision": round(float(prec[c]), 4),
                "recall":    round(float(rec[c]),  4),
                "f1":        round(float(f1[c]),   4),
                "AP":        round(float(aps[c]),  4) if not np.isnan(aps[c]) else None,
            })

        results[head_name] = {
            "macro_precision": round(macro_p,  4),
            "macro_recall":    round(macro_r,  4),
            "macro_f1":        round(macro_f1, 4),
            "mAP":             round(mAP,      4),
            "n_pos_classes":   int(n_pos_classes),
            "n_total_classes": len(label_names),
            "per_class":       per_class,
        }

    viol = violation_rate(agent_probs_all, action_probs_all, invalid_pairs, threshold)

    print("=" * 72)
    print(f"Experiment 1b Evaluation  |  epoch {epoch_loaded}  |  threshold={threshold:.2f}")
    print("=" * 72)
    print(f"{'Head':<10} {'macro-P':>8} {'macro-R':>8} {'macro-F1':>9} {'mAP':>7}")
    print("-" * 50)
    for head_name in ("agent", "action", "loc", "duplex", "triplet"):
        r = results[head_name]
        print(
            f"{head_name:<10} {r['macro_precision']:>8.3f} {r['macro_recall']:>8.3f} "
            f"{r['macro_f1']:>9.3f} {r['mAP']:>7.3f}"
        )
    print("-" * 50)
    print(f"\nConstraint violation rate: {viol:.4f}  ({viol*100:.2f}% of invalid pairs)")

    print("\n--- Per-class F1 (action head, sorted by F1 desc) ---")
    for row in sorted(results["action"]["per_class"], key=lambda x: -x["f1"]):
        if row["n_gt"] > 0:
            print(f"  {row['class']:30s}  F1={row['f1']:.3f}  AP={row['AP']:.3f}  (n_gt={row['n_gt']})")

    print("\n--- Per-class F1 (agent head) ---")
    for row in sorted(results["agent"]["per_class"], key=lambda x: -x["f1"]):
        if row["n_gt"] > 0:
            print(f"  {row['class']:30s}  F1={row['f1']:.3f}  AP={row['AP']:.3f}  (n_gt={row['n_gt']})")

    print("\n--- Reference ---")
    print("  Exp1  (oracle GT boxes, BCE):         agent=35.7  action=22.2  loc=33.6  duplex=12.3  triplet=8.8")
    print("  Exp1b (oracle GT boxes, focal, LoRA): ep1 action mAP=21.4% (training stopped)")
    print("  3D-RetinaNet baseline:                agent=50.3  action=22.9  duplex=13.4  triplet=9.2")

    if args.out:
        out = {
            "checkpoint":  str(args.ckpt),
            "epoch":       epoch_loaded,
            "threshold":   threshold,
            "agentness_threshold": args.agentness_threshold,
            "n_fg_tokens": n_instances,
            "avg_detections_per_frame": round(total_detections / max(total_frames, 1), 2),
            "constraint_violation_rate": round(viol, 6),
            "heads": results,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nFull results saved to: {args.out}")

    return results


# ── Entry point ────────────────────────────────────────────────────────────────

def resolve_ckpt(ckpt_arg: str) -> Path:
    p = Path(ckpt_arg)
    if not p.is_absolute() and p.parent == Path("."):
        p = Path(C.CKPT_DIR) / p
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",     default="best.pt")
    parser.add_argument("--model",    default=C.MODEL_ID)
    parser.add_argument("--anno",     default=C.ANNO_FILE)
    parser.add_argument("--frames",   default=C.FRAMES_DIR)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Sigmoid threshold for binary classification (default 0.5)")
    parser.add_argument("--agentness-threshold", type=float,
                        default=C.AGENTNESS_THRESHOLD, dest="agentness_threshold",
                        help="Agentness threshold for foreground filtering")
    parser.add_argument("--sweep-threshold", action="store_true")
    parser.add_argument("--out",      default=None)
    args = parser.parse_args()

    args.ckpt = resolve_ckpt(args.ckpt)

    if not args.ckpt.exists():
        print(f"ERROR: checkpoint not found: {args.ckpt}")
        avail = list(Path(C.CKPT_DIR).glob("*.pt"))
        if avail:
            print(f"Available: {[p.name for p in sorted(avail)]}")
        else:
            print("No checkpoints found — run training first.")
        sys.exit(1)

    evaluate(args)


if __name__ == "__main__":
    main()

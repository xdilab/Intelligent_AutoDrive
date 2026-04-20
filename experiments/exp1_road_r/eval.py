#!/usr/bin/env python3
"""
Experiment 1 evaluation script — per-class precision, recall, F1, mAP on val split.

Usage
-----
# Evaluate best checkpoint (default):
python -u experiments/exp1_road_r/eval.py

# Evaluate a specific checkpoint:
python -u experiments/exp1_road_r/eval.py --ckpt experiments/exp1_road_r/checkpoints/best.pt

# Sweep a threshold range to find optimal F1:
python -u experiments/exp1_road_r/eval.py --sweep-threshold

# Save full per-class results to JSON:
python -u experiments/exp1_road_r/eval.py --out experiments/exp1_road_r/logs/eval_results.json

Metrics reported
----------------
For each label type (agent / action / loc / duplex / triplet):
  - Per-class: precision, recall, F1 at threshold=0.5
  - Macro-average: mean P, R, F1 across all classes (ignoring classes with 0 GT)
  - AP per class: area under precision-recall curve (trapz)
  - mAP: mean AP across classes with ≥1 positive GT instance

Also reports:
  - T-norm constraint violation rate: fraction of predictions where an invalid
    (agent, action) pair is simultaneously predicted above threshold
  - Comparison note: ECCV24 Track 3 winner reported 69% mAP on ROAD++ activity recognition
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# Force line-buffered stdout so tail -f sees output immediately
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import config as C
from dataset import ROADWaymoDataset
from model import QwenROADModel
from train import collate_fn, preprocess_clip, stack_targets


# ── AP computation ─────────────────────────────────────────────────────────────

def compute_ap(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under precision-recall curve (interpolated, 101-point)."""
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    rec  = tp / max(labels.sum(), 1)
    prec = tp / np.maximum(tp + fp, 1)
    # 101-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = prec[rec >= t]
        ap += p.max() if len(p) > 0 else 0.0
    return ap / 101


def compute_prf(preds_bin: np.ndarray, gt: np.ndarray):
    """Per-class precision, recall, F1 from binary prediction arrays [N, C]."""
    tp = (preds_bin * gt).sum(0)
    fp = (preds_bin * (1 - gt)).sum(0)
    fn = ((1 - preds_bin) * gt).sum(0)
    prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    rec  = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    f1   = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return prec, rec, f1


# ── Constraint violation rate ──────────────────────────────────────────────────

def build_invalid_pairs(duplex_childs: list, n_agents: int = 10, n_actions: int = 22):
    """
    Return (agent_idx, action_idx) pairs that are NOT in duplex_childs.
    duplex_childs: list of [agent_idx, action_idx] valid pairs.
    """
    valid = set(tuple(p) for p in duplex_childs)
    invalid = []
    for a in range(n_agents):
        for b in range(n_actions):
            if (a, b) not in valid:
                invalid.append((a, b))
    return invalid


def violation_rate(
    agent_probs: np.ndarray,   # [N, 10]
    action_probs: np.ndarray,  # [N, 22]
    invalid_pairs: list,
    threshold: float = 0.5,
) -> float:
    """
    Fraction of agent instances where any invalid (agent, action) pair is
    simultaneously predicted above threshold.
    """
    if len(agent_probs) == 0:
        return 0.0
    n_violated = 0
    agent_bin  = agent_probs  >= threshold
    action_bin = action_probs >= threshold
    for a_idx, b_idx in invalid_pairs:
        # any agent instance where both are active
        violated = (agent_bin[:, a_idx] & action_bin[:, b_idx])
        n_violated += violated.sum()
    # normalise: violations per instance (not per pair)
    return int(n_violated) / (len(agent_probs) * len(invalid_pairs))


# ── Main evaluation ────────────────────────────────────────────────────────────

LOG_EVERY = 100   # print progress every N clips


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16

    print(f"Device: {device} | dtype: {dtype}")
    print(f"Checkpoint: {args.ckpt}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("Loading val dataset …")
    val_ds = ROADWaymoDataset(
        args.anno, args.frames,
        split="val", clip_len=C.CLIP_LEN, stride=C.CLIP_STRIDE,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    print(f"  {len(val_ds):,} val clips")

    # ── Annotation metadata ───────────────────────────────────────────────────
    with open(args.anno) as f:
        anno_data = json.load(f)
    agent_labels   = anno_data["agent_labels"]    # 10 names
    action_labels  = anno_data["action_labels"]   # 22 names
    loc_labels     = anno_data["loc_labels"]      # 16 names
    duplex_labels  = anno_data["duplex_labels"]   # 49 names
    triplet_labels = anno_data["triplet_labels"]  # 86 names
    duplex_childs  = anno_data["duplex_childs"]

    invalid_pairs = build_invalid_pairs(duplex_childs)

    # ── Processor ────────────────────────────────────────────────────────────
    print(f"Loading processor …")
    processor = AutoProcessor.from_pretrained(args.model)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Loading model …")
    model = QwenROADModel(
        model_id   = args.model,
        d_model    = C.VIT_DIM,
        freeze_vit = True,
        tube_heads = C.TUBE_N_HEADS,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    epoch_loaded = ckpt.get("epoch", "?")
    print(f"  Loaded epoch {epoch_loaded} checkpoint.")

    model.eval()

    # ── Accumulate predictions and GT ─────────────────────────────────────────
    all_scores  = defaultdict(list)   # key → list of [n_agents, n_class] arrays
    all_gt      = defaultdict(list)

    all_agent_probs  = []
    all_action_probs = []

    t0 = time.time()
    n_clips = 0
    n_skipped = 0

    with torch.no_grad():
        for pil_frames, frame_targets in val_loader:
            pixel_values, image_grid_thw = preprocess_clip(
                pil_frames, processor, device, dtype
            )
            targets, boxes_per_frame = stack_targets(frame_targets, device)
            if targets is None:
                n_skipped += 1
                continue

            preds = model(pixel_values, image_grid_thw, boxes_per_frame)
            if preds is None:
                n_skipped += 1
                continue

            for k in ("agent", "action", "loc", "duplex", "triplet"):
                all_scores[k].append(preds[k].float().cpu().numpy())
                all_gt[k].append(targets[k].float().cpu().numpy())

            all_agent_probs.append(preds["agent"].float().cpu().numpy())
            all_action_probs.append(preds["action"].float().cpu().numpy())

            n_clips += 1
            if n_clips % LOG_EVERY == 0:
                elapsed = time.time() - t0
                print(f"  eval clip {n_clips}/{len(val_ds)} | {elapsed:.0f}s elapsed")

    print(f"\nEvaluated {n_clips:,} clips ({n_skipped} skipped, no annotations)")

    # ── Stack across all clips ────────────────────────────────────────────────
    stacked_scores = {k: np.concatenate(v, axis=0) for k, v in all_scores.items()}
    stacked_gt     = {k: np.concatenate(v, axis=0) for k, v in all_gt.items()}

    agent_probs_all  = np.concatenate(all_agent_probs, axis=0)
    action_probs_all = np.concatenate(all_action_probs, axis=0)

    n_instances = stacked_gt["agent"].shape[0]
    print(f"Total agent instances evaluated: {n_instances:,}\n")

    # ── Sweep threshold if requested ──────────────────────────────────────────
    if args.sweep_threshold:
        print("Sweeping threshold for macro-F1 (action head) …")
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.1, 0.9, 0.05):
            prec, rec, f1 = compute_prf(
                (stacked_scores["action"] >= t).astype(float),
                stacked_gt["action"],
            )
            # macro over classes with ≥1 GT positive
            has_pos = stacked_gt["action"].sum(0) > 0
            mf1 = f1[has_pos].mean()
            print(f"  t={t:.2f}  macro-F1={mf1:.4f}")
            if mf1 > best_f1:
                best_f1, best_t = mf1, t
        print(f"\nBest threshold: {best_t:.2f} (macro-F1={best_f1:.4f})\n")
        args.threshold = best_t

    threshold = args.threshold

    # ── Per-head metrics ──────────────────────────────────────────────────────
    label_sets = {
        "agent":   agent_labels,
        "action":  action_labels,
        "loc":     loc_labels,
        "duplex":  duplex_labels,
        "triplet": triplet_labels,
    }

    results = {}
    summary_rows = []

    for head_name, label_names in label_sets.items():
        scores = stacked_scores[head_name]   # [N, C]
        gt     = stacked_gt[head_name]       # [N, C]

        preds_bin = (scores >= threshold).astype(float)
        prec, rec, f1 = compute_prf(preds_bin, gt)

        aps = np.array([compute_ap(scores[:, c], gt[:, c]) for c in range(len(label_names))])

        # Only include classes with at least 1 GT positive instance
        has_pos = gt.sum(0) > 0
        n_pos_classes = has_pos.sum()

        macro_p   = prec[has_pos].mean()  if n_pos_classes > 0 else float("nan")
        macro_r   = rec[has_pos].mean()   if n_pos_classes > 0 else float("nan")
        macro_f1  = f1[has_pos].mean()    if n_pos_classes > 0 else float("nan")
        mAP       = float(np.nanmean(aps[has_pos])) if n_pos_classes > 0 else float("nan")

        per_class = []
        for c, name in enumerate(label_names):
            n_gt = int(gt[:, c].sum())
            per_class.append({
                "class":     name,
                "n_gt":      n_gt,
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

        summary_rows.append(
            f"  {head_name:8s}  macro-P={macro_p:.3f}  macro-R={macro_r:.3f}  "
            f"macro-F1={macro_f1:.3f}  mAP={mAP:.3f}  "
            f"({n_pos_classes}/{len(label_names)} classes have GT)"
        )

    # ── Constraint violation rate ─────────────────────────────────────────────
    viol = violation_rate(agent_probs_all, action_probs_all, invalid_pairs, threshold)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("=" * 72)
    print(f"Experiment 1 Evaluation  |  epoch {epoch_loaded}  |  threshold={threshold:.2f}")
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
    print(f"\nConstraint violation rate: {viol:.4f}  ({viol*100:.2f}% of invalid pairs co-predicted)")

    print("\n--- Per-class F1 (action head, sorted by F1 desc) ---")
    action_per_class = sorted(
        results["action"]["per_class"],
        key=lambda x: x["f1"], reverse=True,
    )
    for row in action_per_class:
        if row["n_gt"] > 0:
            print(f"  {row['class']:30s}  F1={row['f1']:.3f}  AP={row['AP']:.3f}  (n_gt={row['n_gt']})")

    print("\n--- Per-class F1 (agent head) ---")
    for row in sorted(results["agent"]["per_class"], key=lambda x: -x["f1"]):
        if row["n_gt"] > 0:
            print(f"  {row['class']:30s}  F1={row['f1']:.3f}  AP={row['AP']:.3f}  (n_gt={row['n_gt']})")

    print("\n--- Reference ---")
    print("  ECCV24 Track 3 winner (activity recognition, ROAD++): 69% mAP")
    print("  ECCV24 Track 1 winner (spatiotemporal detection, ROAD++): 30.82% video-mAP")
    print("  Note: above are detection-based; our Exp 1 uses GT boxes — not directly comparable.")
    print("  Fair comparison: Exp 1b (with detection head) video-mAP vs above.")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if args.out:
        out = {
            "checkpoint": str(args.ckpt),
            "epoch":      epoch_loaded,
            "threshold":  threshold,
            "n_instances": n_instances,
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
    """
    Resolve checkpoint path with two shortcuts:
      - bare filename (e.g. 'latest.pt', 'best.pt') → CKPT_DIR / filename
      - full path → used as-is
    """
    p = Path(ckpt_arg)
    if not p.is_absolute() and p.parent == Path("."):
        # bare filename — resolve relative to CKPT_DIR
        p = Path(C.CKPT_DIR) / p
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",     default="latest.pt",
                        help="Checkpoint to evaluate. Bare filename (e.g. 'latest.pt', "
                             "'best.pt') resolves to CKPT_DIR/filename. "
                             "Default: latest.pt")
    parser.add_argument("--model",    default=C.MODEL_ID)
    parser.add_argument("--anno",     default=C.ANNO_FILE)
    parser.add_argument("--frames",   default=C.FRAMES_DIR)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Sigmoid threshold for binary prediction (default 0.5)")
    parser.add_argument("--sweep-threshold", action="store_true",
                        help="Sweep thresholds 0.1–0.85 to find best macro-F1 on action head")
    parser.add_argument("--out",      default=None,
                        help="Path to save full JSON results")
    args = parser.parse_args()

    args.ckpt = resolve_ckpt(args.ckpt)

    if not args.ckpt.exists():
        print(f"ERROR: checkpoint not found: {args.ckpt}")
        avail = list(Path(C.CKPT_DIR).glob("*.pt"))
        if avail:
            print(f"Available checkpoints in {C.CKPT_DIR}:")
            for p in sorted(avail):
                print(f"  {p.name}")
        else:
            print("No checkpoints found — run training first.")
        sys.exit(1)

    evaluate(args)


if __name__ == "__main__":
    main()

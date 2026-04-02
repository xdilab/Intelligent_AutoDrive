#!/usr/bin/env python3
"""
Evaluate saved SmolVLM predictions against ROAD-Waymo ground truth.

Handles both output formats:
  - Zero-shot (smolvlm_inference.py):    detections[].agent / .action / .location
  - Constrained (smolvlm_constrained.py): detections[].triplet  ("Ped-Stop-Jun")

Metrics (label-set level, not box-level — no bbox IoU here):
  - parse_rate:       fraction of frames where output parsed as valid JSON
  - agent/action/loc: precision, recall, F1 against GT label sets
  - violation_rate:   fraction of predicted triplets not in the valid constraint set
                      (constrained runs only)

Usage:
    python baseline/eval_preds.py
    python baseline/eval_preds.py --preds baseline/results/smolvlm_preds.json
    python baseline/eval_preds.py --preds baseline/results/constrained_preds.json
"""

import argparse
import json
from collections import defaultdict

DEFAULT_PREDS = "/data/repos/ROAD_Reason/baseline/results/smolvlm_preds.json"


def label_sets_from_gt(gt_detections):
    """Return (agents, actions, locations) as sets of strings from GT list."""
    agents, actions, locs = set(), set(), set()
    for det in gt_detections:
        agents.update(det.get("agent", []))
        actions.update(det.get("actions", []))
        locs.update(det.get("locations", []))
    return agents, actions, locs


def label_sets_from_pred(parsed):
    """Return (agents, actions, locations) as sets of strings from model output.

    Handles both formats:
      - Zero-shot:    {"agent": "Ped", "action": ["Stop"], "location": "Jun"}
      - Constrained:  {"triplet": "Ped-Stop-Jun"}
    """
    if not parsed:
        return set(), set(), set()
    agents, actions, locs = set(), set(), set()
    for det in parsed.get("detections", []):
        # Constrained format: triplet string "Agent-Action-Location"
        triplet = det.get("triplet", "")
        if triplet:
            parts = triplet.split("-", 2)
            if len(parts) == 3:
                agents.add(parts[0])
                actions.add(parts[1])
                locs.add(parts[2])
            continue

        # Zero-shot format: separate fields
        a = det.get("agent", "")
        if isinstance(a, str) and a:
            agents.add(a)
        elif isinstance(a, list):
            agents.update(a)
        act = det.get("action", [])
        if isinstance(act, str):
            act = [act]
        actions.update(act)
        loc = det.get("location", "")
        if isinstance(loc, str) and loc:
            locs.add(loc)
        elif isinstance(loc, list):
            locs.update(loc)
    return agents, actions, locs


def precision_recall(pred_set, gt_set):
    if not pred_set and not gt_set:
        return 1.0, 1.0
    recall    = len(pred_set & gt_set) / len(gt_set)    if gt_set    else 0.0
    precision = len(pred_set & gt_set) / len(pred_set)  if pred_set  else 0.0
    return precision, recall


def evaluate(preds_path):
    with open(preds_path) as f:
        data = json.load(f)

    meta    = data["meta"]
    results = data["results"]

    print(f"Model:  {meta['model']}")
    print(f"Split:  {meta['split']}")
    print(f"Frames: {meta['total_frames']}  (parse ok: {meta['parse_ok']})")
    print()

    totals = defaultdict(float)
    n_parsed = 0

    for r in results:
        gt_agents, gt_actions, gt_locs = label_sets_from_gt(r["gt"])
        pred_agents, pred_actions, pred_locs = label_sets_from_pred(r["parsed"])

        ap, ar = precision_recall(pred_agents, gt_agents)
        actp, actr = precision_recall(pred_actions, gt_actions)
        lp, lr = precision_recall(pred_locs, gt_locs)

        totals["agent_p"]  += ap
        totals["agent_r"]  += ar
        totals["action_p"] += actp
        totals["action_r"] += actr
        totals["loc_p"]    += lp
        totals["loc_r"]    += lr
        n_parsed += 1

    n = max(n_parsed, 1)
    parse_rate = meta["parse_ok"] / meta["total_frames"] if meta["total_frames"] else 0

    def f1(p, r):
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    ap = totals["agent_p"]  / n
    ar = totals["agent_r"]  / n
    cp = totals["action_p"] / n
    cr = totals["action_r"] / n
    lp = totals["loc_p"]    / n
    lr = totals["loc_r"]    / n

    print(f"{'Metric':<22} {'Prec':>6}  {'Recall':>6}  {'F1':>6}")
    print("-" * 46)
    print(f"{'parse_rate':<22} {'':>6}  {'':>6}  {parse_rate:>6.3f}")
    print(f"{'agent':<22} {ap:>6.3f}  {ar:>6.3f}  {f1(ap,ar):>6.3f}")
    print(f"{'action':<22} {cp:>6.3f}  {cr:>6.3f}  {f1(cp,cr):>6.3f}")
    print(f"{'location':<22} {lp:>6.3f}  {lr:>6.3f}  {f1(lp,lr):>6.3f}")

    # Constraint violation rate (constrained runs only)
    viol_rates = [r["violation_rate"] for r in results if r.get("violation_rate") is not None]
    if viol_rates:
        mean_viol = sum(viol_rates) / len(viol_rates)
        print(f"\n{'constraint_violation':<22} {mean_viol:>6.3f}  (mean across {len(viol_rates)} frames)")
        print(f"  (from meta): {meta.get('mean_constraint_violation_rate', 'n/a')}")

    # Per-frame breakdown for errors
    print("\n--- Sample failures ---")
    fails = [r for r in results if r["parsed"] is None][:5]
    for r in fails:
        print(f"  {r['video']} f{r['frame_id']}: {r['parse_err']}")
        print(f"    raw: {r['raw'][:120]!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", default=DEFAULT_PREDS)
    args = parser.parse_args()
    evaluate(args.preds)


if __name__ == "__main__":
    main()

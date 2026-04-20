#!/usr/bin/env python3
"""
Evaluate saved Qwen native grounding predictions against ROAD-Waymo GT.

This evaluator is intentionally lightweight. It is not the official baseline
frame-mAP protocol. Instead, it gives a useful early readout on:

- parse success
- label-set quality (agent/action/location)
- box quality via simple greedy IoU matching
- matched-label quality on grounded detections

Expected input:
  experiments/exp1c_qwen_native_grounding/results/qwen_native_grounding.json
"""

import argparse
import json
from pathlib import Path


DEFAULT_PREDS = "/data/repos/ROAD_Reason/experiments/exp1c_qwen_native_grounding/results/qwen_native_grounding.json"


def iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    iw = max(inter_x2 - inter_x1, 0.0)
    ih = max(inter_y2 - inter_y1, 0.0)
    inter = iw * ih
    area_a = max(xa2 - xa1, 0.0) * max(ya2 - ya1, 0.0)
    area_b = max(xb2 - xb1, 0.0) * max(yb2 - yb1, 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def label_sets_from_gt(gt_detections):
    agents, actions, locs = set(), set(), set()
    for det in gt_detections:
        agents.update(det.get("agent", []))
        actions.update(det.get("actions", []))
        locs.update(det.get("locations", []))
    return agents, actions, locs


def label_sets_from_pred(parsed):
    if not parsed:
        return set(), set(), set()
    agents, actions, locs = set(), set(), set()
    for det in parsed.get("detections", []):
        agent = det.get("agent", "")
        if isinstance(agent, str) and agent:
            agents.add(agent)
        elif isinstance(agent, list):
            agents.update(agent)

        action = det.get("action", [])
        if isinstance(action, str):
            action = [action]
        actions.update(a for a in action if isinstance(a, str) and a)

        loc = det.get("location", "")
        if isinstance(loc, str) and loc:
            locs.add(loc)
        elif isinstance(loc, list):
            locs.update(loc)
    return agents, actions, locs


def precision_recall(pred_set, gt_set):
    if not pred_set and not gt_set:
        return 1.0, 1.0
    recall = len(pred_set & gt_set) / len(gt_set) if gt_set else 0.0
    precision = len(pred_set & gt_set) / len(pred_set) if pred_set else 0.0
    return precision, recall


def pred_boxes_norm(parsed, width, height):
    if not parsed:
        return []
    boxes = []
    for det in parsed.get("detections", []):
        box = det.get("bbox_2d")
        if not isinstance(box, list) or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in box]
        except Exception:
            continue
        if width <= 0 or height <= 0:
            continue
        boxes.append(
            {
                "bbox": [x1 / width, y1 / height, x2 / width, y2 / height],
                "agent": det.get("agent", ""),
                "action": det.get("action", []),
                "location": det.get("location", ""),
                "confidence": det.get("confidence", ""),
            }
        )
    return boxes


def greedy_match(preds, gts, iou_thresh=0.5):
    """
    Greedy one-to-one IoU matching.
    """
    matches = []
    used_gt = set()

    scored = []
    for pi, pred in enumerate(preds):
        for gi, gt in enumerate(gts):
            score = iou(pred["bbox"], gt["bbox_2d_norm"])
            if score >= iou_thresh:
                scored.append((score, pi, gi))
    scored.sort(reverse=True)

    used_pred = set()
    for score, pi, gi in scored:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        matches.append((pi, gi, score))

    return matches, used_pred, used_gt


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def evaluate(preds_path, iou_thresh):
    with open(preds_path) as f:
        data = json.load(f)

    meta = data["meta"]
    results = data["results"]

    print(f"Model:  {meta['model']}")
    print(f"Split:  {meta['split']}")
    print(f"Frames: {meta['total_frames']}  (parse ok: {meta['parse_ok']})")
    print(f"IoU threshold: {iou_thresh}")
    print()

    totals = {
        "agent_p": 0.0, "agent_r": 0.0,
        "action_p": 0.0, "action_r": 0.0,
        "loc_p": 0.0, "loc_r": 0.0,
        "matched_agent_correct": 0,
        "matched_location_correct": 0,
        "matched_action_jaccard_sum": 0.0,
        "matched_count": 0,
        "pred_boxes": 0,
        "gt_boxes": 0,
        "matched_preds": 0,
        "matched_gts": 0,
    }

    for r in results:
        gt_agents, gt_actions, gt_locs = label_sets_from_gt(r["gt"])
        pred_agents, pred_actions, pred_locs = label_sets_from_pred(r["parsed"])

        ap, ar = precision_recall(pred_agents, gt_agents)
        cp, cr = precision_recall(pred_actions, gt_actions)
        lp, lr = precision_recall(pred_locs, gt_locs)

        totals["agent_p"] += ap
        totals["agent_r"] += ar
        totals["action_p"] += cp
        totals["action_r"] += cr
        totals["loc_p"] += lp
        totals["loc_r"] += lr

        width = height = 0
        if r.get("img_path") and Path(r["img_path"]).exists():
            from PIL import Image
            with Image.open(r["img_path"]) as img:
                width, height = img.size

        preds = pred_boxes_norm(r["parsed"], width, height)
        gts = r["gt"]

        totals["pred_boxes"] += len(preds)
        totals["gt_boxes"] += len(gts)

        matches, used_pred, used_gt = greedy_match(preds, gts, iou_thresh=iou_thresh)
        totals["matched_preds"] += len(used_pred)
        totals["matched_gts"] += len(used_gt)
        totals["matched_count"] += len(matches)

        for pi, gi, _ in matches:
            pred = preds[pi]
            gt = gts[gi]

            gt_agents = set(gt.get("agent", []))
            gt_locs = set(gt.get("locations", []))
            gt_actions = set(gt.get("actions", []))

            if pred.get("agent", "") in gt_agents:
                totals["matched_agent_correct"] += 1
            if pred.get("location", "") in gt_locs:
                totals["matched_location_correct"] += 1

            pred_actions = pred.get("action", [])
            if isinstance(pred_actions, str):
                pred_actions = [pred_actions]
            pred_actions = set(a for a in pred_actions if isinstance(a, str) and a)
            union = len(pred_actions | gt_actions)
            inter = len(pred_actions & gt_actions)
            totals["matched_action_jaccard_sum"] += (inter / union) if union > 0 else 1.0

    n = max(len(results), 1)
    parse_rate = meta["parse_ok"] / meta["total_frames"] if meta["total_frames"] else 0.0

    agent_p = totals["agent_p"] / n
    agent_r = totals["agent_r"] / n
    action_p = totals["action_p"] / n
    action_r = totals["action_r"] / n
    loc_p = totals["loc_p"] / n
    loc_r = totals["loc_r"] / n

    box_precision = totals["matched_preds"] / totals["pred_boxes"] if totals["pred_boxes"] else 0.0
    box_recall = totals["matched_gts"] / totals["gt_boxes"] if totals["gt_boxes"] else 0.0
    box_f1 = f1(box_precision, box_recall)

    matched_n = max(totals["matched_count"], 1)
    matched_agent_acc = totals["matched_agent_correct"] / matched_n
    matched_loc_acc = totals["matched_location_correct"] / matched_n
    matched_action_jaccard = totals["matched_action_jaccard_sum"] / matched_n

    print(f"{'Metric':<28} {'Prec':>7}  {'Recall':>7}  {'F1':>7}")
    print("-" * 58)
    print(f"{'parse_rate':<28} {'':>7}  {'':>7}  {parse_rate:>7.3f}")
    print(f"{'agent label set':<28} {agent_p:>7.3f}  {agent_r:>7.3f}  {f1(agent_p, agent_r):>7.3f}")
    print(f"{'action label set':<28} {action_p:>7.3f}  {action_r:>7.3f}  {f1(action_p, action_r):>7.3f}")
    print(f"{'location label set':<28} {loc_p:>7.3f}  {loc_r:>7.3f}  {f1(loc_p, loc_r):>7.3f}")
    print(f"{'box match @ IoU':<28} {box_precision:>7.3f}  {box_recall:>7.3f}  {box_f1:>7.3f}")

    print("\nMatched-detection label quality")
    print(f"  agent accuracy:            {matched_agent_acc:.3f}")
    print(f"  location accuracy:         {matched_loc_acc:.3f}")
    print(f"  action set Jaccard:        {matched_action_jaccard:.3f}")
    print(f"  matched detections:        {totals['matched_count']}")
    print(f"  predicted boxes:           {totals['pred_boxes']}")
    print(f"  gt boxes:                  {totals['gt_boxes']}")

    print("\n--- Sample parse failures ---")
    fails = [r for r in results if r["parsed"] is None][:5]
    for r in fails:
        print(f"  {r['video']} f{r['frame_id']}: {r['parse_err']}")
        print(f"    raw: {r['raw'][:120]!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", default=DEFAULT_PREDS)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    evaluate(args.preds, args.iou)


if __name__ == "__main__":
    main()

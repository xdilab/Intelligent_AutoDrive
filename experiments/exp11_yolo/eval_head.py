"""Exp11 RoIAlign hybrid eval: YOLO boxes + trained head on pooled I3D feats.

Row scoring mirrors the score-transfer eval exactly (agentness/agent from
YOLO, action/loc/duplex/triplet from the head) so the two rows are directly
comparable; only the score source changes (IoU-copy -> pooled-feature head).

Usage: python -u eval_head.py --head head_roialign_lam0.pt --out results.json
"""
import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "/data/repos/PedestrianIntent++/ROAD_plus_plus_Baseline")
import modules.evaluation as baseline_eval
from modules.utils import get_individual_labels

E = Path("/data/repos/ROAD_Reason/experiments/exp11_yolo")
BOX_W, BOX_H = 840, 600
_HEADS = ("agent", "action", "loc", "duplex", "triplet")
_LABEL_TYPES = ("agentness", "agent", "action", "loc", "duplex", "triplet")

ap = argparse.ArgumentParser()
ap.add_argument("--head", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--gate", action="store_true", help="multiply head sigmoids by YOLO conf")
args = ap.parse_args()

print("[eval] loading ...", flush=True)
head = nn.Linear(256, 184)
head.load_state_dict(torch.load(args.head, weights_only=True)["state"])
head.eval()
feats = pickle.load(open(E / "roi_feats_i3d_val.pkl", "rb"))["feats"]
yolo = pickle.load(open(E / "dets_v8x_best_val_fullcand.pkl", "rb"))["records"]
yolo = {k.rsplit("_", 1)[0] + "/" + str(int(k.rsplit("_", 1)[1])): v for k, v in yolo.items()}
i3d_payload = pickle.load(open(E / "dets_i3d_val_fullcand.pkl", "rb"))
i3d = i3d_payload["records"]; labels = i3d_payload["labels"]
all_classes = [["agentness"], labels["agent"], labels["action"],
               labels["loc"], labels["duplex"], labels["triplet"]]
num_c = [len(c) for c in all_classes]
offsets = np.cumsum([0] + num_c); nT = len(num_c)

gt_all = [[] for _ in range(nT)]
det_all = [[[] for _ in range(num_c[t])] for t in range(nT)]
n_frames = 0; t0 = time.time()
with torch.no_grad():
    for key in sorted(yolo):
        if key not in i3d or key not in feats:
            continue
        yrec = yolo[key]
        yb = yrec["boxes_xyxyn"].astype(np.float32)
        f = feats[key]
        if f.shape[0] != yb.shape[0]:      # row mismatch safeguard
            continue
        sig = torch.sigmoid(head(torch.from_numpy(f).float())).numpy() if f.shape[0] \
            else np.zeros((0, 184), np.float32)
        if args.gate and f.shape[0]:
            sig = sig * yrec["conf"].astype(np.float32)[:, None]
        gt = i3d[key]["gt"]
        if gt is None or gt["boxes"].shape[0] == 0:
            gt_boxes = np.zeros((0, 4), np.float32)
            mh = {h: np.zeros((0, num_c[i + 1]), np.float32) for i, h in enumerate(_HEADS)}
        else:
            gt_boxes = gt["boxes"].astype(np.float32) / np.array([BOX_W, BOX_H, BOX_W, BOX_H], np.float32)
            mh = {h: gt[h].astype(np.float32) for h in _HEADS}
        for t in range(nT):
            if t == 0:
                g = np.zeros((gt_boxes.shape[0], 5), np.float32); g[:, :4] = gt_boxes
            else:
                g = get_individual_labels(gt_boxes, mh[_HEADS[t - 1]]).astype(np.float32)
            gt_all[t].append(g)
        yconf = yrec["conf"].astype(np.float32); ycls = yrec["cls"].astype(int)
        det_all[0][0].append(np.concatenate([yb, yconf[:, None]], 1))
        for c in range(num_c[1]):
            m = ycls == c
            det_all[1][c].append(np.concatenate([yb[m], yconf[m, None]], 1))
        for t in range(2, nT):
            for c in range(num_c[t]):
                col = offsets[t] + c
                det_all[t][c].append(np.concatenate([yb, sig[:, col:col + 1]], 1))
        n_frames += 1
        if n_frames % 10000 == 0:
            print(f"[eval] {n_frames:,} frames  {time.time()-t0:.0f}s", flush=True)

print(f"[eval] {n_frames:,} frames | running baseline f-mAP ...", flush=True)
mAP, _, ap_strs = baseline_eval.evaluate(gt_all, det_all, all_classes, iou_thresh=0.5)
summary = {}
print(f"\n=== RoIAlign hybrid f-mAP ({args.head}) ===")
for t in range(nT):
    summary[_LABEL_TYPES[t]] = float(mAP[t])
    print(f"  {_LABEL_TYPES[t]:>10s}: {mAP[t]:.4f}%")
Path(args.out).write_text(json.dumps(
    {"head": args.head, "summary": summary, "n_frames": n_frames,
     "per_class": {_LABEL_TYPES[t]: ap_strs[t] for t in range(nT)},
     "rows": "yolo_v8x_best_ep1_fullcand",
     "scores": "roialign_head" + ("_confgated" if args.gate else "")}, indent=2))
print(f"[eval] wrote {args.out}", flush=True)

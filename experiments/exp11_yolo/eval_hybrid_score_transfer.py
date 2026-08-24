"""Exp11 hybrid control (score transfer): YOLO boxes + I3D scores, no training.

Rows = YOLOv8x best.pt full-candidate boxes. Agent scores from YOLO itself.
Action/loc/duplex/triplet (and agentness) transferred from the best-IoU I3D
candidate (IoU >= 0.5); YOLO boxes with no I3D match get zero scores on the
transferred heads (reported as match-rate). GT and protocol identical to the
exp9 evaluator path; baseline evaluate() at IoU=0.5 over the FULL val split.

Comparators: baseline all-anchor (agent 17.76); NOT the top-40 control family.
Usage: python -u eval_hybrid_score_transfer.py [--out results.json]
"""
import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "/data/repos/PedestrianIntent++/ROAD_plus_plus_Baseline")
import modules.evaluation as baseline_eval
from modules.utils import get_individual_labels

E = Path("/data/repos/ROAD_Reason/experiments/exp11_yolo")
YOLO_PKL = E / "dets_v8x_best_val_fullcand.pkl"
I3D_PKL = E / "dets_i3d_val_fullcand.pkl"
BOX_W, BOX_H = 840, 600
IOU_MATCH = 0.5
_HEADS = ("agent", "action", "loc", "duplex", "triplet")
_LABEL_TYPES = ("agentness", "agent", "action", "loc", "duplex", "triplet")

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(E / "results_hybrid_score_transfer.json"))
args = ap.parse_args()

def iou_matrix(a, b):
    """a [n,4], b [m,4] normalized xyxy -> [n,m] IoU."""
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[None, :, 0], b[None, :, 1], b[None, :, 2], b[None, :, 3]
    ix = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    iy = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = ix * iy
    aa = (ax2 - ax1) * (ay2 - ay1)
    bb = (bx2 - bx1) * (by2 - by1)
    return inter / np.clip(aa + bb - inter, 1e-9, None)

print("[hybrid] loading dumps ...", flush=True)
yolo = pickle.load(open(YOLO_PKL, "rb"))["records"]
i3d_payload = pickle.load(open(I3D_PKL, "rb"))
i3d = i3d_payload["records"]
labels = i3d_payload["labels"]
all_classes = [["agentness"], labels["agent"], labels["action"],
               labels["loc"], labels["duplex"], labels["triplet"]]
num_c = [len(c) for c in all_classes]
offsets = np.cumsum([0] + num_c)   # 184 layout offsets
nT = len(num_c)

# YOLO stems are "<video>_<fid:05d>"; I3D keys are "<video>/<fid>"
def to_i3d_key(stem):
    v, f = stem.rsplit("_", 1)
    return f"{v}/{int(f)}"

gt_all = [[] for _ in range(nT)]
det_all = [[[] for _ in range(num_c[t])] for t in range(nT)]
n_yolo = n_matched = n_frames = 0
t0 = time.time()
for stem in sorted(yolo):
    k = to_i3d_key(stem)
    if k not in i3d:
        continue
    yrec, irec = yolo[stem], i3d[k]
    yb = yrec["boxes_xyxyn"].astype(np.float32)          # [n,4] normalized
    ib = irec["boxes"].astype(np.float32) / np.array([BOX_W, BOX_H, BOX_W, BOX_H], np.float32)
    isig = 1 / (1 + np.exp(-irec["logits"].astype(np.float32)))   # [m,184]
    n = yb.shape[0]
    trans = np.zeros((n, offsets[-1]), np.float32)
    if n and ib.shape[0]:
        M = iou_matrix(yb, np.clip(ib, 0, 1))
        best = M.argmax(1); bv = M[np.arange(n), best]
        hit = bv >= IOU_MATCH
        trans[hit] = isig[best[hit]]
        n_matched += int(hit.sum())
    n_yolo += n; n_frames += 1

    gt = irec["gt"]
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
    # dets: agentness = YOLO conf; agent = YOLO cls/conf; rest = transferred I3D sigmoids
    yconf = yrec["conf"].astype(np.float32); ycls = yrec["cls"].astype(int)
    det_all[0][0].append(np.concatenate([yb, yconf[:, None]], 1))
    for c in range(num_c[1]):
        m = ycls == c
        det_all[1][c].append(np.concatenate([yb[m], yconf[m, None]], 1))
    for t in range(2, nT):
        for c in range(num_c[t]):
            col = offsets[t] + c
            det_all[t][c].append(np.concatenate([yb, trans[:, col:col + 1]], 1))
    if n_frames % 5000 == 0:
        print(f"[hybrid] {n_frames:,} frames  {time.time()-t0:.0f}s", flush=True)

match_rate = n_matched / max(n_yolo, 1)
print(f"[hybrid] {n_frames:,} frames | {n_yolo:,} YOLO boxes | "
      f"match-rate {100*match_rate:.1f}% @IoU>={IOU_MATCH}", flush=True)
print("[hybrid] running baseline f-mAP ...", flush=True)
mAP, _, ap_strs = baseline_eval.evaluate(gt_all, det_all, all_classes, iou_thresh=0.5)
summary = {}
print("\n=== Hybrid score-transfer f-mAP (full val, full-candidate) ===")
for t in range(nT):
    summary[_LABEL_TYPES[t]] = float(mAP[t])
    print(f"  {_LABEL_TYPES[t]:>10s}: {mAP[t]:.4f}%")
Path(args.out).write_text(json.dumps(
    {"summary": summary, "match_rate": match_rate, "n_frames": n_frames,
     "n_yolo_boxes": n_yolo, "iou_match": IOU_MATCH,
     "per_class": {_LABEL_TYPES[t]: ap_strs[t] for t in range(nT)},
     "rows": "yolo_v8x_best_ep1_fullcand", "scores": "i3d_transfer"}, indent=2))
print(f"[hybrid] wrote {args.out}", flush=True)

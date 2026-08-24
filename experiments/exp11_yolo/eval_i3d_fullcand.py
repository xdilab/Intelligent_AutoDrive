"""Score the frozen 3D-RetinaNet's OWN full-candidate detections on the exp11
frame set — the exact-same-rows baseline for the exp11 section."""
import json, pickle, sys, time
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "/data/repos/PedestrianIntent++/ROAD_plus_plus_Baseline")
import modules.evaluation as baseline_eval
from modules.utils import get_individual_labels

E = Path("/data/repos/ROAD_Reason/experiments/exp11_yolo")
BOX_W, BOX_H = 840, 600
_HEADS = ("agent", "action", "loc", "duplex", "triplet")
_LABEL_TYPES = ("agentness", "agent", "action", "loc", "duplex", "triplet")

payload = pickle.load(open(E / "dets_i3d_val_fullcand.pkl", "rb"))
i3d = payload["records"]; labels = payload["labels"]
# restrict to the same frames the hybrid rows used (YOLO∩I3D)
yolo = pickle.load(open(E / "dets_v8x_best_val_fullcand.pkl", "rb"))["records"]
keys = sorted(set(k.rsplit("_",1)[0]+"/"+str(int(k.rsplit("_",1)[1])) for k in yolo) & set(i3d))
all_classes = [["agentness"], labels["agent"], labels["action"],
               labels["loc"], labels["duplex"], labels["triplet"]]
num_c = [len(c) for c in all_classes]
offsets = np.cumsum([0] + num_c); nT = len(num_c)
gt_all = [[] for _ in range(nT)]
det_all = [[[] for _ in range(num_c[t])] for t in range(nT)]
t0 = time.time()
for n, key in enumerate(keys):
    rec = i3d[key]
    ib = np.clip(rec["boxes"].astype(np.float32) /
                 np.array([BOX_W, BOX_H, BOX_W, BOX_H], np.float32), 0, 1)
    sig = 1 / (1 + np.exp(-rec["logits"].astype(np.float32)))
    sc = rec["scores"].astype(np.float32)
    gt = rec["gt"]
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
    det_all[0][0].append(np.concatenate([ib, sc[:, None]], 1))
    for t in range(1, nT):
        for c in range(num_c[t]):
            col = offsets[t] + c
            det_all[t][c].append(np.concatenate([ib, sig[:, col:col + 1]], 1))
    if (n + 1) % 10000 == 0:
        print(f"[i3d] {n+1:,}/{len(keys):,}  {time.time()-t0:.0f}s", flush=True)

print(f"[i3d] {len(keys):,} frames | running baseline f-mAP ...", flush=True)
mAP, _, ap_strs = baseline_eval.evaluate(gt_all, det_all, all_classes, iou_thresh=0.5)
summary = {}
print("\n=== 3D-RetinaNet own detections, full-candidate, exp11 frames ===")
for t in range(nT):
    summary[_LABEL_TYPES[t]] = float(mAP[t])
    print(f"  {_LABEL_TYPES[t]:>10s}: {mAP[t]:.4f}%")
(E / "results_i3d_fullcand_baseline.json").write_text(json.dumps(
    {"summary": summary, "n_frames": len(keys),
     "per_class": {_LABEL_TYPES[t]: ap_strs[t] for t in range(nT)},
     "rows": "i3d_own_fullcand_top300"}, indent=2))
print("[i3d] wrote results_i3d_fullcand_baseline.json", flush=True)

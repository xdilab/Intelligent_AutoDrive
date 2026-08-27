"""Exp12 eval — CLIP-feature heads on YOLO full-candidate val rows.

Row construction mirrors exp11's record rows exactly: agentness/agent from
YOLO, action/loc/duplex/triplet from the head, conf-gated. GT from the exp11
I3D dump (same frames). Comparators: stacked-MLP sweep row.
"""
import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "/data/repos/PedestrianIntent++/ROAD_plus_plus_Baseline")
import modules.evaluation as baseline_eval
from modules.utils import get_individual_labels

E12 = Path(__file__).resolve().parent
E11 = E12.parent / "exp11_yolo"
BOX_W, BOX_H = 840, 600
_HEADS = ("agent", "action", "loc", "duplex", "triplet")
_LABEL_TYPES = ("agentness", "agent", "action", "loc", "duplex", "triplet")

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--feat-cache", default=str(E12 / "clip_feats_val.pkl"))
ap.add_argument("--key-style", default="stem", choices=("stem", "slash"))
args = ap.parse_args()

ck = torch.load(args.ckpt, weights_only=False)
FD = ck.get("feat_dim", 1024)
if ck["head"] == "flat":
    head = nn.Linear(FD, 184)
else:
    class PhraseHead(nn.Module):
        def __init__(self, embeds, in_dim=1024):
            super().__init__()
            self.proj = nn.Linear(in_dim, embeds.shape[1])
            self.register_buffer("P", F.normalize(embeds.float(), dim=-1))
            self.log_tau = nn.Parameter(torch.tensor(2.3))
            self.bias = nn.Parameter(torch.zeros(embeds.shape[0]))
        def forward(self, x):
            z = F.normalize(self.proj(x), dim=-1)
            return z @ self.P.t() * self.log_tau.exp() + self.bias
    pe = torch.load(E12 / "phrase_embeds.pt", weights_only=False)
    head = PhraseHead(pe["embeds"], in_dim=FD)
head.load_state_dict(ck["state"]); head.eval()

print("[eval] loading caches ...", flush=True)
feats = pickle.load(open(args.feat_cache, "rb"))["feats"]
yolo = pickle.load(open(E11 / "dets_v8x_best_val_fullcand.pkl", "rb"))["records"]
i3d_payload = pickle.load(open(E11 / "dets_i3d_val_fullcand.pkl", "rb"))
i3d = i3d_payload["records"]; labels = i3d_payload["labels"]
all_classes = [["agentness"], labels["agent"], labels["action"],
               labels["loc"], labels["duplex"], labels["triplet"]]
num_c = [len(c) for c in all_classes]
offsets = np.cumsum([0] + num_c); nT = len(num_c)

gt_all = [[] for _ in range(nT)]
det_all = [[[] for _ in range(num_c[t])] for t in range(nT)]
n_frames = 0; t0 = time.time()
with torch.no_grad():
    for stem in sorted(yolo):
        key = stem.rsplit("_", 1)[0] + "/" + str(int(stem.rsplit("_", 1)[1]))
        fkey = key if args.key_style == "slash" else stem
        if key not in i3d or fkey not in feats:
            continue
        yrec = yolo[stem]
        yb = yrec["boxes_xyxyn"].astype(np.float32)
        f = feats[fkey]
        if f.shape[0] != yb.shape[0]:
            continue
        f32 = np.nan_to_num(f.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        sig = torch.sigmoid(head(torch.from_numpy(f32))).numpy() if f.shape[0] \
            else np.zeros((0, 184), np.float32)
        sig = sig * yrec["conf"].astype(np.float32)[:, None]          # gate
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
print(f"\n=== exp12 {ck['head']} head f-mAP ===")
for t in range(nT):
    summary[_LABEL_TYPES[t]] = float(mAP[t])
    print(f"  {_LABEL_TYPES[t]:>10s}: {mAP[t]:.4f}%")
Path(args.out).write_text(json.dumps(
    {"ckpt": args.ckpt, "head": ck["head"], "summary": summary, "n_frames": n_frames,
     "per_class": {_LABEL_TYPES[t]: ap_strs[t] for t in range(nT)},
     "rows": "yolo_v8x_best_ep1_fullcand", "scores": f"clip_{ck['head']}_confgated"}, indent=2))
print(f"[eval] wrote {args.out}", flush=True)

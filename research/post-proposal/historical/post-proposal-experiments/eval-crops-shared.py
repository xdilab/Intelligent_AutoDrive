"""Exp12 eval — CLIP-feature heads on YOLO full-candidate val rows.

Row construction mirrors exp11's record rows exactly: agentness/agent from
YOLO, action/loc/duplex/triplet from the head, conf-gated. GT from the exp11
I3D dump (same frames). Comparators: stacked-MLP sweep row.
"""
import argparse, json, pickle, sys, time, hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "/data/repos/PedestrianIntent++/ROAD_plus_plus_Baseline")
import modules.evaluation as baseline_eval
from modules.utils import get_individual_labels

import os
E12 = Path(os.environ.get("ROAD_E12", "/data/repos/ROAD_Reason/experiments/exp12_phrase_head"))
E11 = Path(os.environ.get("ROAD_E11", str(E12.parent / "exp11_yolo")))
BOX_W, BOX_H = 840, 600
_HEADS = ("agent", "action", "loc", "duplex", "triplet")
_LABEL_TYPES = ("agentness", "agent", "action", "loc", "duplex", "triplet")

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--feat-cache", default=str(E12 / "clip_feats_val.pkl"))
ap.add_argument("--key-style", default="stem", choices=("stem", "slash"))
ap.add_argument("--comp-mlp", default=None)
ap.add_argument("--phrase-ckpt", default=None,
                help="exp14 fusion: phrase head whose raw composition sigmoids join the MLP input")
ap.add_argument("--coords", action="store_true",
                help="append normalized [cx, cy, w, h] to the head input (coordinate ablation)")
args = ap.parse_args()


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


def _load_embeds():
    import os as _os
    return torch.load(_os.environ.get("EMBEDS", str(E12 / "phrase_embeds.pt")), weights_only=False)


ck = torch.load(args.ckpt, weights_only=False)
FD = ck.get("feat_dim", 1024)
if ck["head"] == "flat":
    head = nn.Linear(FD, 184)
else:
    head = PhraseHead(_load_embeds()["embeds"], in_dim=FD)
head.load_state_dict(ck["state"]); head.eval()
comp = None
if args.comp_mlp:
    _c = torch.load(args.comp_mlp, weights_only=False)
    from pathlib import Path as _P
    assert _c.get("head_ckpt") is None or _P(_c["head_ckpt"]).name == _P(args.ckpt).name, \
        f"comp mlp trained against {_c['head_ckpt']}, got {args.ckpt}"
    comp = nn.Sequential(nn.Linear(_c["in_dim"], 512), nn.ReLU(), nn.Linear(512, 135))
    comp.load_state_dict(_c["state"]); comp.eval()
phead = None
if args.phrase_ckpt:
    assert comp is not None and _c["in_dim"] == 49 + 135 + 1024, \
        "--phrase-ckpt requires a fusion comp MLP (in_dim 1208)"
    _pn = _c.get("phrase_ckpt")
    assert _pn is None or Path(_pn).name == Path(args.phrase_ckpt).name, \
        f"fusion mlp trained against {_pn}, got {args.phrase_ckpt}"
    pck = torch.load(args.phrase_ckpt, weights_only=False)
    phead = nn.Linear(pck.get("feat_dim",1024),184) if pck["head"] == "flat" else PhraseHead(_load_embeds()["embeds"], in_dim=pck.get("feat_dim", 1024))
    phead.load_state_dict(pck["state"]); phead.eval()

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
manifest = json.loads(Path(args.manifest).read_text())
frame_keys = manifest["frames"]
assert len(frame_keys) == len(set(frame_keys)), "Duplicate manifest frames"
actual_hash = hashlib.sha256(b"".join(k.encode()+np.asarray(yolo[k]["boxes_xyxyn"]).tobytes()+np.asarray(yolo[k]["conf"]).tobytes()+np.asarray(yolo[k]["cls"]).tobytes() for k in frame_keys)).hexdigest()
assert actual_hash == manifest["candidate_sha256"], "Candidate population changed"
empty_feature_frames = []
n_frames = 0; t0 = time.time()
with torch.no_grad():
    for stem in frame_keys:
        key = stem.rsplit("_", 1)[0] + "/" + str(int(stem.rsplit("_", 1)[1]))
        fkey = key if args.key_style == "slash" else stem
        assert stem in yolo and key in i3d, f"Missing detector/GT record: {stem}"
        yrec = yolo[stem]
        yb = yrec["boxes_xyxyn"].astype(np.float32)
        if fkey in feats:
            f = feats[fkey]
        elif len(yb) == 0:
            f = np.empty((0, FD), dtype=np.float16)
            empty_feature_frames.append(stem)
        else:
            raise ValueError(f"Missing features for nonempty frame: {stem}")
        assert f.shape == (len(yb), FD), f"Feature row/dimension mismatch: {stem}"
        assert np.isfinite(f).all(), f"Nonfinite features: {stem}"
        f32 = f.astype(np.float32)
        if args.coords and f.shape[0]:
            cf = np.stack([(yb[:, 0] + yb[:, 2]) / 2, (yb[:, 1] + yb[:, 3]) / 2,
                           yb[:, 2] - yb[:, 0], yb[:, 3] - yb[:, 1]], 1).astype(np.float32)
            f32 = np.concatenate([f32, cf], 1)
        sig = torch.sigmoid(head(torch.from_numpy(f32))).numpy() if f.shape[0] \
            else np.zeros((0, 184), np.float32)
        sig_raw = sig.copy() if comp is not None else None
        sig = sig * yrec["conf"].astype(np.float32)[:, None]          # gate
        if comp is not None and f.shape[0]:
            if phead is not None:
                psig_raw = torch.sigmoid(phead(torch.from_numpy(f32))).numpy()
                zin = torch.from_numpy(np.concatenate([sig_raw[:, :49], psig_raw[:, 49:184], f32], 1))
            else:
                zin = torch.from_numpy(np.concatenate([sig_raw[:, :49], f32], 1))
            c = torch.sigmoid(comp(zin)).numpy() * yrec["conf"].astype(np.float32)[:, None]
            sig[:, 49:98] = c[:, :49]; sig[:, 98:184] = c[:, 49:135]
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
    {"ckpt": args.ckpt, "head": ck["head"], "summary": summary, "n_frames": n_frames, "frame_sha256": manifest["frame_sha256"],
     "candidate_sha256": manifest["candidate_sha256"], "empty_feature_frames": empty_feature_frames,
     "per_class": {_LABEL_TYPES[t]: ap_strs[t] for t in range(nT)},
     "rows": "yolo_v8x_best_ep1_fullcand", "scores": f"clip_{ck['head']}_confgated" + ("_compmlp" if comp is not None else "")}, indent=2))
print(f"[eval] wrote {args.out}", flush=True)

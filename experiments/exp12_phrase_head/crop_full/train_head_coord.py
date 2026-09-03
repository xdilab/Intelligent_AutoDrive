"""Coordinate-append ablation — probe protocol (2/12 train shards, boxes persisted).

Trains TWO flat heads on identical rows from the re-cached probe subset:
  base:  Linear(1024 -> 184) on crop features alone
  coord: Linear(1028 -> 184) on [crop features | cx, cy, w, h] (normalized)

Hypothesis (Brandon 2026-09-01, wiki exp12-phrase-head-attribution): crops
normalize ego-relative position away, which is why location lags; four
coordinates may buy it back. Same recipe as train_clip_head.py (focal +
exp6 alphas, Adam 1e-3, bs 16384, 10 epochs, seed 0).

Usage: python -u train_head_coord.py
"""
import pickle, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True)
CF = Path(__file__).resolve().parent
E12 = CF.parent
ALPHAS = "/data/repos/ROAD_Reason/experiments/exp6_detection_steered/cache/flat_alphas.pt"
torch.manual_seed(0)


def focal(logits, targets, gamma=2.0, alpha=None):
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = targets * prob + (1.0 - targets) * (1.0 - prob)
    loss = ce * (1.0 - pt).pow(gamma)
    if alpha is not None:
        alpha = alpha.to(logits.device)
        loss = (targets * alpha + (1.0 - targets) * (1.0 - alpha)) * loss
    return loss.mean()


def coordfeat(bx):
    """xyxyn [n,4] -> normalized [cx, cy, w, h] [n,4]."""
    x1, y1, x2, y2 = bx[:, 0], bx[:, 1], bx[:, 2], bx[:, 3]
    return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], 1).astype(np.float32)


print("[coord] loading probe shard caches ...", flush=True)
Xs, Ys, Cs = [], [], []
for sh in ("crop_feats_train_probe.shard0of2.pkl", "crop_feats_train_probe.shard1of2.pkl"):
    d = pickle.load(open(CF / sh, "rb"))
    ks = [k for k in sorted(d["feats"]) if d["feats"][k].shape[0]]
    Xs.append(np.concatenate([d["feats"][k] for k in ks], 0).astype(np.float32))
    Ys.append(np.concatenate([d["targets"][k] for k in ks], 0).astype(np.float32))
    Cs.append(np.concatenate([coordfeat(d["boxes"][k].astype(np.float32)) for k in ks], 0))
    print(f"[coord] {sh}: {Xs[-1].shape[0]:,} rows / {len(ks):,} frames", flush=True)
X = np.nan_to_num(np.concatenate(Xs, 0), nan=0., posinf=0., neginf=0.)
Y = np.concatenate(Ys, 0)
C = np.concatenate(Cs, 0)
assert X.shape[0] == Y.shape[0] == C.shape[0] and C.shape[1] == 4
print(f"[coord] total rows {X.shape[0]:,}  pos-agentness {int(Y[:,0].astype(np.float64).sum()):,}", flush=True)

dev = torch.device("cuda:0")
alphas = torch.load(ALPHAS, weights_only=True)
Yt = torch.from_numpy(Y)

for name, feats, in_dim in (("base", X, 1024), ("coord", np.concatenate([X, C], 1), 1028)):
    torch.manual_seed(0)
    Xt = torch.from_numpy(feats)
    head = nn.Linear(in_dim, 184).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    N = Xt.shape[0]; t0 = time.time()
    for ep in range(10):
        perm = torch.randperm(N)
        tot = 0.0; nb = 0
        for i in range(0, N, 16384):
            idx = perm[i:i + 16384]
            loss = focal(head(Xt[idx].to(dev)), Yt[idx].to(dev), alpha=alphas)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        print(f"[coord] {name} ep{ep+1}/10 loss {tot/nb:.5f} ({time.time()-t0:.0f}s)", flush=True)
    out = E12 / f"clip_head_flat_probe_{name}.pt"
    torch.save({"state": head.state_dict(), "head": "flat", "rows": int(N),
                "feat_dim": in_dim, "coords": name == "coord",
                "train_cache": "crop_feats_train_probe.shard{0,1}of2.pkl"}, out)
    print(f"[coord] saved {out}", flush=True)

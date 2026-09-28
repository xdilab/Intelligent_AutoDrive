"""Stacked composition MLP over CROP-feature primitives (full-coverage run).

Same video-level 2-fold OOF protocol as exp11 train_comp_mlp.py and exp12
train_comp_mlp_phrase.py, on the merged full crop train cache (1024-d feats,
targets built at cache time: GT multi-hot + random neg + YOLO junk, all rows
in one cache). Fold heads are throwaway flat Linear(1024->184) or PhraseHeads
per --prims; the MLP eats OOF primitive sigmoids[49] + the 1024-d crop
feature (in_dim 1073). Record head at eval: clip_head_{flat,phrase}_crop_full.pt.

Usage: python -u train_comp_mlp_crop.py --prims {flat,phrase}
"""
import argparse, pickle, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True)
CF = Path(__file__).resolve().parent          # crop_full/
E12 = CF.parent
ALPHAS = "/data/repos/ROAD_Reason/experiments/exp6_detection_steered/cache/flat_alphas.pt"
FEAT_DIM, N_PRIM, N_COMP = 1024, 49, 135
torch.manual_seed(0)

ap = argparse.ArgumentParser()
ap.add_argument("--prims", required=True, choices=("flat", "phrase"))
args = ap.parse_args()

def focal(logits, targets, gamma=2.0, alpha=None):
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = targets * prob + (1.0 - targets) * (1.0 - prob)
    loss = ce * (1.0 - pt).pow(gamma)
    if alpha is not None:
        alpha = alpha.to(logits.device)
        loss = (targets * alpha + (1.0 - targets) * (1.0 - alpha)) * loss
    return loss.mean()

class PhraseHead(nn.Module):
    def __init__(self, embeds, in_dim=FEAT_DIM):
        super().__init__()
        self.proj = nn.Linear(in_dim, embeds.shape[1])
        self.register_buffer("P", F.normalize(embeds.float(), dim=-1))
        self.log_tau = nn.Parameter(torch.tensor(2.3))
        self.bias = nn.Parameter(torch.zeros(embeds.shape[0]))
    def forward(self, x):
        z = F.normalize(self.proj(x), dim=-1)
        return z @ self.P.t() * self.log_tau.exp() + self.bias

print("[comb] loading merged crop train cache ...", flush=True)
d = pickle.load(open(CF / "crop_feats_train.pkl", "rb"))
ks = [k for k in sorted(d["feats"]) if d["feats"][k].shape[0]]
X = np.concatenate([d["feats"][k] for k in ks], 0).astype(np.float32)
Y = np.concatenate([d["targets"][k] for k in ks], 0).astype(np.float32)
V = np.concatenate([np.full(d["feats"][k].shape[0], k.rsplit("_", 1)[0]) for k in ks])
X = np.nan_to_num(X, nan=0., posinf=0., neginf=0.)
assert X.shape[1] == FEAT_DIM and Y.shape[1] == 184 and X.shape[0] == V.shape[0]
print(f"[comb] rows {X.shape[0]:,}  videos {len(set(V.tolist()))}", flush=True)

vids = sorted(set(V.tolist()))
foldA = set(v for i, v in enumerate(vids) if i % 2 == 0)
maskA = np.isin(V, list(foldA))
dev = torch.device("cuda:0")
alphas = torch.load(ALPHAS, weights_only=True)
pe = torch.load(E12 / "phrase_embeds.pt", weights_only=False)["embeds"]
Xt = torch.from_numpy(X); Yt = torch.from_numpy(Y)

def train_fold(rows, tag):
    torch.manual_seed(0)
    h = (nn.Linear(FEAT_DIM, 184) if args.prims == "flat" else PhraseHead(pe)).to(dev)
    opt = torch.optim.Adam(h.parameters(), lr=1e-3)
    idxs = np.nonzero(rows)[0]
    t0 = time.time()
    for ep in range(10):
        perm = idxs[torch.randperm(len(idxs)).numpy()]
        for i in range(0, len(perm), 16384):
            b = perm[i:i + 16384]
            loss = focal(h(Xt[b].to(dev)), Yt[b].to(dev), alpha=alphas)
            opt.zero_grad(); loss.backward(); opt.step()
    print(f"[comb] fold head {tag}: {len(idxs):,} rows ({time.time()-t0:.0f}s)", flush=True)
    return h.eval()

hA = train_fold(maskA, "A"); hB = train_fold(~maskA, "B")
Z = np.zeros((X.shape[0], N_PRIM), np.float32)
with torch.no_grad():
    for rows, h in ((maskA, hB), (~maskA, hA)):     # OOF: opposite fold's head
        idxs = np.nonzero(rows)[0]
        for i in range(0, len(idxs), 65536):
            b = idxs[i:i + 65536]
            Z[b] = torch.sigmoid(h(Xt[b].to(dev)))[:, :N_PRIM].cpu().numpy()
del hA, hB
print("[comb] OOF sigmoids done; training comp MLP ...", flush=True)

Zin = torch.from_numpy(np.concatenate([Z, X], 1))     # 49 + 1024 = 1073
Yc = Yt[:, N_PRIM:184]
torch.manual_seed(0)
mlp = nn.Sequential(nn.Linear(N_PRIM + FEAT_DIM, 512), nn.ReLU(), nn.Linear(512, N_COMP)).to(dev)
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
al = alphas[N_PRIM:184]
N = Zin.shape[0]; t0 = time.time()
for ep in range(10):
    perm = torch.randperm(N)
    tot = 0.0; nb = 0
    for i in range(0, N, 16384):
        b = perm[i:i + 16384]
        loss = focal(mlp(Zin[b].to(dev)), Yc[b].to(dev), alpha=al)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss.detach()); nb += 1
    print(f"[comb] ep{ep+1}/10 loss {tot/nb:.5f} ({time.time()-t0:.0f}s)", flush=True)
rec = E12 / f"clip_head_{args.prims}_crop_full.pt"
out = CF / f"comp_mlp_{args.prims}_crop_full.pt"
torch.save({"state": mlp.state_dict(), "head_ckpt": str(rec),
            "oof": True, "in_dim": N_PRIM + FEAT_DIM}, out)
print(f"[comb] saved {out}", flush=True)

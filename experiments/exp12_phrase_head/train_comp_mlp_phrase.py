"""Combination cell: stacked composition MLP over PHRASE-head primitives.

Same protocol as exp11's train_comp_mlp.py (video-level 2-fold OOF stacking)
but the fold heads are PhraseHeads on I3D features: two throwaway phrase
fold-heads generate each fold's out-of-fold primitive sigmoids; the MLP
(sigmoids[:49] + 256-d feature -> 512 -> 135) then learns composition.
Record head at eval: clip_head_phrase_i3d.pt.
"""
import json, pickle, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True)
E12 = Path(__file__).resolve().parent
E11 = E12.parent / "exp11_yolo"
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

class PhraseHead(nn.Module):
    def __init__(self, embeds, in_dim=256):
        super().__init__()
        self.proj = nn.Linear(in_dim, embeds.shape[1])
        self.register_buffer("P", F.normalize(embeds.float(), dim=-1))
        self.log_tau = nn.Parameter(torch.tensor(2.3))
        self.bias = nn.Parameter(torch.zeros(embeds.shape[0]))
    def forward(self, x):
        z = F.normalize(self.proj(x), dim=-1)
        return z @ self.P.t() * self.log_tau.exp() + self.bias

print("[comb] loading caches ...", flush=True)
def load(path, zero_targets=False):
    d = pickle.load(open(path, "rb"))
    ks = [k for k in sorted(d["feats"]) if d["feats"][k].shape[0]]
    X = np.concatenate([d["feats"][k] for k in ks], 0).astype(np.float32)
    if zero_targets or not d.get("targets"):
        Y = np.zeros((X.shape[0], 184), np.float32)
    else:
        Y = np.concatenate([d["targets"][k] for k in ks], 0).astype(np.float32)
    vids = np.concatenate([np.full(d["feats"][k].shape[0], k.split("/")[0] if "/" in k else k.rsplit("_", 1)[0])
                           for k in ks])
    return X, Y, vids

X1, Y1, V1 = load(E11 / "roi_feats_i3d_train.pkl")
X2, Y2, V2 = load(E11 / "roi_feats_i3d_train_junk.pkl")
X = np.nan_to_num(np.concatenate([X1, X2], 0), nan=0., posinf=0., neginf=0.)
Y = np.concatenate([Y1, Y2], 0); V = np.concatenate([V1, V2], 0)
print(f"[comb] rows {X.shape[0]:,}", flush=True)

vids = sorted(set(V.tolist()))
foldA = set(v for i, v in enumerate(vids) if i % 2 == 0)
maskA = np.isin(V, list(foldA))
dev = torch.device("cuda:0")
torch.cuda.set_per_process_memory_fraction(0.25, 0)
pe = torch.load(E12 / "phrase_embeds.pt", weights_only=False)["embeds"]
alphas = torch.load(ALPHAS, weights_only=True)
Xt = torch.from_numpy(X); Yt = torch.from_numpy(Y)

def train_fold_phrase(rows):
    torch.manual_seed(0)
    h = PhraseHead(pe).to(dev)
    opt = torch.optim.Adam(h.parameters(), lr=1e-3)
    idxs = np.nonzero(rows)[0]
    for ep in range(10):
        perm = idxs[torch.randperm(len(idxs)).numpy()]
        for i in range(0, len(perm), 16384):
            b = perm[i:i + 16384]
            xb, yb = Xt[b].to(dev), Yt[b].to(dev)
            loss = focal(h(xb), yb, alpha=alphas)
            opt.zero_grad(); loss.backward(); opt.step()
    return h.eval()

print("[comb] training phrase fold-heads ...", flush=True)
hA = train_fold_phrase(maskA); hB = train_fold_phrase(~maskA)
Z = np.zeros((X.shape[0], 49), np.float32)
with torch.no_grad():
    for rows, h in ((maskA, hB), (~maskA, hA)):     # OOF: opposite fold's head
        idxs = np.nonzero(rows)[0]
        for i in range(0, len(idxs), 65536):
            b = idxs[i:i + 65536]
            Z[b] = torch.sigmoid(h(Xt[b].to(dev)))[:, :49].cpu().numpy()
del hA, hB
print("[comb] OOF sigmoids done; training comp MLP ...", flush=True)

Zin = torch.from_numpy(np.concatenate([Z, X], 1))     # 49 + 256 = 305
Yc = Yt[:, 49:184]
torch.manual_seed(0)
mlp = nn.Sequential(nn.Linear(305, 512), nn.ReLU(), nn.Linear(512, 135)).to(dev)
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
al = alphas[49:184]
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
torch.save({"state": mlp.state_dict(), "head_ckpt": str(E12 / "clip_head_phrase_i3d.pt"),
            "oof": True, "in_dim": 305}, E12 / "comp_mlp_phrase_i3d.pt")
print("[comb] saved comp_mlp_phrase_i3d.pt", flush=True)

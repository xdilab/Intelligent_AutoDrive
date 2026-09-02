"""Exp14 — fusion composition MLP over crop features (full-coverage run).

Same video-level 2-fold OOF protocol as train_comp_mlp_crop.py, but the MLP
eats BOTH heads' OOF evidence: flat primitive sigmoids[49] + phrase
composition sigmoids[135] + the 1024-d crop feature (in_dim 1208). Motivated
by the tail analysis (wiki findings/exp12-crop-full-record, 2026-09-02): the
phrase head's strong triplet columns were discarded by the flat-only stack.

Record heads at eval: clip_head_flat_crop_full.pt (primitives + action/loc)
and clip_head_phrase_crop_full.pt (composition evidence).

Usage: python -u train_comp_mlp_fusion.py
"""
import pickle, sys, time
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


print("[fusion] loading merged crop train cache ...", flush=True)
d = pickle.load(open(CF / "crop_feats_train.pkl", "rb"))
ks = [k for k in sorted(d["feats"]) if d["feats"][k].shape[0]]
X = np.concatenate([d["feats"][k] for k in ks], 0).astype(np.float32)
Y = np.concatenate([d["targets"][k] for k in ks], 0).astype(np.float32)
V = np.concatenate([np.full(d["feats"][k].shape[0], k.rsplit("_", 1)[0]) for k in ks])
X = np.nan_to_num(X, nan=0., posinf=0., neginf=0.)
assert X.shape[1] == FEAT_DIM and Y.shape[1] == 184 and X.shape[0] == V.shape[0]
print(f"[fusion] rows {X.shape[0]:,}  videos {len(set(V.tolist()))}", flush=True)

vids = sorted(set(V.tolist()))
foldA = set(v for i, v in enumerate(vids) if i % 2 == 0)
maskA = np.isin(V, list(foldA))
dev = torch.device("cuda:0")
alphas = torch.load(ALPHAS, weights_only=True)
pe = torch.load(E12 / "phrase_embeds.pt", weights_only=False)["embeds"]
Xt = torch.from_numpy(X); Yt = torch.from_numpy(Y)


def train_fold(rows, kind, tag):
    torch.manual_seed(0)
    h = (nn.Linear(FEAT_DIM, 184) if kind == "flat" else PhraseHead(pe)).to(dev)
    opt = torch.optim.Adam(h.parameters(), lr=1e-3)
    idxs = np.nonzero(rows)[0]
    t0 = time.time()
    for ep in range(10):
        perm = idxs[torch.randperm(len(idxs)).numpy()]
        for i in range(0, len(perm), 16384):
            b = perm[i:i + 16384]
            loss = focal(h(Xt[b].to(dev)), Yt[b].to(dev), alpha=alphas)
            opt.zero_grad(); loss.backward(); opt.step()
    print(f"[fusion] fold head {kind}/{tag}: {len(idxs):,} rows ({time.time()-t0:.0f}s)", flush=True)
    return h.eval()


fA = train_fold(maskA, "flat", "A");   fB = train_fold(~maskA, "flat", "B")
pA = train_fold(maskA, "phrase", "A"); pB = train_fold(~maskA, "phrase", "B")

Zp = np.zeros((X.shape[0], N_PRIM), np.float32)   # OOF flat primitives
Zc = np.zeros((X.shape[0], N_COMP), np.float32)   # OOF phrase compositions
with torch.no_grad():
    for rows, hf, hp in ((maskA, fB, pB), (~maskA, fA, pA)):   # OOF: opposite fold
        idxs = np.nonzero(rows)[0]
        for i in range(0, len(idxs), 65536):
            b = idxs[i:i + 65536]
            xb = Xt[b].to(dev)
            Zp[b] = torch.sigmoid(hf(xb))[:, :N_PRIM].cpu().numpy()
            Zc[b] = torch.sigmoid(hp(xb))[:, N_PRIM:184].cpu().numpy()
del fA, fB, pA, pB
print("[fusion] OOF sigmoids done; training fusion MLP ...", flush=True)

IN_DIM = N_PRIM + N_COMP + FEAT_DIM                # 49 + 135 + 1024 = 1208
Zin = torch.from_numpy(np.concatenate([Zp, Zc, X], 1))
Yc = Yt[:, N_PRIM:184]
torch.manual_seed(0)
mlp = nn.Sequential(nn.Linear(IN_DIM, 512), nn.ReLU(), nn.Linear(512, N_COMP)).to(dev)
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
    print(f"[fusion] ep{ep+1}/10 loss {tot/nb:.5f} ({time.time()-t0:.0f}s)", flush=True)

out = CF / "comp_mlp_fusion_crop_full.pt"
torch.save({"state": mlp.state_dict(),
            "head_ckpt": str(E12 / "clip_head_flat_crop_full.pt"),
            "phrase_ckpt": str(E12 / "clip_head_phrase_crop_full.pt"),
            "oof": True, "in_dim": IN_DIM}, out)
print(f"[fusion] saved {out}", flush=True)

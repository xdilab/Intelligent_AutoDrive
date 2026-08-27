"""Exp12 — train heads on cached CLIP_S RoI features.

--head flat:   Linear(1024 -> 184)                        [C1 encoder-swap control]
--head phrase: proj Linear(1024 -> 512) -> cosine vs frozen phrase embeds
               x learnable temperature + per-class bias    [C2, formulation A]

Loss: focal-on-all with exp6 per-class alphas (verbatim copy convention).
Rows: GT + random background + YOLO junk (targets built at cache time).
"""
import argparse, pickle, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True)
E12 = Path(__file__).resolve().parent
ALPHAS = "/data/repos/ROAD_Reason/experiments/exp6_detection_steered/cache/flat_alphas.pt"

ap = argparse.ArgumentParser()
ap.add_argument("--head", required=True, choices=("flat", "phrase"))
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--bs", type=int, default=16384)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--feat-dim", type=int, default=1024)
ap.add_argument("--train-cache", default=str(E12 / "clip_feats_train.pkl"))
ap.add_argument("--extra-cache", default=None, help="second cache concatenated (e.g. exp11 junk)")
ap.add_argument("--tag", default="")
args = ap.parse_args()
torch.manual_seed(0)

def sigmoid_focal_with_logits(logits, targets, gamma=2.0, alpha=None):
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = targets * prob + (1.0 - targets) * (1.0 - prob)
    loss = ce * (1.0 - pt).pow(gamma)
    if alpha is not None:
        alpha = alpha.to(logits.device)
        alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha)
        loss = alpha_t * loss
    return loss.mean()

class PhraseHead(nn.Module):
    def __init__(self, embeds, in_dim=1024):
        super().__init__()
        self.proj = nn.Linear(in_dim, embeds.shape[1])
        self.register_buffer("P", F.normalize(embeds.float(), dim=-1))  # [184, D] frozen
        self.log_tau = nn.Parameter(torch.tensor(2.3))                   # tau ~ 10
        self.bias = nn.Parameter(torch.zeros(embeds.shape[0]))
    def forward(self, x):
        z = F.normalize(self.proj(x), dim=-1)
        return z @ self.P.t() * self.log_tau.exp() + self.bias

print("[train] loading cache ...", flush=True)
d = pickle.load(open(args.train_cache, "rb"))
keys = [k for k in sorted(d["feats"]) if d["feats"][k].shape[0]]
X = np.concatenate([d["feats"][k] for k in keys], 0).astype(np.float32)
Y = np.concatenate([d["targets"][k] for k in keys], 0).astype(np.float32)
if args.extra_cache:
    d2 = pickle.load(open(args.extra_cache, "rb"))
    k2 = [k for k in sorted(d2["feats"]) if d2["feats"][k].shape[0]]
    X2 = np.concatenate([d2["feats"][k] for k in k2], 0).astype(np.float32)
    if d2.get("targets"):
        Y2 = np.concatenate([d2["targets"][k] for k in k2], 0).astype(np.float32)
    else:
        Y2 = np.zeros((X2.shape[0], 184), np.float32)
    X = np.concatenate([X, X2], 0); Y = np.concatenate([Y, Y2], 0)
    print(f"[train] + extra rows {X2.shape[0]:,}", flush=True)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)  # sanitize fp16 inf rows
print(f"[train] rows {X.shape[0]:,}  pos-agentness {int(Y[:,0].astype(np.float64).sum()):,}", flush=True)

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if dev.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.12, 0)
X = torch.from_numpy(X); Y = torch.from_numpy(Y)
if args.head == "flat":
    head = nn.Linear(args.feat_dim, 184).to(dev)
else:
    pe = torch.load(E12 / "phrase_embeds.pt", weights_only=False)
    head = PhraseHead(pe["embeds"], in_dim=args.feat_dim).to(dev)
alphas = torch.load(ALPHAS, weights_only=True)
opt = torch.optim.Adam(head.parameters(), lr=args.lr)
N = X.shape[0]; t0 = time.time()
for ep in range(args.epochs):
    perm = torch.randperm(N)
    tot = 0.0; nb = 0
    for i in range(0, N, args.bs):
        idx = perm[i:i + args.bs]
        xb, yb = X[idx].to(dev), Y[idx].to(dev)
        loss = sigmoid_focal_with_logits(head(xb), yb, gamma=2.0, alpha=alphas)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss.detach()); nb += 1
    extra = f"  tau {float(head.log_tau.exp()):.2f}" if args.head == "phrase" else ""
    print(f"[train] ep{ep+1}/{args.epochs}  loss {tot/nb:.5f}{extra}  ({time.time()-t0:.0f}s)", flush=True)
out = E12 / f"clip_head_{args.head}{args.tag}.pt"
torch.save({"state": head.state_dict(), "head": args.head, "rows": int(N), "feat_dim": args.feat_dim,
            "train_cache": "clip_feats_train.pkl"}, out)
print(f"[train] saved {out}", flush=True)

"""Exp11 RoIAlign hybrid — train Linear(256->184) head on cached I3D RoI feats.

Rows: GT boxes (multi-hot targets) + background boxes (all-zero targets).
Loss: focal-on-all with exp6 flat alphas + corrected Godel t-norm (lam arg).
Verbatim loss code from exp9/losses.py (module-collision copy convention).

Usage: python -u train_head.py --lam 0 [--epochs 10]
"""
import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True)
E = Path("/data/repos/ROAD_Reason/experiments/exp11_yolo")
EXP6_ALPHAS = Path("/data/repos/ROAD_Reason/experiments/exp6_detection_steered/cache/flat_alphas.pt")
CONSTRAINTS = Path("/data/repos/ROAD_Reason/experiments/exp9_joint_heterogeneous/constraints_verified.json")
N_AGENTS, N_ACTIONS, N_LOCS, N_DUP, N_TRI = 10, 22, 16, 49, 86
AGENT_OFF, ACTION_OFF, LOC_OFF = 1, 11, 33
NUM_CLASSES = 184

ap = argparse.ArgumentParser()
ap.add_argument("--lam", type=float, required=True)
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--bs", type=int, default=65536)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--junk-cache", default=None)
ap.add_argument("--out-tag", default="")
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

class CorrectedTNorm(nn.Module):
    def __init__(self, lam):
        super().__init__()
        spec = json.loads(CONSTRAINTS.read_text())
        valid_d = set(map(tuple, spec["duplex_childs_derived"]))
        valid_t = set(map(tuple, spec["triplet_childs_derived"]))
        assert len(valid_d) == N_DUP and len(valid_t) == N_TRI
        inv_d = [(i, j) for i in range(N_AGENTS) for j in range(N_ACTIONS) if (i, j) not in valid_d]
        inv_t = [(i, j, k) for i in range(N_AGENTS) for j in range(N_ACTIONS)
                 for k in range(N_LOCS) if (i, j, k) not in valid_t]
        self.register_buffer("inv_d", torch.tensor(inv_d, dtype=torch.long))
        self.register_buffer("inv_t", torch.tensor(inv_t, dtype=torch.long))
        self.lam = lam
    def forward(self, logits):
        if self.lam == 0.0 or logits.shape[0] == 0:
            return logits.new_zeros(())
        p = torch.sigmoid(logits.float())
        pa = p[:, AGENT_OFF + self.inv_d[:, 0]]; pc = p[:, ACTION_OFF + self.inv_d[:, 1]]
        loss = torch.min(pa, pc).mean()
        pa = p[:, AGENT_OFF + self.inv_t[:, 0]]; pc = p[:, ACTION_OFF + self.inv_t[:, 1]]
        pl = p[:, LOC_OFF + self.inv_t[:, 2]]
        loss = loss + torch.min(torch.min(pa, pc), pl).mean()
        return self.lam * loss

print("[train] loading cache ...", flush=True)
d = pickle.load(open(E / "roi_feats_i3d_train.pkl", "rb"))
X = np.concatenate([d["feats"][k] for k in sorted(d["feats"]) if d["feats"][k].shape[0]], 0)
Y = np.concatenate([d["targets"][k] for k in sorted(d["feats"]) if d["feats"][k].shape[0]], 0)
if args.junk_cache:
    dj = pickle.load(open(args.junk_cache, "rb"))
    Xj = np.concatenate([dj["feats"][k] for k in sorted(dj["feats"]) if dj["feats"][k].shape[0]], 0)
    Yj = np.concatenate([dj["targets"][k] for k in sorted(dj["feats"]) if dj["feats"][k].shape[0]], 0)
    X = np.concatenate([X, Xj], 0); Y = np.concatenate([Y, Yj], 0)
    print(f"[train] + junk rows {Xj.shape[0]:,}", flush=True)
assert X.shape[0] == Y.shape[0] and Y.shape[1] == NUM_CLASSES
print(f"[train] rows {X.shape[0]:,}  pos-agentness {int(Y[:,0].astype(np.float64).sum()):,}", flush=True)

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if dev.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.25, 0)
X = torch.from_numpy(X).float(); Y = torch.from_numpy(Y).float()
head = nn.Linear(256, NUM_CLASSES).to(dev)
alphas = torch.load(EXP6_ALPHAS, weights_only=True)
tnorm = CorrectedTNorm(args.lam).to(dev)
opt = torch.optim.Adam(head.parameters(), lr=args.lr)
N = X.shape[0]
t0 = time.time()
for ep in range(args.epochs):
    perm = torch.randperm(N)
    tot = tf = tt = 0.0; nb = 0
    for i in range(0, N, args.bs):
        idx = perm[i:i + args.bs]
        xb, yb = X[idx].to(dev), Y[idx].to(dev)
        logits = head(xb)
        focal = sigmoid_focal_with_logits(logits, yb, gamma=2.0, alpha=alphas)
        tn = tnorm(logits)
        loss = focal + tn
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss); tf += float(focal); tt += float(tn); nb += 1
    print(f"[train] ep{ep+1}/{args.epochs}  loss {tot/nb:.5f}  focal {tf/nb:.5f}  "
          f"tnorm {tt/nb:.5f}  ({time.time()-t0:.0f}s)", flush=True)
out = E / f"head_roialign_lam{args.lam:g}{args.out_tag}.pt"
torch.save({"state": head.state_dict(), "lam": args.lam, "epochs": args.epochs,
            "rows": int(N)}, out)
print(f"[train] saved {out}", flush=True)

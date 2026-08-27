"""Exp11 stacked composition head — primitive sigmoids -> MLP for duplex/triplet.

A 2-layer MLP maps UNGATED primitive sigmoids (cols 0..48: agentness/agent/
action/loc) to the 135 compositional classes [49 duplex | 86 triplet] in
dataset label order. Variant "sig" feeds the 49 primitive sigmoids alone
(49-d input); "sigfeat" concatenates the 256-d RoI feature (305-d input).

Stacking protocol (default): 2-fold OUT-OF-FOLD inputs, folded BY VIDEO
(frames within a video are correlated — split on the video name, the cache-key
part before "/", alternating sorted unique names). Two fold heads
(Linear(256->184), train_head.py row-of-record recipe: focal-on-all with full
exp6 flat alphas, gamma 2.0, Adam 1e-3, bs 16384, 10 epochs, seed 0, fp32)
train internally — head_A on fold-A rows, head_B on fold-B — and each fold's
rows get the OTHER fold's head sigmoids, so MLP train-time inputs match
eval-time calibration (head applied to unseen rows). At EVAL the MLP consumes
the record head's (head_roialign_lam0_junkneg.pt) sigmoids; the fold heads
exist only to generate training inputs and are NOT saved. --no-oof restores
in-sample sigmoids from the record head (ablation; saves *_insample.pt).

Rows: GT+neg train cache + YOLO junk cache (all-zero targets), same row set
as the junkneg head. Loss: focal-on-all with exp6 flat alphas cols 49..183.
Verbatim loss code from train_head.py (module-collision copy convention).

Usage: python -u train_comp_mlp.py --variant sig [--no-oof] [--epochs 10]
"""
import argparse, pickle, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True)
E = Path("/data/repos/ROAD_Reason/experiments/exp11_yolo")
EXP6_ALPHAS = Path("/data/repos/ROAD_Reason/experiments/exp6_detection_steered/cache/flat_alphas.pt")
HEAD_CKPT = E / "head_roialign_lam0_junkneg.pt"
NUM_CLASSES = 184
N_PRIM = 49          # cols 0..48: agentness(1) + agent(10) + action(22) + loc(16)
N_COMP = 135         # cols 49..183: duplex(49) + triplet(86)
FOLD_EPOCHS, FOLD_BS, FOLD_LR = 10, 16384, 1e-3   # fold-head row-of-record recipe

ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True, choices=("sig", "sigfeat"))
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--bs", type=int, default=16384)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--no-oof", dest="oof", action="store_false",
                help="in-sample sigmoids from the record head (ablation)")
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

def load_cache(path):
    d = pickle.load(open(path, "rb"))
    ks = [k for k in sorted(d["feats"]) if d["feats"][k].shape[0]]
    X = np.concatenate([d["feats"][k] for k in ks], 0)
    Y = np.concatenate([d["targets"][k] for k in ks], 0)
    vids = [k.rsplit("/", 1)[0] for k in ks]
    cnt = [d["feats"][k].shape[0] for k in ks]
    return X, Y, vids, cnt

print("[train] loading caches ...", flush=True)
X1, Y1, v1, c1 = load_cache(E / "roi_feats_i3d_train.pkl")
Xj, Yj, vj, cj = load_cache(E / "roi_feats_i3d_train_junk.pkl")
X = np.concatenate([X1, Xj], 0).astype(np.float32)   # fp16 sums overflow — cast first
Y = np.concatenate([Y1, Yj], 0).astype(np.float32)
assert X.shape[0] == Y.shape[0] and Y.shape[1] == NUM_CLASSES
N = X.shape[0]
videos = sorted(set(v1) | set(vj))
fold_of = {v: i % 2 for i, v in enumerate(videos)}   # alternate sorted names: A=0, B=1
row_fold = np.repeat(np.array([fold_of[v] for v in v1 + vj], np.int64),
                     np.array(c1 + cj, np.int64))
assert row_fold.shape[0] == N
nA_v = sum(1 for v in videos if fold_of[v] == 0)
nA = int((row_fold == 0).sum())
print(f"[train] rows {N:,} (junk {Xj.shape[0]:,})  videos {len(videos)}  "
      f"foldA {nA_v}v/{nA:,}r  foldB {len(videos)-nA_v}v/{N-nA:,}r  "
      f"pos-comp {int(Y[:, N_PRIM:].sum()):,}", flush=True)

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if dev.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.10, 0)
X = torch.from_numpy(X)
Yfull = torch.from_numpy(Y)
Yc = Yfull[:, N_PRIM:NUM_CLASSES]                    # [N,135] duplex+triplet targets
alphas_full = torch.load(EXP6_ALPHAS, weights_only=True)

def train_fold_head(idx_np, tag):
    """Linear(256->184) on the given rows, train_head.py recipe, frozen after."""
    torch.manual_seed(0)
    h = nn.Linear(256, NUM_CLASSES).to(dev)
    opt = torch.optim.Adam(h.parameters(), lr=FOLD_LR)
    idx_all = torch.from_numpy(idx_np)
    n = idx_all.shape[0]
    t0 = time.time()
    tot = 0.0; nb = 1
    for ep in range(FOLD_EPOCHS):
        perm = idx_all[torch.randperm(n)]
        tot = 0.0; nb = 0
        for i in range(0, n, FOLD_BS):
            idx = perm[i:i + FOLD_BS]
            xb, yb = X[idx].to(dev), Yfull[idx].to(dev)
            loss = sigmoid_focal_with_logits(h(xb), yb, gamma=2.0, alpha=alphas_full)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
    print(f"[train] fold head {tag}: rows {n:,}  final focal {tot/nb:.5f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    h.eval()
    for p in h.parameters():
        p.requires_grad_(False)
    return h

@torch.no_grad()
def head_sig(h, idx_np):
    """Ungated sigmoid(h(X[idx]))[:, :49], batched."""
    out = torch.empty((idx_np.shape[0], N_PRIM), dtype=torch.float32)
    for i in range(0, idx_np.shape[0], args.bs):
        idx = torch.from_numpy(idx_np[i:i + args.bs])
        out[i:i + idx.shape[0]] = torch.sigmoid(h(X[idx].to(dev)))[:, :N_PRIM].cpu()
    return out

sig = torch.empty((N, N_PRIM), dtype=torch.float32)
if args.oof:
    idxA = np.nonzero(row_fold == 0)[0]
    idxB = np.nonzero(row_fold == 1)[0]
    head_A = train_fold_head(idxA, "A")
    head_B = train_fold_head(idxB, "B")
    print("[train] out-of-fold primitive sigmoids (ungated) ...", flush=True)
    sig[torch.from_numpy(idxA)] = head_sig(head_B, idxA)   # A rows scored by B's head
    sig[torch.from_numpy(idxB)] = head_sig(head_A, idxB)   # B rows scored by A's head
    del head_A, head_B
else:
    head = nn.Linear(256, NUM_CLASSES)
    head.load_state_dict(torch.load(HEAD_CKPT, weights_only=True)["state"])
    head.eval().to(dev)
    for p in head.parameters():
        p.requires_grad_(False)
    print("[train] in-sample primitive sigmoids (ungated, --no-oof) ...", flush=True)
    sig = head_sig(head, np.arange(N))

if args.variant == "sig":
    Z = sig                                          # [N,49]
else:
    Z = torch.cat([sig, X], 1)                       # [N,305]
in_dim = Z.shape[1]
torch.manual_seed(0)                                 # identical MLP init across modes
mlp = nn.Sequential(nn.Linear(in_dim, 512), nn.ReLU(), nn.Linear(512, N_COMP)).to(dev)
alphas = alphas_full[N_PRIM:NUM_CLASSES]
opt = torch.optim.Adam(mlp.parameters(), lr=args.lr)
print(f"[train] variant {args.variant}  in_dim {in_dim}  oof {args.oof}", flush=True)
t0 = time.time()
for ep in range(args.epochs):
    perm = torch.randperm(N)
    tot = tf = 0.0; nb = 0
    for i in range(0, N, args.bs):
        idx = perm[i:i + args.bs]
        zb, yb = Z[idx].to(dev), Yc[idx].to(dev)
        logits = mlp(zb)
        focal = sigmoid_focal_with_logits(logits, yb, gamma=2.0, alpha=alphas)
        loss = focal
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss); tf += float(focal); nb += 1
    print(f"[train] ep{ep+1}/{args.epochs}  loss {tot/nb:.5f}  focal {tf/nb:.5f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
out = E / (f"comp_mlp_{args.variant}.pt" if args.oof else f"comp_mlp_{args.variant}_insample.pt")
torch.save({"state": mlp.state_dict(), "variant": args.variant, "in_dim": in_dim,
            "epochs": args.epochs, "rows": int(N),
            "head_ckpt": str(HEAD_CKPT), "oof": bool(args.oof)}, out)
print(f"[train] saved {out}", flush=True)

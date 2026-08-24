"""Exp11 RoIAlign hybrid — cache I3D P3 features at external boxes.

train split: GT boxes (labels known) + up to 8 random background boxes/frame
             (IoU<0.3 vs GT, all-zero targets — exp2f negative recipe).
val split:   YOLOv8x best.pt full-candidate boxes (eval rows).

Features: P3 (stride 8, D=256), RoIAlign output 1x1, aligned=True — identical
arithmetic to exp4's in-wrapper pooling, at the box's own frame t.

Usage: CUDA_VISIBLE_DEVICES=0 python -u cache_roi_feats.py --split {train,val}
"""
import argparse, pickle, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

sys.stdout.reconfigure(line_buffering=True)
torch.cuda.set_per_process_memory_fraction(0.18, 0)

EXP6 = Path("/data/repos/ROAD_Reason/experiments/exp6_detection_steered")
sys.path.insert(0, str(EXP6))
import config as C
sys.path.insert(0, str(C.EXP4_DIR))
from model import RetinaNetWrapper
from dataloader import ROADWaymoDataset, clip_to_tensor, rescale_frame_targets

E11 = Path("/data/repos/ROAD_Reason/experiments/exp11_yolo")
_HEADS = ("agent", "action", "loc", "duplex", "triplet")
N_NEG, NEG_IOU = 8, 0.3
BOX_W, BOX_H = 840, 600

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True, choices=("train", "val"))
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(0)

device = torch.device("cuda:0")
ds = ROADWaymoDataset(anno_file=C.ANNO_FILE, frames_dir=C.FRAMES_DIR,
                      split=args.split, clip_len=C.CLIP_LEN, stride=C.CLIP_LEN)
print(f"[cache] {args.split}: {len(ds):,} clips", flush=True)
det = RetinaNetWrapper(ckpt_path=C.RETINANET_CKPT, top_k=40,
                       d_retina=C.RETINANET_SPATIAL_DIM).to(device).eval()

yolo = None
if args.split == "val":
    yolo = pickle.load(open(E11 / "dets_v8x_best_val_fullcand.pkl", "rb"))["records"]
    yolo = {k.rsplit("_", 1)[0] + "/" + str(int(k.rsplit("_", 1)[1])): v
            for k, v in yolo.items()}

def sample_negatives(gt):                      # gt [n,4] pixel xyxy
    out = []
    for _ in range(40):
        if len(out) >= N_NEG:
            break
        w = rng.uniform(0.02, 0.35) * BOX_W
        h = rng.uniform(0.02, 0.35) * BOX_H
        x1 = rng.uniform(0, BOX_W - w); y1 = rng.uniform(0, BOX_H - h)
        b = np.array([x1, y1, x1 + w, y1 + h], np.float32)
        if gt.shape[0]:
            ix = np.clip(np.minimum(b[2], gt[:, 2]) - np.maximum(b[0], gt[:, 0]), 0, None)
            iy = np.clip(np.minimum(b[3], gt[:, 3]) - np.maximum(b[1], gt[:, 1]), 0, None)
            inter = ix * iy
            union = (b[2]-b[0])*(b[3]-b[1]) + (gt[:,2]-gt[:,0])*(gt[:,3]-gt[:,1]) - inter
            if (inter / np.clip(union, 1e-9, None)).max() > NEG_IOU:
                continue
        out.append(b)
    return np.stack(out) if out else np.zeros((0, 4), np.float32)

@torch.no_grad()
def pool_at(clip, boxes_per_frame):
    """clip [T,3,600,840]; boxes_per_frame list of [n_t,4] pixel xyxy -> list [n_t,256]."""
    x = clip.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous().to(device)
    det._sources_cache.clear()
    _dec, _conf, _ego = det.net(x)
    p3 = det._sources_cache[-1][0]                       # [1,D,Tp,Hp,Wp]
    _, D, Tp, Hp, Wp = p3.shape
    T = clip.shape[0]
    if Tp != T:
        p3 = F.interpolate(p3, size=(T, Hp, Wp), mode="trilinear", align_corners=False)
    fmaps = p3[0].permute(1, 0, 2, 3)                    # [T,D,Hp,Wp]
    sx, sy = Wp / float(BOX_W), Hp / float(BOX_H)
    outs = []
    for t, bx in enumerate(boxes_per_frame):
        if bx.shape[0] == 0:
            outs.append(np.zeros((0, D), np.float16)); continue
        b = torch.from_numpy(bx.astype(np.float32)).to(device)
        b = b * torch.tensor([sx, sy, sx, sy], device=device)
        rois = torch.cat([torch.zeros(len(b), 1, device=device), b], 1)
        f = roi_align(fmaps[t:t+1], rois, output_size=(1, 1),
                      spatial_scale=1.0, aligned=True).squeeze(-1).squeeze(-1)
        outs.append(f.half().cpu().numpy())
    return outs

num_c = None
feats, targets, nb = {}, {}, {}
t0 = time.time()
for i in range(len(ds)):
    if args.limit and i >= args.limit:
        break
    vname, fids = ds.clips[i]
    pil_frames, frame_targets = ds[i]
    clip, h, w = clip_to_tensor(pil_frames)
    gt_frames = rescale_frame_targets(frame_targets, h, w)
    if num_c is None:
        num_c = [len(getattr(ds, f"{hd}_labels")) for hd in _HEADS]
    boxes_pf, targ_pf, keys_pf = [], [], []
    for t, fid in enumerate(fids):
        key = f"{vname}/{fid}"
        if key in feats:
            boxes_pf.append(np.zeros((0, 4), np.float32)); targ_pf.append(None); keys_pf.append(None)
            continue
        if args.split == "train":
            ft = gt_frames[t]
            if ft is None or ft["boxes"].shape[0] == 0:
                gtb = np.zeros((0, 4), np.float32)
                mh = [np.zeros((0, c), np.float32) for c in num_c]
            else:
                gtb = ft["boxes"].cpu().numpy().astype(np.float32)
                mh = [ft[hd].cpu().numpy().astype(np.float32) for hd in _HEADS]
            neg = sample_negatives(gtb)
            bx = np.concatenate([gtb, neg], 0)
            tg = np.zeros((bx.shape[0], 1 + sum(num_c)), np.float32)
            tg[:gtb.shape[0], 0] = 1.0                    # agentness
            off = 1
            for hi, m in enumerate(mh):
                tg[:gtb.shape[0], off:off + num_c[hi]] = m
                off += num_c[hi]
        else:
            rec = yolo.get(key)
            bx = (rec["boxes_xyxyn"] * np.array([BOX_W, BOX_H, BOX_W, BOX_H], np.float32)
                  ).astype(np.float32) if rec is not None else np.zeros((0, 4), np.float32)
            tg = None
        boxes_pf.append(bx); targ_pf.append(tg); keys_pf.append(key)
    pooled = pool_at(clip, boxes_pf)
    for t, key in enumerate(keys_pf):
        if key is None:
            continue
        feats[key] = pooled[t]
        nb[key] = pooled[t].shape[0]
        if targ_pf[t] is not None:
            targets[key] = targ_pf[t].astype(np.float16)
    if (i + 1) % 200 == 0:
        print(f"[cache] {i+1}/{len(ds)} clips  frames={len(feats):,}  "
              f"{(time.time()-t0)/(i+1):.2f}s/clip", flush=True)

out = E11 / f"roi_feats_i3d_{args.split}.pkl"
with open(out, "wb") as f:
    pickle.dump({"feats": feats, "targets": targets if args.split == "train" else None,
                 "n_boxes": nb,
                 "meta": {"split": args.split, "pool": "p3-roialign-1x1", "dim": 256,
                          "rows": "gt+neg8" if args.split == "train" else "yolo_v8x_fullcand"}},
                f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"[cache] wrote {len(feats):,} frames -> {out} ({time.time()-t0:.0f}s)", flush=True)

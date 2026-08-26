"""Exp12 — cache frozen InternVideo2_CLIP_S RoI features at exp11 rows.

val:   YOLO full-candidate boxes (dets_v8x_best_val_fullcand.pkl), all frames.
train: GT boxes (targets from JSON labels) + 8 random negatives + <=16 YOLO
       junk negatives per frame (protocol-equal to exp11 head training).

Per frame: clean native frame -> 224x224 -> x8 static clip -> CLIP_S vision
tower -> post-blocks tokens, T-mean -> [16,16,1024] grid -> RoI pool per box
-> fp16 [n,1024]. Loader = exp10's meta-init workaround, verbatim.

Usage: CUDA_VISIBLE_DEVICES=0 python -u cache_clip_feats.py --split {train,val}
"""
import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)
torch.cuda.set_per_process_memory_fraction(0.16, 0)

E12 = Path(__file__).resolve().parent
E11 = E12.parent / "exp11_yolo"
FRAMES = Path("/data/datasets/ROAD_plusplus/rgb-images")
JSON_PATH = "/data/datasets/ROAD_plusplus/road_waymo_trainval_v1.1.json"
REPO = "OpenGVLab/InternVideo2_CLIP_S"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BOX_W, BOX_H = 840, 600
N_NEG, NEG_IOU, N_JUNK, JUNK_IOU = 8, 0.3, 16, 0.3
_HEADS = ("agent", "action", "loc", "duplex", "triplet")

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True, choices=("train", "val"))
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(42)

def load_clip_s(device):
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    cfg = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
    cls = get_class_from_dynamic_module(
        "modeling_internvideo2encoder.InternVideo2_CLIP_small", REPO)
    m = cls(cfg)
    missing, unexpected = m.load_state_dict(
        load_file(hf_hub_download(REPO, "model.safetensors")), strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return m.half().to(device).eval()

def pool_roi(grid, box):
    H, W, _ = grid.shape
    x1, y1, x2, y2 = box.tolist()
    cl = max(0, int(x1 * W)); ch = min(W, max(cl + 1, round(x2 * W + 0.5)))
    rl = max(0, int(y1 * H)); rh = min(H, max(rl + 1, round(y2 * H + 0.5)))
    return grid[rl:rh, cl:ch, :].mean(dim=(0, 1))

def iou_vs(b, gt):
    if gt.shape[0] == 0:
        return np.zeros(b.shape[0], np.float32)
    ix = np.clip(np.minimum(b[:, None, 2], gt[None, :, 2]) - np.maximum(b[:, None, 0], gt[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(b[:, None, 3], gt[None, :, 3]) - np.maximum(b[:, None, 1], gt[None, :, 1]), 0, None)
    inter = ix * iy
    aa = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    bb = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    return (inter / np.clip(aa[:, None] + bb[None, :] - inter, 1e-9, None)).max(1)

d = json.load(open(JSON_PATH))
ALL = {h: d[f"all_{h}_labels"] for h in _HEADS}
BENCH = {h: d[f"{h}_labels"] for h in _HEADS}
num_c = [len(BENCH[h]) for h in _HEADS]
N_OUT = 1 + sum(num_c)          # 184
remap = {h: {i: BENCH[h].index(n) for i, n in enumerate(ALL[h]) if n in BENCH[h]} for h in _HEADS}

device = torch.device("cuda:0")
model = load_clip_s(device)
tokens = {}
model.vision_encoder.blocks[-1].register_forward_hook(
    lambda _m, _i, o: tokens.__setitem__("post", o if isinstance(o, torch.Tensor) else o[0]))

if args.split == "val":
    rows = pickle.load(open(E11 / "dets_v8x_best_val_fullcand.pkl", "rb"))["records"]
    keys = sorted(rows)
else:
    junk_dump = pickle.load(open(E11 / "dets_v8x_best_train_fullcand.pkl", "rb"))["records"]
    keys = []
    for vname, vid in sorted(d["db"].items()):
        if "train" not in vid["split_ids"]:
            continue
        for fk, fr in vid["frames"].items():
            if fr.get("annotated"):
                keys.append(f"{vname}_{int(fk):05d}")
if args.limit:
    keys = keys[: args.limit]
print(f"[cache] {args.split}: {len(keys):,} frames", flush=True)

feats, targets, nb = {}, {}, {}
t0 = time.time()
CKPT = E12 / f"clip_feats_{args.split}.pkl"
for ki, stem in enumerate(keys):
    vname, fid = stem.rsplit("_", 1)
    if args.split == "val":
        bx = rows[stem]["boxes_xyxyn"].astype(np.float32)
        tg = None
    else:
        fr = d["db"][vname]["frames"][str(int(fid))]
        gt_list, mh_rows = [], []
        for a in fr.get("annos", {}).values():
            x1, y1, x2, y2 = [min(max(z, 0.0), 1.0) for z in a["box"]]
            if x2 <= x1 or y2 <= y1:
                continue
            gt_list.append([x1, y1, x2, y2])
            row = np.zeros(N_OUT, np.float32); row[0] = 1.0
            off = 1
            for hi, h in enumerate(_HEADS):
                for aid in a[f"{h}_ids"]:
                    j = remap[h].get(aid)
                    if j is not None:
                        row[off + j] = 1.0
                off += num_c[hi]
            mh_rows.append(row)
        gt = np.array(gt_list, np.float32).reshape(-1, 4)
        negs = []
        for _ in range(60):
            if len(negs) >= N_NEG:
                break
            w = rng.uniform(0.02, 0.35); h = rng.uniform(0.02, 0.35)
            x1 = rng.uniform(0, 1 - w); y1 = rng.uniform(0, 1 - h)
            b = np.array([x1, y1, x1 + w, y1 + h], np.float32)
            if gt.shape[0] == 0 or iou_vs(b[None], gt)[0] <= NEG_IOU:
                negs.append(b)
        negs = np.array(negs, np.float32).reshape(-1, 4)
        jrec = junk_dump.get(stem)
        junk = np.zeros((0, 4), np.float32)
        if jrec is not None:
            yb = jrec["boxes_xyxyn"].astype(np.float32)
            m = iou_vs(yb, gt) < JUNK_IOU
            cand = np.nonzero(m)[0]
            if cand.shape[0] > N_JUNK:
                cand = rng.choice(cand, N_JUNK, replace=False)
            junk = yb[cand]
        bx = np.concatenate([gt, negs, junk], 0)
        tg = np.zeros((bx.shape[0], N_OUT), np.float16)
        if gt.shape[0]:
            tg[: gt.shape[0]] = np.stack(mh_rows).astype(np.float16)
    if bx.shape[0] == 0:
        continue
    img = Image.open(FRAMES / vname / f"{int(fid):05d}.jpg").convert("RGB")
    arr = np.asarray(img.resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).repeat(1, 8, 1, 1, 1)
    tokens.clear()
    with torch.no_grad():
        model.encode_vision(x.half().to(device))
    tok = tokens["post"]
    tok = tok[:, tok.shape[1] - 8 * 256:, :]
    grid = tok.view(8, 16, 16, -1).float().mean(dim=0)
    f = torch.stack([pool_roi(grid, torch.from_numpy(b)) for b in np.clip(bx, 0, 1)]).half().cpu().numpy()
    feats[stem] = f; nb[stem] = int(bx.shape[0])
    if tg is not None:
        targets[stem] = tg
    if (ki + 1) % 2000 == 0:
        print(f"[cache] {ki+1}/{len(keys)} | {(time.time()-t0)/(ki+1):.3f}s/frame", flush=True)
        with open(str(CKPT) + ".partial", "wb") as fh:
            pickle.dump({"feats": feats, "targets": targets or None, "n_boxes": nb,
                         "meta": {"partial": True}}, fh)

with open(CKPT, "wb") as fh:
    pickle.dump({"feats": feats, "targets": targets or None, "n_boxes": nb,
                 "meta": {"encoder": REPO, "split": args.split, "dim": 1024,
                          "rows": "yolo_fullcand" if args.split == "val" else "gt+neg8+junk16"}}, fh)
print(f"[cache] wrote {len(feats):,} frames -> {CKPT} ({time.time()-t0:.0f}s)", flush=True)

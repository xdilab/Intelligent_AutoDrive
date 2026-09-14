"""Exp12 crop-RoI cell (Moradi #3): per-box padded crops through CLIP_S.

Per box: pad the box PAD x around its center (clamped), crop that SAME region
from each of the 8 real frames (GPU roi_align to 224x224, bilinear), stack ->
[8,3,224,224] clip -> CLIP_S vision tower -> keyframe-slice tokens ->
mean over the 16x16 grid -> 1024-d fp16. Full candidate coverage.

val: YOLO full-candidate boxes. train: GT + 8 random negatives + <=16 junk
(same row protocol as the featuremap cells).

Usage: CUDA_VISIBLE_DEVICES=N python -u cache_crop_feats.py --split {train,val} [--pad 2.0]
"""
import argparse, json, pickle, sys, time, hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

import os
E12 = Path(os.environ.get("ROADCROP_OUT", str(Path(__file__).resolve().parent)))
E11 = Path(os.environ.get("ROADCROP_DUMPS", "/data/repos/ROAD_Reason/experiments/exp11_yolo"))
FRAMES = Path(os.environ.get("ROADCROP_FRAMES", "/data/datasets/ROAD_plusplus/rgb-images"))
JSON_PATH = os.environ.get("ROADCROP_JSON", "/data/datasets/ROAD_plusplus/road_waymo_trainval_v1.1.json")
REPO = "OpenGVLab/InternVideo2_CLIP_S"
REV = "1f9fca1389fd883defc652634d95a21121c85a8c"
ADAPTED = Path(os.environ["ADAPTED_CHECKPOINT"])
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
N_NEG, NEG_IOU, N_JUNK, JUNK_IOU = 8, 0.3, 16, 0.3
ENC_BS = 64                  # crops per encoder forward
_HEADS = ("agent", "action", "loc", "duplex", "triplet")

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True, choices=("train", "val"))
ap.add_argument("--pad", type=float, default=2.0)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1, help="split keys into N interleaved shards")
ap.add_argument("--ishard", type=int, default=0)
ap.add_argument("--subset-every", type=int, default=1, help="take every Nth frame")
ap.add_argument("--shard", action="store_true", help="train: 2/12 shard subset")
args = ap.parse_args()
rng = np.random.default_rng(42)

def load_clip_s(device, adapted=False):
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    cfg = AutoConfig.from_pretrained(REPO, trust_remote_code=True, revision=REV, local_files_only=True)
    cls = get_class_from_dynamic_module(
        "modeling_internvideo2encoder.InternVideo2_CLIP_small", REPO, revision=REV, local_files_only=True)
    m = cls(cfg)
    missing, unexpected = m.load_state_dict(
        load_file(hf_hub_download(REPO, "model.safetensors", revision=REV, local_files_only=True)), strict=False)
    assert not missing and not unexpected
    if adapted:
        ck = torch.load(ADAPTED, map_location="cpu", weights_only=False)
        assert ck["revision"] == REV and ck["epoch"] == 2
        base = m.state_dict()
        assert ck["state_dict"] and set(ck["state_dict"]) <= set(base)
        for name, value in ck["state_dict"].items():
            assert value.shape == base[name].shape and torch.isfinite(value).all(), name
        base.update(ck["state_dict"]); m.load_state_dict(base, strict=True)
    for p in m.parameters(): p.requires_grad=False
    return m.half().to(device).eval()

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
N_OUT = 1 + sum(num_c)
remap = {h: {i: BENCH[h].index(n) for i, n in enumerate(ALL[h]) if n in BENCH[h]} for h in _HEADS}

device = torch.device("cuda:0")
models = {"original": load_clip_s(device), "adapted": load_clip_s(device, True)}
tokens = {}
for name, model in models.items():
    model.vision_encoder.blocks[-1].register_forward_hook(
        lambda _m, _i, o, name=name: tokens.__setitem__(name, o if isinstance(o, torch.Tensor) else o[0]))
checkpoint_sha = hashlib.sha256(ADAPTED.read_bytes()).hexdigest()
MEAN = MEAN.to(device).half(); STD = STD.to(device).half()

if args.split == "val":
    rows = pickle.load(open(E11 / "dets_v8x_best_val_fullcand.pkl", "rb"))["records"]
    keys = sorted(rows)
else:
    junk_dump = pickle.load(open(E11 / "dets_v8x_best_train_fullcand.pkl", "rb"))["records"]
    keys = [f"{v}_{int(fk):05d}" for v, vid in sorted(d["db"].items()) if "train" in vid["split_ids"]
            for fk, fr in vid["frames"].items() if fr.get("annotated")]
if args.split == "train" and args.shard:
    keys = [k for i, k in enumerate(keys) if i % 12 in (0, 1)]
if args.subset_every > 1:
    keys = keys[:: args.subset_every]
if args.nshard > 1:
    keys = keys[args.ishard :: args.nshard]
if args.limit:
    keys = keys[: args.limit]
print(f"[crop] {args.split}: {len(keys):,} frames | pad {args.pad}x | enc_bs {ENC_BS}", flush=True)

_tag = '_probe' if (args.subset_every > 1 or args.shard) else ''
if args.nshard > 1:
    _tag += f".shard{args.ishard}of{args.nshard}"
CKPT = E12 / f"crop_feats_{args.split}{_tag}.pkl"
feats, adapted_feats, targets, nb, bxs = {}, {}, {}, {}, {}
import sys as _sys
if CKPT.exists():
    print(f"[crop] {CKPT} already complete; exiting", flush=True); _sys.exit(0)
_partial = Path(str(CKPT) + ".partial")
if _partial.exists():
    try:
        _p = pickle.load(open(_partial, "rb"))
        assert _p["checkpoint_sha256"] == checkpoint_sha
        feats.update(_p["feats"]); adapted_feats.update(_p["adapted_feats"]); nb.update(_p["n_boxes"])
        if _p.get("targets"): targets.update(_p["targets"])
        if _p.get("boxes"): bxs.update(_p["boxes"])
        print(f"[crop] resumed {len(feats):,} frames from partial", flush=True)
    except Exception as _e:
        raise RuntimeError("Partial cache validation failed") from _e
t0 = time.time()
for ki, stem in enumerate(keys):
    if stem in feats:
        continue
    vname, fid = stem.rsplit("_", 1)
    fid_i = int(fid)
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(("paired-row-v1:"+stem).encode()).digest()[:8], "little"))
    # ---- rows for this frame (normalized xyxy) + targets (train) ----
    if args.split == "val":
        bx = rows[stem]["boxes_xyxyn"].astype(np.float32)
        tg = None
    else:
        fr = d["db"][vname]["frames"][str(fid_i)]
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
    n = bx.shape[0]
    if n == 0:
        feats[stem] = np.zeros((0,1024),np.float16); adapted_feats[stem] = feats[stem].copy(); nb[stem]=0; bxs[stem]=bx.copy()
        if tg is not None: targets[stem]=tg
        continue
    # ---- load the 8 real frames once, normalized, on GPU ----
    nf = d["db"][vname]["numf"]
    fids = [min(max(fid_i - 3 + j, 1), nf) for j in range(8)]
    key_t = fids.index(min(max(fid_i, 1), nf))
    imgs = []
    for fj in fids:
        im = Image.open(FRAMES / vname / f"{fj:05d}.jpg").convert("RGB")
        imgs.append(np.asarray(im, dtype=np.uint8))
    H, W = imgs[0].shape[:2]
    ft = torch.from_numpy(np.stack(imgs)).to(device).permute(0, 3, 1, 2).half().div_(255.0)
    ft = (ft - MEAN) / STD                                            # [8,3,H,W]
    # ---- padded crop regions in pixels ----
    cx = (bx[:, 0] + bx[:, 2]) / 2 * W
    cy = (bx[:, 1] + bx[:, 3]) / 2 * H
    bw = np.maximum((bx[:, 2] - bx[:, 0]) * W, 8) * args.pad
    bh = np.maximum((bx[:, 3] - bx[:, 1]) * H, 8) * args.pad
    px1 = np.clip(cx - bw / 2, 0, W - 1); px2 = np.clip(cx + bw / 2, 1, W)
    py1 = np.clip(cy - bh / 2, 0, H - 1); py2 = np.clip(cy + bh / 2, 1, H)
    boxes_px = torch.from_numpy(np.stack([px1, py1, px2, py2], 1)).to(device).half()
    # rois for all 8 frames x n boxes: batch idx = frame idx
    rois = torch.cat([
        torch.arange(8, device=device).repeat_interleave(n).unsqueeze(1).half(),
        boxes_px.repeat(8, 1)], 1)                                    # [8n, 5]
    crops = roi_align(ft, rois, output_size=(224, 224), spatial_scale=1.0,
                      aligned=True)                                    # [8n,3,224,224]
    crops = crops.view(8, n, 3, 224, 224).permute(1, 0, 2, 3, 4)      # [n,8,3,224,224]
    # ---- encoder in chunks ----
    for condition, model in models.items():
        out = np.zeros((n, 1024), np.float16)
        with torch.no_grad():
            for i in range(0, n, ENC_BS):
                chunk = crops[i:i + ENC_BS].contiguous()
                tokens.clear(); model.encode_vision(chunk)
                tok = tokens[condition]
                tok = tok[:, tok.shape[1] - 8 * 256:, :]
                tv = tok.view(chunk.shape[0], 8, 256, -1)[:, key_t]
                out[i:i + ENC_BS] = tv.float().mean(dim=1).half().cpu().numpy()
        assert np.isfinite(out).all(), (stem, condition)
        (feats if condition == "original" else adapted_feats)[stem] = out
    nb[stem] = n; bxs[stem] = bx.copy()
    if tg is not None:
        targets[stem] = tg
    if (ki + 1) % 500 == 0:
        print(f"[crop] {ki+1}/{len(keys)} | {(time.time()-t0)/(ki+1):.3f}s/frame", flush=True)
        temp = Path(str(CKPT)+".writing")
        with temp.open("wb") as fh:
            pickle.dump({"feats":feats,"adapted_feats":adapted_feats,"targets":targets or None,"n_boxes":nb,"boxes":bxs,"checkpoint_sha256":checkpoint_sha,"meta":{"partial":True}},fh)
        temp.replace(_partial)

assert set(feats) == set(adapted_feats) == set(nb) == set(bxs)
row_hash=hashlib.sha256()
for stem in sorted(bxs):
    row_hash.update(stem.encode());row_hash.update(bxs[stem].astype(np.float32).tobytes())
    if stem in targets:row_hash.update(targets[stem].tobytes())
meta={"encoder":REPO,"revision":REV,"split":args.split,"dim":1024,"pad":args.pad,"clips":True,"row_seed":"sha256(paired-row-v1:frame)","row_sha256":row_hash.hexdigest(),"partial":False,"frame_count":len(feats)}
with Path(str(CKPT)+".writing").open("wb") as fh:
    pickle.dump({"feats":feats,"adapted_feats":adapted_feats,"targets":targets or None,"n_boxes":nb,"boxes":bxs,"checkpoint_sha256":checkpoint_sha,"meta":meta},fh,protocol=pickle.HIGHEST_PROTOCOL)
Path(str(CKPT)+".writing").replace(CKPT)
print(json.dumps({"complete":str(CKPT),"frames":len(feats),"seconds":time.time()-t0,**meta}),flush=True)

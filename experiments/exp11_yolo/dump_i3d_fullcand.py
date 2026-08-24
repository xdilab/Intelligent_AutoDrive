"""Full-candidate I3D (3D-RetinaNet) val dump for the exp11 hybrid control.

exp6's dump_detections.py with the new-row protocol: FULL val coverage
(stride = CLIP_LEN = 8, every annotated frame exactly once), top-K 300,
conf 0.001 (mirrors the YOLO full-candidate dump). Memory-capped to run
beside YOLO26x training.

Usage: CUDA_VISIBLE_DEVICES=0 python -u dump_i3d_fullcand.py
"""
import pickle, sys, time
from pathlib import Path
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
torch.cuda.set_per_process_memory_fraction(0.18, 0)

EXP6 = Path("/data/repos/ROAD_Reason/experiments/exp6_detection_steered")
sys.path.insert(0, str(EXP6))
import config as C
sys.path.insert(0, str(C.EXP4_DIR))
from model import RetinaNetWrapper
from dataloader import ROADWaymoDataset, clip_to_tensor, rescale_frame_targets

OUT = Path("/data/repos/ROAD_Reason/experiments/exp11_yolo/dets_i3d_val_fullcand.pkl")
_HEADS = ("agent", "action", "loc", "duplex", "triplet")

device = torch.device("cuda:0")
ds = ROADWaymoDataset(anno_file=C.ANNO_FILE, frames_dir=C.FRAMES_DIR,
                      split="val", clip_len=C.CLIP_LEN, stride=C.CLIP_LEN)
print(f"[dump] val clips: {len(ds):,} (full coverage)", flush=True)
det = RetinaNetWrapper(ckpt_path=C.RETINANET_CKPT, top_k=300,
                       d_retina=C.RETINANET_SPATIAL_DIM,
                       conf_thresh=0.001).to(device).eval()

records = {}
t0 = time.time()
with torch.no_grad():
    for i in range(len(ds)):
        vname, fids = ds.clips[i]
        pil_frames, frame_targets = ds[i]
        clip, h, w = clip_to_tensor(pil_frames)
        out = det(clip.unsqueeze(0).to(device))
        boxes = out["boxes"][0].float().cpu().numpy()
        scores = out["scores"][0].float().cpu().numpy()
        logits = out["logits_tube"][0].cpu().numpy()
        gt_frames = rescale_frame_targets(frame_targets, h, w)
        for t, fid in enumerate(fids):
            key = f"{vname}/{fid}"
            if key in records:
                continue
            ft = gt_frames[t]
            gt = None
            if ft is not None:
                gt = {"boxes": ft["boxes"].cpu().numpy().astype(np.float32)}
                for hname in _HEADS:
                    gt[hname] = ft[hname].cpu().numpy().astype(np.float32)
            records[key] = {"boxes": boxes[:, t, :].astype(np.float32),
                            "scores": scores.astype(np.float32),
                            "logits": logits[:, t, :].astype(np.float16),
                            "gt": gt}
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"[dump] {i+1}/{len(ds)} clips  frames={len(records):,}  "
                  f"{el/(i+1):.2f}s/clip", flush=True)

payload = {"records": records,
           "labels": {h: getattr(ds, f"{h}_labels") for h in _HEADS},
           "meta": {"split": "val", "stride": C.CLIP_LEN, "topk": 300,
                    "conf_thresh": 0.001, "n_frames": len(records),
                    "protocol": "full-candidate full-coverage"}}
with open(OUT, "wb") as f:
    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"[dump] wrote {len(records):,} frames -> {OUT} ({time.time()-t0:.0f}s)", flush=True)

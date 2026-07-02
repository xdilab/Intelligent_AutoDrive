"""Exp4 — evaluate a checkpoint's frame-level f-mAP on ROAD-Waymo val.

Wraps the baseline's frame-level evaluation pipeline:
  modules.evaluation.evaluate(gts, dets, all_classes, iou_thresh=0.5)

This is the same evaluator the baseline 3D-RetinaNet uses; the only thing we
swap out is the source of predictions. The 17.76% agent f-mAP we report for the
baseline came from this exact function.

Pipeline:
  1. Build val dataloader (stride matches the baseline: SEQ_LEN * 4 = 32).
  2. Build the RetinaMoonFusion model, optionally load a checkpoint.
  3. For each clip, run the model under bf16 autocast → {logits, boxes, scores}.
  4. For each (clip, frame), for each label-type slice (1/10/22/16/49/86):
       - GT: get_individual_labels(boxes, multi-hot labels) → expanded [N, 5].
       - Det: filter_detections(args, scores, boxes) per class → [N_det, 5].
  5. Call baseline's evaluate.evaluate(gts, dets, names, iou_thresh) → per-class AP.

Usage:
  python -u eval.py [--ckpt path/to/model.pth] [--gpu 0] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

import torch.nn as nn

import config as C
from model import RetinaMoonFusion, RetinaNetWrapper
from dataloader import ROADWaymoDataset, exp4_collate_fn


class BaselineEvalModel(nn.Module):
    """Wraps the frozen baseline 3D-RetinaNet and exposes ALL anchors as the
    model's K dimension. Used for `--baseline` mode to validate the eval
    pipeline: should reproduce ~17.76% agent f-mAP on the epoch-25 checkpoint.

    Forward output:
        logits:  [B, A_total, T, 184]    raw per-anchor flat_conf (logits, NOT sigmoid'd)
        boxes:   [B, A_total, T, 4]      pixel xyxy
        scores:  [B, A_total]             dummy (not used by eval)
    """
    def __init__(self):
        super().__init__()
        # Reuse RetinaNetWrapper (it already loads the 17.76% ckpt + the forward
        # hook for P3 — we ignore the hook outputs here).
        self.det = RetinaNetWrapper(
            ckpt_path=C.RETINANET_CKPT, top_k=1, d_retina=C.RETINANET_SPATIAL_DIM,
        )
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, clip):
        # Mimic RetinaNetWrapper.forward but skip the top-K selection.
        # IMPORTANT: clear the forward-hook cache or we leak 500MB+ per call.
        self.det._sources_cache.clear()
        B, T, _, H, W = clip.shape
        x = clip.permute(0, 2, 1, 3, 4).contiguous()
        decoded, flat_conf, _ego = self.det.net(x)
        self.det._sources_cache.clear()  # release after use
        # decoded:    [B, T, A, 4]   pixel xyxy
        # flat_conf:  [B, T, A, 184] raw logits
        boxes  = decoded.permute(0, 2, 1, 3).contiguous()             # [B, A, T, 4]
        logits = flat_conf.permute(0, 2, 1, 3).contiguous()           # [B, A, T, 184]
        scores = torch.zeros(B, boxes.shape[1], device=clip.device)   # unused
        return {"logits": logits, "boxes": boxes, "scores": scores}

# Bootstrap baseline imports.
_BASELINE = "/data/repos/PedestrianIntent++/ROAD_plus_plus_Baseline"
if _BASELINE not in sys.path:
    sys.path.insert(0, _BASELINE)
import modules.evaluation as baseline_eval                        # noqa: E402
from modules.utils import filter_detections, get_individual_labels  # noqa: E402


# Match the baseline's val protocol: CONF_THRESH=0.025, NMS_THRESH=0.5, TOPK=10.
def _filter_args() -> Namespace:
    a = Namespace()
    a.CONF_THRESH = 0.025
    a.NMS_THRESH = 0.5
    a.TOPK = 10
    return a


_HEAD_NAMES = ("agent", "action", "loc", "duplex", "triplet")
_LABEL_TYPE_NAMES = ("agentness", "agent", "action", "loc", "duplex", "triplet")


def _frame_gt_for_label_type(nlt: int, num_c: int,
                             gt_boxes_np: np.ndarray,
                             gt_per_head_mh: dict) -> np.ndarray:
    """Build [N_expanded, 5] GT rows for one frame and one label type."""
    if gt_boxes_np.shape[0] == 0:
        return np.zeros((0, 5), dtype=np.float32)
    if nlt == 0:
        # agentness: every GT box contributes one row, class=0
        out = np.zeros((gt_boxes_np.shape[0], 5), dtype=np.float32)
        out[:, :4] = gt_boxes_np
        return out
    head = _HEAD_NAMES[nlt - 1]
    return get_individual_labels(gt_boxes_np, gt_per_head_mh[head]).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None,
                   help="Path to a trained checkpoint (.pth). If omitted, evaluates with random Q-Former.")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--limit", type=int, default=0,
                   help="Eval on first N clips (debug). 0 = all.")
    p.add_argument("--stride", type=int, default=32,
                   help="Frame stride between consecutive clips. Baseline uses SEQ_LEN*4 = 32 at val.")
    p.add_argument("--out", type=str, default=None,
                   help="Optional path to dump JSON {label_type: {class: AP, mAP: x}}.")
    p.add_argument("--baseline", action="store_true",
                   help="Eval the frozen 3D-RetinaNet baseline directly (all anchors). "
                        "Sanity check: should reproduce ~17.76% agent f-mAP.")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}  stride={args.stride}  ckpt={args.ckpt}", flush=True)

    # ---- Class names for f-mAP table headers ----
    with open(C.ANNO_FILE) as f:
        anno = json.load(f)
    all_classes = [
        ["agentness"],
        anno["agent_labels"],     # 10
        anno["action_labels"],    # 22
        anno["loc_labels"],       # 16
        anno["duplex_labels"],    # 49
        anno["triplet_labels"],   # 86
    ]
    num_classes_list = C.NUM_CLASSES_LIST                   # [1, 10, 22, 16, 49, 86]
    num_label_types = len(num_classes_list)
    assert all(len(all_classes[i]) == num_classes_list[i] for i in range(num_label_types))

    # ---- Dataset + loader (baseline-protocol stride at val time) ----
    ds = ROADWaymoDataset(
        anno_file=C.ANNO_FILE, frames_dir=C.FRAMES_DIR,
        split=args.split, clip_len=C.CLIP_LEN, stride=args.stride,
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
        collate_fn=exp4_collate_fn, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    n_clips = len(ds)
    print(f"[eval] {args.split} clips: {n_clips:,}", flush=True)

    # ---- Model ----
    print(f"[eval] building model (mode={'baseline' if args.baseline else 'exp4'}) ...",
          flush=True)
    t0 = time.time()
    if args.baseline:
        model = BaselineEvalModel().to(device).eval()
    else:
        model = RetinaMoonFusion().to(device).eval()
    print(f"[eval] built in {time.time()-t0:.1f}s", flush=True)
    if args.ckpt and not args.baseline:
        sd = torch.load(args.ckpt, map_location=device, weights_only=False)
        sd_model = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
        missing, unexpected = model.load_state_dict(sd_model, strict=False)
        print(f"[eval] loaded {args.ckpt}  (missing={len(missing)}, unexpected={len(unexpected)})",
              flush=True)
    else:
        print("[eval] NO checkpoint loaded — Q-Former + head are RANDOM (sanity baseline only).",
              flush=True)

    filter_a = _filter_args()
    # Use autocast only for the exp4 path (MoonViT was trained in bf16).
    # Baseline RetinaNet was trained in fp32; running it in bf16 here would
    # shift logits and bias the eval.
    if args.baseline:
        from contextlib import nullcontext
        autocast = nullcontext()
    else:
        autocast = torch.amp.autocast("cuda", dtype=torch.bfloat16)

    # Accumulators in baseline-compatible structure.
    gt_boxes_all = [[] for _ in range(num_label_types)]
    det_boxes = [[[] for _ in range(num_classes_list[nlt])] for nlt in range(num_label_types)]

    t0 = time.time()
    with torch.no_grad():
        for it, (clip, frame_targets) in enumerate(loader):
            if args.limit and it >= args.limit:
                break
            clip = clip.to(device, non_blocking=True)
            with autocast:
                out = model(clip)
            B, K, T, _ = out["logits"].shape
            assert B == 1, "eval expects batch_size=1"
            sigmoid_logits = out["logits"][0].float().sigmoid()    # [K, T, 184]
            decoded_boxes = out["boxes"][0].float()                 # [K, T, 4] (fp32 for NMS)

            for t in range(T):
                ft = frame_targets[t]
                if ft is None:
                    gt_boxes_np = np.zeros((0, 4), dtype=np.float32)
                    gt_per_head_mh = {h: np.zeros((0, n), dtype=np.float32)
                                      for h, n in zip(_HEAD_NAMES, num_classes_list[1:])}
                else:
                    gt_boxes_np = ft["boxes"].cpu().numpy().astype(np.float32)
                    gt_per_head_mh = {h: ft[h].cpu().numpy().astype(np.float32)
                                      for h in _HEAD_NAMES}

                # Per-label-type GT + per-class dets.
                cc = 0
                for nlt in range(num_label_types):
                    num_c = num_classes_list[nlt]
                    gt_boxes_all[nlt].append(
                        _frame_gt_for_label_type(nlt, num_c, gt_boxes_np, gt_per_head_mh)
                    )
                    frame_boxes = decoded_boxes[:, t, :]                       # [K, 4]
                    for cl_ind in range(num_c):
                        scores = sigmoid_logits[:, t, cc].clone().squeeze()    # [K]
                        cc += 1
                        cls_dets = filter_detections(filter_a, scores, frame_boxes)
                        det_boxes[nlt][cl_ind].append(cls_dets)

            if (it + 1) % 25 == 0:
                elapsed = time.time() - t0
                eta = elapsed * (n_clips - it - 1) / max(it + 1, 1)
                print(f"[eval] processed {it+1}/{n_clips}  "
                      f"elapsed={elapsed:.0f}s  ETA={eta/60:.0f} min", flush=True)

    print("[eval] running f-mAP evaluation ...", flush=True)
    t0 = time.time()
    mAP, ap_all, ap_strs = baseline_eval.evaluate(
        gt_boxes_all, det_boxes, all_classes, iou_thresh=0.5,
    )
    print(f"[eval] f-mAP eval took {time.time()-t0:.0f}s", flush=True)

    print("\n=== Per-class AP (frame-level, IoU=0.5) ===")
    for nlt in range(num_label_types):
        print(f"\n--- {_LABEL_TYPE_NAMES[nlt]} (mAP = {mAP[nlt]:.4f}%) ---")
        # Each ap_strs[nlt] entry: "class_name : num_pos : det_count : AP"
        for s in ap_strs[nlt]:
            print(f"  {s}")

    print("\n=== Summary mAP per label type ===")
    for nlt in range(num_label_types):
        print(f"  {_LABEL_TYPE_NAMES[nlt]:>10s}: {mAP[nlt]:.4f}%")

    if args.out:
        result = {
            "ckpt": args.ckpt or "untrained",
            "split": args.split,
            "stride": args.stride,
            "limit": args.limit,
            "n_clips_processed": min(args.limit, n_clips) if args.limit else n_clips,
            "per_label_type": {
                _LABEL_TYPE_NAMES[nlt]: {
                    "mAP_pct": float(mAP[nlt]),
                    "per_class_AP_pct": {
                        all_classes[nlt][i]: float(ap_all[nlt][i])
                        for i in range(num_classes_list[nlt])
                    },
                }
                for nlt in range(num_label_types)
            },
        }
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"[eval] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

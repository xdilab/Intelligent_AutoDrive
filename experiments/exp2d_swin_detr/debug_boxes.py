#!/usr/bin/env python3
"""Quick diagnostic: compare predicted boxes vs GT boxes on a few clips."""

import json
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(EXP_DIR))

import config as C
from model import FrozenDETRModel


def to_xyxy(boxes_cxcywh):
    cx, cy, w, h = boxes_cxcywh.unbind(-1)
    x1 = (cx - 0.5 * w).clamp(0.0, 1.0)
    y1 = (cy - 0.5 * h).clamp(0.0, 1.0)
    x2 = (cx + 0.5 * w).clamp(0.0, 1.0)
    y2 = (cy + 0.5 * h).clamp(0.0, 1.0)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou_np(box_a, box_b):
    """IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FrozenDETRModel(
        clip_model=C.CLIP_MODEL,
        clip_cache_dir=C.CLIP_CACHE_DIR,
        clip_cls_dim=C.CLIP_CLS_DIM,
        clip_patch_dim=C.CLIP_PATCH_DIM,
        clip_grid=C.CLIP_GRID,
        d_model=C.D_MODEL,
        head_sizes=C.HEAD_SIZES,
        clip_len=C.CLIP_LEN,
        num_queries=C.NUM_QUERIES,
        num_encoder_layers=C.NUM_ENCODER_LAYERS,
        num_decoder_layers=C.NUM_DECODER_LAYERS,
        nhead=C.NHEAD,
        dim_ffn=C.DIM_FFN,
        dropout=C.DROPOUT,
        n_deform_points=C.N_DEFORM_POINTS,
        encoder_n_levels=C.ENCODER_N_LEVELS,
        decoder_n_levels=C.FPN_LEVELS,
        backbone_name=C.BACKBONE,
        backbone_freeze_stages=C.BACKBONE_FREEZE_STAGES,
        fpn_in_channels=C.FPN_IN_CHANNELS,
    )
    ckpt = torch.load(str(Path(C.CKPT_DIR) / "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(device)
    model.eval()

    with open(C.ANNO_FILE) as f:
        anno_data = json.load(f)

    # Find first 3 val clips with GT annotations
    n_clips_done = 0
    for videoname, vdata in anno_data["db"].items():
        if "val" not in (vdata.get("split_ids") or []):
            continue
        ann_fids = sorted(
            [fid for fid, fd in vdata.get("frames", {}).items()
             if fd.get("annotated", 0) == 1],
            key=lambda x: int(x),
        )
        if len(ann_fids) < C.CLIP_LEN:
            continue

        clip_fids = ann_fids[:C.CLIP_LEN]

        # Count GT agents in this clip
        gt_count = 0
        for fid in clip_fids:
            annos = vdata["frames"][fid].get("annos", {})
            gt_count += len([a for a in annos.values() if isinstance(a, dict) and a.get("box")])
        if gt_count < 3:
            continue

        print(f"\n{'='*80}")
        print(f"Video: {videoname}, frames: {clip_fids}")
        print(f"{'='*80}")

        pil_frames = []
        for fid in clip_fids:
            img_path = Path(C.FRAMES_DIR) / videoname / f"{int(fid):05d}.jpg"
            pil_frames.append(Image.open(img_path).convert("RGB"))

        width, height = pil_frames[0].size
        print(f"Image size: {width}x{height}")

        with torch.no_grad():
            outputs = model(pil_frames)

        probs = {k: v.sigmoid() for k, v in outputs["pred_logits"].items()}
        agentness = probs["agentness"].squeeze(1)  # [N_queries]

        # Show agentness distribution
        print(f"\nAgentness distribution ({C.NUM_QUERIES} queries):")
        for thresh in [0.9, 0.7, 0.5, 0.3, 0.1, 0.01]:
            n = (agentness > thresh).sum().item()
            print(f"  > {thresh}: {n} queries")

        # Get top-K predictions
        topk = min(20, agentness.shape[0])
        top_scores, top_idx = agentness.topk(topk)

        # Look at frame 0 in detail
        t = 0
        fid = clip_fids[t]
        print(f"\n--- Frame {fid} (t={t}) ---")

        # GT boxes for this frame
        annos = vdata["frames"][fid].get("annos", {})
        gt_boxes_norm = []
        gt_labels = []
        for aid, anno in annos.items():
            if isinstance(anno, dict) and anno.get("box"):
                gt_boxes_norm.append(anno["box"])
                agent_ids = anno.get("agent_ids", [])
                gt_labels.append(f"agent_ids={agent_ids}")

        print(f"\nGT boxes ({len(gt_boxes_norm)}):")
        for i, (box, label) in enumerate(zip(gt_boxes_norm, gt_labels)):
            px_box = [box[0]*width, box[1]*height, box[2]*width, box[3]*height]
            area_frac = (box[2]-box[0]) * (box[3]-box[1])
            print(f"  GT {i}: norm={[f'{b:.4f}' for b in box]}, "
                  f"px=[{px_box[0]:.0f},{px_box[1]:.0f},{px_box[2]:.0f},{px_box[3]:.0f}], "
                  f"area={area_frac:.4f} ({label})")

        # Predicted boxes for frame 0
        pred_boxes_cxcywh = outputs["pred_boxes"][:, t, :]  # [N, 4]
        pred_boxes_xyxy = to_xyxy(pred_boxes_cxcywh)  # [N, 4] in [0,1]

        print(f"\nTop {topk} predictions (by agentness):")
        for rank in range(topk):
            qi = top_idx[rank].item()
            score = top_scores[rank].item()
            box = pred_boxes_xyxy[qi].cpu().numpy()
            cx, cy, w, h = pred_boxes_cxcywh[qi].cpu().numpy()
            area_frac = (box[2]-box[0]) * (box[3]-box[1])
            px_box = [box[0]*width, box[1]*height, box[2]*width, box[3]*height]

            # Best IoU with any GT
            best_iou = 0
            best_gt_idx = -1
            for gi, gt_box in enumerate(gt_boxes_norm):
                iou = box_iou_np(box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gi

            iou_str = f"IoU={best_iou:.3f} (GT {best_gt_idx})" if best_gt_idx >= 0 else "no GT"
            print(f"  Q{qi:3d}: agentness={score:.3f}, "
                  f"cxcywh=[{cx:.3f},{cy:.3f},{w:.3f},{h:.3f}], "
                  f"px=[{px_box[0]:.0f},{px_box[1]:.0f},{px_box[2]:.0f},{px_box[3]:.0f}], "
                  f"area={area_frac:.4f}, {iou_str}")

        # Summary: for each GT, what's the best matching prediction?
        print(f"\nGT → best prediction matching:")
        for gi, gt_box in enumerate(gt_boxes_norm):
            best_iou = 0
            best_qi = -1
            best_score = 0
            for qi in range(agentness.shape[0]):
                box = pred_boxes_xyxy[qi].cpu().numpy()
                iou = box_iou_np(box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_qi = qi
                    best_score = agentness[qi].item()
            px_gt = [gt_box[0]*width, gt_box[1]*height, gt_box[2]*width, gt_box[3]*height]
            px_pred = pred_boxes_xyxy[best_qi].cpu().numpy() if best_qi >= 0 else [0,0,0,0]
            px_pred = [px_pred[0]*width, px_pred[1]*height, px_pred[2]*width, px_pred[3]*height]
            match = "MATCH" if best_iou >= 0.5 else "MISS"
            print(f"  GT {gi}: best Q{best_qi} IoU={best_iou:.3f} agentness={best_score:.3f} [{match}]")
            print(f"    GT px: [{px_gt[0]:.0f},{px_gt[1]:.0f},{px_gt[2]:.0f},{px_gt[3]:.0f}]")
            print(f"    Pred px: [{px_pred[0]:.0f},{px_pred[1]:.0f},{px_pred[2]:.0f},{px_pred[3]:.0f}]")

        n_clips_done += 1
        if n_clips_done >= 3:
            break


if __name__ == "__main__":
    main()

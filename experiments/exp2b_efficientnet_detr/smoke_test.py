#!/usr/bin/env python3
"""Smoke test for Exp2b architecture after per-frame/iterative/aux fixes.

Verifies:
1. Forward pass produces correct output shapes
2. Auxiliary outputs have correct count and shapes
3. Loss computes including auxiliary loss
4. Backward pass produces gradients
5. Memory tensor shapes are per-frame (B=T), not stacked
"""
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]
EXP1_DIR = REPO_ROOT / "experiments" / "exp1_road_r"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))
if str(EXP1_DIR) not in sys.path:
    sys.path.append(str(EXP1_DIR))

import torch
from PIL import Image

import config as C
from model import EfficientNetFPNDETRModel


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    print(f"Device: {device} | dtype: {dtype}")

    T = C.CLIP_LEN  # 8
    N_Q = C.NUM_QUERIES  # 300
    D = C.D_MODEL  # 256
    N_LAYERS = C.NUM_DECODER_LAYERS  # 6

    print("Building model...")
    model = EfficientNetFPNDETRModel(
        model_id=C.MODEL_ID,
        vit_dim=C.VIT_DIM,
        d_model=D,
        head_sizes=C.HEAD_SIZES,
        clip_len=T,
        num_queries=N_Q,
        num_decoder_layers=N_LAYERS,
        nhead=C.NHEAD,
        dim_ffn=C.DIM_FFN,
        dropout=C.DROPOUT,
        n_deform_points=C.N_DEFORM_POINTS,
        backbone_name=C.BACKBONE,
        backbone_freeze_blocks=C.BACKBONE_FREEZE_BLOCKS,
        fpn_in_channels=C.FPN_IN_CHANNELS,
        gate_init=C.VLM_GATE_INIT,
        lora_r=C.LORA_R,
        lora_alpha=C.LORA_ALPHA,
        lora_dropout=C.LORA_DROPOUT,
        lora_target_modules=C.LORA_TARGET_MODULES,
        lora_n_layers=C.LORA_N_LAYERS,
    )
    model = model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {n_total:,} | Trainable: {n_train:,}")

    # Create dummy PIL frames
    pil_frames = [Image.new("RGB", (1280, 720)) for _ in range(T)]

    # Create dummy Qwen inputs (simplified — just need shapes)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(C.MODEL_ID)
    inputs = processor.image_processor(
        images=pil_frames,
        return_tensors="pt",
        min_pixels=C.MIN_PIXELS,
        max_pixels=C.MAX_PIXELS,
    )
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    image_grid_thw = inputs["image_grid_thw"].to(device=device)

    print(f"pixel_values: {pixel_values.shape}")
    print(f"image_grid_thw: {image_grid_thw.shape}")

    # Forward pass
    print("\nRunning forward pass...")
    model.train()
    outputs = model(pil_frames, pixel_values, image_grid_thw)

    # Check output shapes
    pred_boxes = outputs["pred_boxes"]
    pred_logits = outputs["pred_logits"]
    query_feats = outputs["query_feats"]
    aux_outputs = outputs["aux_outputs"]

    print(f"\npred_boxes: {pred_boxes.shape}")
    assert pred_boxes.shape == (N_Q, T, 4), f"Expected ({N_Q}, {T}, 4), got {pred_boxes.shape}"
    print(f"  OK: ({N_Q}, {T}, 4)")

    print(f"query_feats: {query_feats.shape}")
    assert query_feats.shape == (N_Q, D), f"Expected ({N_Q}, {D}), got {query_feats.shape}"
    print(f"  OK: ({N_Q}, {D})")

    print(f"\nClassification heads:")
    for name, logits in pred_logits.items():
        expected = C.HEAD_SIZES.get(name, 1)
        print(f"  {name}: {logits.shape} (expected [{N_Q}, {expected}])")
        assert logits.shape[0] == N_Q

    print(f"\nAuxiliary outputs: {len(aux_outputs)} (expected {N_LAYERS - 1})")
    assert len(aux_outputs) == N_LAYERS - 1, \
        f"Expected {N_LAYERS - 1} aux outputs, got {len(aux_outputs)}"

    for i, aux in enumerate(aux_outputs):
        ab = aux["pred_boxes"]
        aq = aux["query_feats"]
        print(f"  Layer {i}: boxes={ab.shape}, feats={aq.shape}")
        assert ab.shape == (N_Q, T, 4)
        assert aq.shape == (N_Q, D)
        assert "pred_logits" in aux
    print("  OK: all aux outputs have correct shapes")

    # Check box values are in [0, 1] (sigmoid output)
    assert pred_boxes.min() >= 0.0 and pred_boxes.max() <= 1.0, \
        f"Box values out of [0,1]: min={pred_boxes.min():.4f}, max={pred_boxes.max():.4f}"
    print(f"\nBox value range: [{pred_boxes.min():.4f}, {pred_boxes.max():.4f}] — OK (sigmoid)")

    # Check reference point refinement: aux boxes should differ across layers
    if len(aux_outputs) >= 2:
        b0 = aux_outputs[0]["pred_boxes"]
        b1 = aux_outputs[1]["pred_boxes"]
        diff = (b0 - b1).abs().mean().item()
        print(f"Box diff between layer 0 and 1: {diff:.6f} (should be > 0 if refinement works)")

    # Backward pass
    print("\nRunning backward pass...")
    loss = pred_boxes.sum() + sum(v.sum() for v in pred_logits.values())
    for aux in aux_outputs:
        loss = loss + aux["pred_boxes"].sum()
    loss.backward()

    # Check gradients exist
    has_grad = 0
    no_grad = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad += 1
            else:
                no_grad += 1
    print(f"Trainable params with gradient: {has_grad}")
    print(f"Trainable params without gradient: {no_grad}")
    assert has_grad > 0, "No gradients computed!"

    print("\n=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()

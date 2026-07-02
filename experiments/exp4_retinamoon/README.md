# Exp4 — 3D-RetinaNet + MoonViT Late Fusion

Late-fusion successor to the exp2 series. Frozen detector + frozen vision-language
encoder, trainable Q-Former + flat 184-dim head. Designed to fix the structural
limitations identified in [`findings/exp2-series-narrative` §Root Cause #4](../../wiki/findings/exp2-series-narrative.md):

| Problem in exp2c–2g | Why exp4 fixes it |
|---|---|
| CLIP@336 squashed vs R50@800×1333 (spatial mismatch) | MoonViT has 2D RoPE → handles any HxW natively. Smoke test uses SigLIP-SO400M (same arch) at 384px until the Kimi-VL vision_tower is staged. |
| Frozen DINO encoder can't adapt to CLIP-shaped 4th level | No encoder fusion. Detector and VLM operate independently; fusion happens *after* detection in a trainable Q-Former. |
| 300 queries × Hungarian → ~20 positives per clip | Detector contributes K=100 proposals (anchor-dense). All K tubes flow through the head; focal-on-all (exp2f's fix) handles unmatched. |

## Architecture

```
Clip [B, T=8, 3, 800, 1333]
       │
       ├─► Frozen 3D-RetinaNet ──► boxes [B, K, T, 4]
       │   (epoch 25 ckpt, 17.76%   scores [B, K]
       │    agent f-mAP)            feat_spatial [B, K, T, 256]
       │                                                │
       └─► Frozen MoonViT / SigLIP                      │
           feat_enc_grid [B, T, H_e, W_e, 1152]         │
                       │                                │
                       └─► RoIAlign at tube boxes ──► feat_enc [B, K, T, M=4, 1152]
                                                        │
                                                        ▼
                                               Q-Former (trainable)
                                               proj_spatial: 256 → 256
                                               proj_enc:    1152 → 256
                                               queries: [N_query=4, 256]
                                               2 × (self-attn + cross-attn + FFN)
                                                        │
                                               fused [B, K, T, 256]
                                                        │
                                                        ▼
                                               FlatHead (trainable)
                                               Linear(256 → 184)
                                                        │
                                               logits [B, K, T, 184]
                                               (agentness + agent + action + loc
                                                + duplex + triplet)
```

## Files

| File | Role |
|---|---|
| `config.py` | Paths, dims, encoder choice (SigLIP vs Kimi-VL gate) |
| `model.py` | `RetinaMoonFusion` = detector + encoder + Q-Former + head |
| `smoke_test.py` | One-clip forward pass; shape + grad-flow assertions |
| `README.md` | This file |

## Reproduce — smoke test

```bash
cd /data/repos/ROAD_Reason/experiments/exp4_retinamoon
CUDA_VISIBLE_DEVICES=1 python -u smoke_test.py
```

First run downloads `google/siglip-so400m-patch14-384` to `~/.cache/huggingface/`
(~800 MB). Subsequent runs are instant.

Expected output: `[smoke] ALL CHECKS PASSED` with peak GPU mem reported.

## Status

- [x] Config + model + smoke-test scaffolded (2026-06-02)
- [x] Smoke test on SigLIP-SO400M
- [ ] Replace mock detector with real frozen 3D-RetinaNet baseline
- [ ] Swap SigLIP → Kimi-VL vision_tower (needs ~3 GB free on /home or /data)
- [ ] Dataloader matching baseline VideoDataset conventions
- [ ] Training loop (flat-head focal-on-all, AdamW, 30 epochs, batch=1, grad_accum=4)
- [ ] First epoch eval vs 3D-RetinaNet baseline (target ≥ 17.76% agent f-mAP)

## Caveats

- **Smoke test uses a mock detector.** It emits random K boxes + random spatial
  features at the correct shapes. This validates fusion plumbing only; real
  detector integration is a separate milestone.
- **SigLIP ≠ MoonViT.** Same architecture family (SigLIP-SO400M is MoonViT's
  initialization) but MoonViT adds 2D RoPE + continual training. For real
  experiments we must swap.
- **K=100 from anchor NMS is a starting point.** May be raised or lowered after
  measuring recall on ROAD-Waymo val.

## Related

- `wiki/findings/exp2-series-narrative` — full motivation
- `wiki/papers/kimi-2025-moonvit` — MoonViT architecture
- `wiki/comparisons/fusion-for-detection-lit-review` — fusion design space
- `~/.claude/memory/project_clip_fusion_mismatch.md` — root cause this fixes

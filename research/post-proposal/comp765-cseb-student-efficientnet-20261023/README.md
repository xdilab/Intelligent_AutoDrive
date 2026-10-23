# COMP 765 EfficientNet-B0 student follow-up

Started October23,2026. Separate architecture follow-up to the completed ResNet student study. Frozen full EfficientNet-B0 features,1280 channels/stride32, shared448x672 input; same sceneGRU and semantic head widths, with input projections changed to1280. Same64train/12selection/198official videos,32-frame training sequences,30epochs, supervised BCE versus BCE+teacher BCE, seeds0–2. All three pairs will complete regardless of outcome.

Only current/past images enter the student. Reuse exact original candidate/label/teacher tensors, regenerate only small student feature caches. Full official detector AP@0.5, all86triplets and fixedTail47/Deep28/Common39; no crop-only substitution. Teacher/detector exposure and prior benchmark reuse remain disclosed in [PROVENANCE-ADDENDUM.md](PROVENANCE-ADDENDUM.md). Architecture changes include feature resolution, pretrained weights, channels and projection parameter counts; speed is measured, not inferred from parameter counts.

Protocol: [protocol.json](protocol.json). Run root: `/home/brandon/.local/share/road-reason/comp765-cseb-student-efficientnet-20261023/`. Existing studies and realtime bundle are preserved. No CV, slides or automatic remote push.

Use the existing road_reason environment with PYTHONPATH=/home/brandon/.local/share/road-reason/cseb-demo-runtime/packages, CUDA_VISIBLE_DEVICES=0 and PYTHONDONTWRITEBYTECODE=1. Commands and measured launch ETA are recorded with the run.

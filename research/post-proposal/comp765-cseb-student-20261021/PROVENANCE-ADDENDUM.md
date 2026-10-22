# Pre-evaluation provenance correction — October 21, 2026

**Read this together with the frozen protocol.** The original protocol and split manifest remain byte-for-byte intact. This addendum corrects an overly broad independence statement; it changes no videos, candidates, models, losses, seeds, checkpoint selection, or evaluation rule. It was recorded during target preparation, before student training and before any new student final AP.

The 198 official validation videos are disjoint from the CSEB **semantic-head** training/development/blend-selection pools and from student training/selection. However, all 198 were in the fixed shared YOLO detector's validation pool. All 36,717 scored official frames appear among its 37,932 validation images. Therefore, statements in the original protocol that final validation is disjoint from teacher/student selection must be read as referring to semantic heads only. They do **not** establish independence from the detector's selection history.

Evidence:

- Detector: `experiments/exp11_yolo/runs/v8x_agent_1280_eccv/weights/best.pt`, SHA256 recorded in `provenance.json`.
- Its `args.yaml` enables validation and points to `/data/datasets/road_waymo_yolo/data.yaml`, whose validation path is `images/val`.
- Checkpoint metadata records Ultralytics 8.3.226 and validation-derived fitness. The matching installed trainer updates `best_fitness` from validation and writes `best.pt` when fitness matches the best value. `results.csv` retains the validation metrics.
- Full video/frame intersections, metadata, source paths and hashes: `detector-history-audit.json` under the study output root.

The shared detector was trained on 115,602 frames from all 600 ROAD-Waymo training videos. Thus the new study’s 64-video training scope refers to the student semantic head, not to the total historical supervision used by the fixed detector. The ResNet50 backbone remains the existing frozen ImageNet model.

The official benchmark is retained because the user requested the established protocol where feasible and disclosure of prior research use. Both students use the identical fixed detector/candidates; no new final AP tunes any student choice. This remains a matched student-objective comparison on a historically reused validation benchmark. It is **not a pristine held-out test of the complete perception system**, and the paired-seed interval does not quantify uncertainty over new road videos.

A fully independent labeled test set is not established by these assets. Creating one or switching datasets would be a separate study decision, not a silent replacement for the requested official benchmark.

# Full-video mining and two-event context rules

New COMP765 extension, 2026-10-26. Prior pilot and student checkpoints remain immutable. Three fixed distilled ResNet50-layer3/scene-GRU students are evaluated under seven postprocessing conditions. No student retraining.

See protocol.json for the frozen design. Full conditions mine all annotated frames of the same64training videos; short conditions reproduce the pilot32-frame segments. Pair rules are incremental conjunction effects beyond either constituent single-event conditional. No new development or final-AP tuning is allowed.

Outputs: /home/brandon/.local/share/road-reason/comp765-student-association-context-20261026/

Status: implementation and preflight pending; no new evaluation results yet.

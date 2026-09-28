# Full Stage 5/6 encoder adaptation

Brandon authorized full data instead of further small training pilots on 2026-09-14. Six matched runs: Stage5/Stage6 × frozen encoder/classification partial unfreezing/classification+contrastive partial unfreezing.

Every paired-cache training crop from all 420 expert videos is used, including GT positives, 8 random negatives and up to16 YOLO-junk negatives per frame. All90 development videos select baseline versus3 full epochs;90 gate videos remain reserved. Same exact boxes, labels, frame bytes, model initialization, order and optimizer budget within each architecture. This warm-start study uses seed0 existing expert checkpoints. It is not a multi-seed conclusion.

Complete Stage5 composition input:49 primitive sigmoid scores+1024crop feature. Complete Stage6 input additionally includes135 phrase scores. MLP is512-ReLU-135. Output scores use flat primitives and MLP compositions. Train all downstream modules jointly; original frozen text prototypes. Auxiliary contrastive projection is separate from the Stage6 inference phrase branch and never becomes a Stage5 inference input. Last visual transformer block unfrozen only in adaptation conditions. AdamW visual LR1e-5, downstream LR1e-4, focal gamma2 and existing class weights, contrastive lambda.001. Full development and final inference use batches of up to64 crops under no_grad, matching the previously feasible inference batch size. Accumulate variable microbatches (max16) into >=256-crop optimization batches. No dropout; BF16 compute/FP32 parameters. Checkpoint every25 optimizer updates for resumption.

Required preflight is one gradient step per complete architecture, not another training pilot: source inference parity, MLP/visual gradients, Stage6 phrase-branch gradients, actual update and frozen hashes. Passed reports stay in the separately named stage56-encoder-pilot cluster directory; no six-arm small-data training was submitted.

Jobs: prepare exact rows + image hashes; full training array; dependent full YOLO evaluation array using the locked36,717-frame manifest, original IoU.5 evaluator, candidate confidence once, and tail47/deep28/common39. The final set never selects checkpoints. Final inference saves per-frame arrays for resumability. Results live on NCShare and the wiki, not in this code directory.

Run root /work/bbyrd1/stage56-full-20260914. Source code canonical home ROAD_Reason/research/post-proposal/stage56-full. Walltime requests are ceilings, not estimates. Estimate remaining time from actual full-data throughput.

## September 15 recovery

All original six training tasks failed during baseline development evaluation when the old shared staging frame directory disappeared. No training checkpoints had been saved. Cause of the directory disappearance is unknown. Restore uses intact local source images into this study's own `frames/` directory, selected by the original158,081-file inventory. `ROAD_FRAME_ROOT` overrides storage location without changing split, targets, crops, loss, or architecture. A CPU gate verifies all file sizes and every prepared train/dev frame hash before replacement jobs run.

Development evaluation now atomically caches per-frame targets and predictions with model/data/crop-code provenance. Resume tests verify identical AP, recomputation of only missing frames, and rejection of a changed model. Existing final-detector inference was already resumable. `recovery-launch.json` and active job files identify replacement arrays; collectors read job IDs dynamically.

## Fixed epoch-one detector evaluation (September 17)

User authorized using two spare GPUs for early reporting while all six training arms continue. `evaluate-epoch1.py` reuses the final evaluator's exact candidate hash,36,717 frames, score assembly, IoU0.5 metric and tail groups. It loads `epoch-1.pt` regardless of development selection, validates stage/condition/protocol/data and trainable-state keys, and saves checkpoint provenance. Predictions go to `predictions-epoch1/`, results to `epoch1-final-*.json`; final selection/evaluation is untouched. No tuning from official-validation results.

`submit-epoch1.py` is idempotent through saved job IDs. Two-task arrays run sequentially: Stage5 classification/contrastive, frozen controls, Stage6 classification/contrastive. The last pair also depends on a CPU-only checkpoint gate, so GPUs are not reserved while awaiting epoch1. `afterany` between pairs allows remaining evaluations to proceed if an earlier pair fails; the watchdog records failures. Every job name starts full56- and its ID is registered in results/*-job.txt. The two-GPU bound applies to this early-evaluation chain; the account QoS enforces8 total GPUs alongside training/final evaluation.

Before submission, check every temporal image required by shared-frames.json and preserve results/epoch1-input-check.json. Syntax checked and inference/metric body diff-reviewed against existing evaluator. Actual successful inference validates checkpoint loading at runtime; do not claim completed evaluation until result JSON exists.

### Evaluation input reuse (2026-09-18)

`eval_input.py` is shared by epoch-1 and final detector evaluation. It decodes
source JPEGs once into a bounded 24-frame LRU cache, prepares the next clip on
one CPU thread, normalizes each clip once on GPU, and reuses it across unchanged
64-candidate RoIAlign batches. Encoder execution, BF16 precision, candidate
ordering, confidence weighting, AP definitions and checkpoint provenance remain
unchanged. Atomic per-frame `.npy` caches still resume by skipping completed
frames. Source images are treated as immutable; decoded bytes are SHA-256 hashed
on read. This optimization does not alter `base.py` or running training.

Validation: `test_input_resume.py` checks skip/empty-frame/order/batch boundaries.
`benchmark_input.py` checks full-frame crop equality and same-checkpoint logit
parity on the cluster. The recorded run had zero crop error for 247 candidates
across three frames and zero logit error on eight candidates. Its preprocessing
timings are not end-to-end throughput measurements. Operational evidence is in
`wiki/artifacts/eval-input-optimization/` and the remote results directory.

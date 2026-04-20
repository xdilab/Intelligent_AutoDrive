# Experiment 1b: ROAD-R Dense Detection

This experiment is the first serious attempt at turning the VLM-backed classifier into a detector-like ROAD-Waymo model.

Compared with `exp1_road_r`, the model now:

- predicts foreground tokens instead of receiving GT boxes
- predicts box geometry with FCOS-style `l,t,r,b`
- uses a learned `agentness` score
- applies LoRA to the Qwen visual backbone
- uses focal loss to handle severe imbalance
- uses Gödel t-norm with real predicted agentness
- warm-starts from the best `exp1_road_r` checkpoint

This is the experiment that most plausibly explains why the VLM-based approach starts to beat the older 3D-RetinaNet + Gödel baseline on the project’s current evaluation setup.

Update after baseline-compatible evaluation:

That original interpretation was too optimistic. The internal eval remained useful as a semantic/foreground-token diagnostic, but the baseline-compatible frame-mAP evaluation showed that the current FCOS-on-Qwen detector stack does not beat the official 3D-RetinaNet + Gödel baseline.

The value of `exp1b_road_r` is therefore diagnostic:
- it suggests the Qwen backbone is learning useful semantics
- it shows that our custom dense localization stack is not yet competitive
- it motivates trying Qwen native grounding directly before moving to OpenMixer

## Goal

This experiment asks:

Can we keep the strong pretrained Qwen visual features from Experiment 1, then add just enough detection machinery and domain adaptation to produce stronger ROAD-Waymo structured predictions without relying on oracle GT boxes?

## GT Usage Clarification

This experiment does not use GT boxes the way `exp1_road_r` does, but it is still fully supervised.

The distinction is:

- In `exp1_road_r`, GT boxes are direct model inputs. The model receives GT boxes and pools features from those exact regions.
- In `exp1b_road_r`, the model forward pass does not receive GT boxes. It predicts densely from image tokens alone.
- After the forward pass, the training code uses GT boxes and GT labels to build FCOS-style supervision targets.

So `exp1b_road_r` is:

- not oracle-box inference
- not label-free training
- not unsupervised

It is best described as a dense supervised detection model:

1. the model sees only images at inference time
2. training still uses GT boxes and GT labels to supervise agentness, box regression, and the structured heads

This is why `exp1b_road_r` is a much fairer comparison to 3D-RetinaNet than `exp1_road_r`, but it is still a standard supervised detector rather than a self-supervised or weakly supervised method.

## Files

- [config.py](/data/repos/ROAD_Reason/experiments/exp1b_road_r/config.py): experiment hyperparameters
- [model.py](/data/repos/ROAD_Reason/experiments/exp1b_road_r/model.py): dense FCOS-style token predictor with LoRA-enabled Qwen ViT
- [assign.py](/data/repos/ROAD_Reason/experiments/exp1b_road_r/assign.py): FCOS token-to-GT assignment
- [losses.py](/data/repos/ROAD_Reason/experiments/exp1b_road_r/losses.py): agentness focal loss, box loss, focal classification, Gödel t-norm
- [train.py](/data/repos/ROAD_Reason/experiments/exp1b_road_r/train.py): training loop, warm-start, optimizer setup
- [eval.py](/data/repos/ROAD_Reason/experiments/exp1b_road_r/eval.py): evaluation and post-processing
- [logs/metrics.jsonl](/data/repos/ROAD_Reason/experiments/exp1b_road_r/logs/metrics.jsonl): training trace
- [logs/eval_results.json](/data/repos/ROAD_Reason/experiments/exp1b_road_r/logs/eval_results.json): saved evaluation summary

## High-Level Design

The model no longer extracts one feature per GT box.

Instead, every spatial token in the Qwen feature map predicts:
- whether it corresponds to an agent
- the box around that agent
- the structured ROAD-Waymo labels attached to that agent

That makes the model much closer in spirit to one-stage dense detectors like FCOS and RetinaNet.

## Inputs and Outputs

### Dataset input

This experiment reuses the same clip dataset as Experiment 1 via the shared [dataset.py](/data/repos/ROAD_Reason/experiments/exp1_road_r/dataset.py):

- 8 RGB frames per clip
- per-frame GT boxes and multi-hot structured labels

### Model input

The forward interface is:

```python
preds = model(pixel_values, image_grid_thw)
```

Unlike `exp1`, there is no `frame_boxes_list` input to the model itself.

That matters because:
- GT boxes are used only to create training targets
- the model learns dense token-level prediction rather than oracle-box classification

### Model output

The model returns:

- `agentness`: `[N, 1]`
- `box`: `[N, 4]`
- `agent`: `[N, 10]`
- `action`: `[N, 22]`
- `loc`: `[N, 16]`
- `duplex`: `[N, 49]`
- `triplet`: `[N, 86]`
- `token_counts`: token count per frame
- `frame_shapes`: spatial token grid shape per frame

Where `N = sum_t H'_t * W'_t`, the total number of spatial tokens across the clip.

## Data Flow

The full training pipeline is:

1. Load an 8-frame clip and its GT annotations.
2. Process the frames with the Qwen image processor.
3. Extract dense spatial token maps from the Qwen visual encoder.
4. Flatten all frame tokens.
5. Predict:
   - foreground probability
   - box distances
   - multi-label structured outputs
6. Build FCOS-style training targets by assigning each token to a GT box if its center lies inside the box.
7. Compute the 4-term loss.

In compact form:

```text
8 RGB frames
-> Qwen image processor
-> Qwen ViT token maps
-> flatten all tokens
-> per-token dense heads
-> FCOS token assignment from GT boxes
-> focal agentness + box + focal labels + Gödel t-norm
```

At inference/eval time, the intended post-processing path is:

```text
dense token predictions
-> threshold by agentness
-> decode FCOS boxes
-> NMS
-> keep surviving detections
-> read structured label heads on those detections
```

## Module Breakdown

### 1. QwenViTExtractor with LoRA

Defined in [model.py](/data/repos/ROAD_Reason/experiments/exp1b_road_r/model.py).

This module:
- loads the same Qwen2.5-VL visual encoder
- initially freezes it
- warm-loads weights from `exp1`
- inserts LoRA adapters into the first 8 attention blocks

The LoRA target modules are:
- `qkv`
- `proj`

Why this matters:
- the experiment keeps most pretrained knowledge
- only a small number of extra parameters adapt to ROAD-Waymo
- adaptation is cheaper and less unstable than fully unfreezing the backbone

### 2. Dense detection heads

Instead of ROI pooling, each token gets its own predictions:

- `agentness`: binary foreground score
- `box`: normalized FCOS `l,t,r,b`
- `agent`, `action`, `loc`, `duplex`, `triplet`: multi-label sigmoids

This is the main architectural jump from `exp1`.

### 3. FCOS-style assignment

Defined in [assign.py](/data/repos/ROAD_Reason/experiments/exp1b_road_r/assign.py).

For each token center:
- if it falls inside a GT box, it becomes foreground
- if it falls inside multiple boxes, it is assigned to the smallest one
- otherwise it is background

Foreground tokens inherit:
- box regression targets
- agent/action/loc/duplex/triplet labels

This is important because it replaces anchor engineering with a simple geometric rule.

It also makes the GT supervision path explicit:
- the model does not ingest GT boxes directly
- GT boxes are used after forward to decide which tokens are foreground and what targets they should regress/classify toward

### 4. Loss

Defined in [losses.py](/data/repos/ROAD_Reason/experiments/exp1b_road_r/losses.py).

The loss is:

```text
L_total = L_agentness + lambda_box * L_box + L_focal + L_tnorm
```

Where:
- `L_agentness`: focal loss on all tokens
- `L_box`: SmoothL1 on foreground tokens only
- `L_focal`: focal multi-label loss on foreground tokens only
- `L_tnorm`: Gödel consistency loss on foreground tokens only

This is a much better fit for dense prediction than `exp1` because:
- most tokens are background
- foreground classes are imbalanced
- logical consistency needs to act on predicted detections, not oracle instances

### 5. Warm-start from Experiment 1

This is one of the most important design decisions.

The code warm-loads:
- the ViT weights
- the structured classification heads

Freshly initialized:
- `agentness`
- `box`

Ignored from `exp1` because they no longer exist here:
- ROI pooling module
- tube-linking module

This means `exp1b` is not starting from scratch. It inherits a representation that already knows something about ROAD-Waymo label structure.

## Training Setup

From [config.py](/data/repos/ROAD_Reason/experiments/exp1b_road_r/config.py):

- Backbone: `Qwen/Qwen2.5-VL-7B-Instruct`
- Clip length: 8
- LoRA rank: 8
- LoRA alpha: 16
- LoRA dropout: 0.05
- LoRA layers: first 8 ViT blocks
- LoRA LR: `5e-5`
- Head LR: `1e-4`
- Warmup: 500 steps
- Max epochs: 15
- t-norm: `godel`
- `lambda_tnorm = 1.0`
- `lambda_box = 1.0`

The optimizer uses two parameter groups:
- LoRA adapters
- dense heads

That is a smart setup because the pretrained backbone should move cautiously while the new heads need to learn quickly.

## Recorded Results

### Training trace

From [logs/metrics.jsonl](/data/repos/ROAD_Reason/experiments/exp1b_road_r/logs/metrics.jsonl):

The file contains two regimes:

1. An older early run with only `L_focal` and `L_tnorm`
2. The current redesign with:
   - `L_agentness`
   - `L_box`
   - `L_focal`
   - `L_tnorm`

For the current redesign:
- Validation action mAP rises from `0.2588` at epoch 1 to `0.3242` at epoch 15
- Validation total loss drops from `0.1034` to roughly `0.0828`
- Best checkpoint is selected by validation action macro-mAP

This is a healthier signal than `exp1`, because the model is learning localization-related behavior and class prediction jointly.

### Evaluation summary

From [logs/eval_results.json](/data/repos/ROAD_Reason/experiments/exp1b_road_r/logs/eval_results.json):

- Agent mAP: `0.6055`
- Action mAP: `0.3242`
- Location mAP: `0.4998`
- Duplex mAP: `0.2310`
- Triplet mAP: `0.1747`
- Macro-F1:
  - Agent: `0.5770`
  - Action: `0.3235`
  - Loc: `0.4589`
  - Duplex: `0.2013`
  - Triplet: `0.1624`
- Constraint violation rate: `0.002879`
- Average detections per frame at agentness threshold 0.5: `20.21`

### Why the internal eval still matters

The internal eval in [eval.py](/data/repos/ROAD_Reason/experiments/exp1b_road_r/eval.py) is not useless. It measures something different:

- quality of label prediction on GT-aligned foreground tokens
- semantic usefulness of the Qwen + LoRA representation
- whether the structured heads are learning ROAD-Waymo categories at all

So the internal eval is best treated as a semantic or GT-aligned recognition diagnostic, not as the final detector score.

## Comparison to Experiment 1

Using the saved eval JSONs:

| Head | Exp1 mAP | Exp1b mAP | Gain |
|---|---:|---:|---:|
| Agent | 0.3567 | 0.6055 | +0.2488 |
| Action | 0.2223 | 0.3242 | +0.1019 |
| Loc | 0.3356 | 0.4998 | +0.1642 |
| Duplex | 0.1228 | 0.2310 | +0.1082 |
| Triplet | 0.0876 | 0.1747 | +0.0871 |

This is a broad improvement, not a narrow single-head win.

That pattern strongly suggests the redesign helped the underlying representation rather than just overfitting one metric.

## Comparison to 3D-RetinaNet + Gödel Baseline

From [analysis/baseline_val_metrics.csv](/data/repos/ROAD_Reason/analysis/baseline_val_metrics.csv), the baseline reports:

- Agent: `17.01%`
- Action: `15.21%`
- Loc: `13.44%`
- Duplex: `13.62%`
- Triplet: `9.37%`

Important caution:

This is not a strict apples-to-apples comparison.

Why not:
- the baseline uses official frame-level detection mAP at IoU 0.5
- `exp1b` evaluation explicitly notes that its current script does not do proper IoU-based detection matching
- the current `exp1b` eval measures classification quality on GT-assigned foreground tokens, then also reports detection counts separately

So the safe claim is:
- `exp1b` clearly beats the baseline on the current internal experiment harness
- `exp1b` is promising for end-to-end detection
- but it has not yet been validated with the exact official baseline metric protocol

## Why Experiment 1b Likely Beat the Baseline

This is the most important section.

### 1. Much stronger pretrained visual representation

The 3D-RetinaNet baseline is built around older video-detection features. `exp1b` starts from Qwen2.5-VL, a large modern vision-language model whose visual encoder already contains broad semantic knowledge.

That likely helps especially for:
- traffic light state recognition
- fine-grained agent categories
- visually subtle motion cues
- long-tail semantics

### 2. Warm-start from a task-adapted structured classifier

`exp1b` does not start cold.

It inherits:
- ViT weights already adapted to the data pipeline
- structured classification heads already exposed to ROAD-Waymo labels

This reduces optimization burden compared with training a dense detector from scratch.

### 3. LoRA gives controlled domain adaptation

Instead of keeping the ViT fully frozen like `exp1`, `exp1b` allows the first 8 blocks to adapt.

That is probably one of the biggest reasons for the jump:
- enough flexibility to adapt to driving scenes
- not so much freedom that the backbone forgets its pretrained knowledge

### 4. Dense token supervision is richer than oracle ROI pooling

Every spatial token inside an object can become a supervised foreground site.

That gives the model:
- more supervised positions
- better spatial grounding
- a path toward real detection

This is conceptually closer to RetinaNet and FCOS than `exp1` was.

### 5. Focal loss is a better answer to imbalance

ROAD-Waymo is heavily imbalanced:
- most tokens are background
- many semantic classes are rare

`exp1b` explicitly addresses this with:
- agentness focal loss
- per-class alpha-weighted focal classification

That is a much better fit than plain BCE for dense prediction.

### 6. Gödel t-norm now operates on real predicted detections

In `exp1`, t-norm receives a fake `agentness = 1.0` because every instance is a GT box.

In `exp1b`, the loss uses:
- predicted agentness
- predicted agent/action/location scores

That makes the structural regularization more realistic and more tightly coupled to actual detection behavior.

### 7. Anchor-free FCOS assignment may fit this setup better

The original baseline uses anchor-based detection. `exp1b` uses center-inside-box token assignment.

Potential benefits:
- simpler supervision
- less anchor tuning
- more direct spatial learning
- cleaner alignment with transformer-style token grids

## Why the Constraint Violation Rate Went Up

This is actually not surprising.

Violation rate:
- `exp1`: `0.000211`
- `exp1b`: `0.002879`

Why it increased:
- `exp1` classifies only GT-pooled instances, so the task is cleaner
- `exp1b` makes many more dense predictions in a harder setting
- learned agentness and dense token classification create more opportunities for inconsistent outputs

Despite the increase, the payoff is large performance gains across all heads. So this looks like a reasonable tradeoff, not a failure.

## Limitations

There are still important limitations before claiming full baseline replacement.

### 1. Current eval is not official detection mAP

This is the biggest one.

The current [eval.py](/data/repos/ROAD_Reason/experiments/exp1b_road_r/eval.py) itself says:
- it does not do proper IoU-based detection matching
- it evaluates label heads on GT-assigned foreground tokens

So the model may be better than the baseline on representation quality while still needing official detection validation.

### 2. Post-processing path is only partially used in scoring

The code has:
- agentness thresholding
- FCOS box decoding
- NMS

But the main classification metrics are still gathered on GT-assigned foreground tokens rather than NMS-surviving matched detections.

### 3. Rare classes are still hard

Some rare actions and rare agent types remain weak, even though overall performance improved.

## Main Interpretation

`exp1b` wins because it combines four things the baseline does not combine in the same way:

1. a much stronger pretrained visual encoder
2. controlled backbone adaptation via LoRA
3. dense token-level supervision with a learned foreground mechanism
4. imbalance-aware optimization with focal loss plus structural regularization

That is the likely explanation for why it beats the 3D-RetinaNet + Gödel baseline on the current internal evaluation.

The most honest summary is:

`exp1b` is a stronger model family than the baseline, but the repository still needs an official IoU-matched detection evaluation to prove the win under the exact same benchmark protocol.

## Conclusion

`exp1b_road_r` is the first experiment in this repo that meaningfully resembles an end-to-end detector rather than a GT-box classifier. It improves every major structured head over `exp1`, and it likely outperforms the older 3D-RetinaNet + Gödel system because it has better pretrained features, better adaptation, better imbalance handling, and a denser supervision signal.

The next scientific step is not another architecture jump. It is to evaluate this model with the same official detection protocol as the baseline so the performance claim becomes fully defensible.

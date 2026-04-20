# Experiment 1: ROAD-R

This experiment treats ROAD-Waymo as a clip-level, oracle-box classification problem built on top of the Qwen2.5-VL vision encoder. It is the simplest bridge between the original 3D-RetinaNet baseline world and the later VLM-based experiments in this repo.

The key idea is:

1. Use the pretrained Qwen2.5-VL visual backbone as a strong feature extractor.
2. Use ground-truth boxes instead of learning detection.
3. Pool one feature per annotated agent.
4. Add cross-frame context with a small attention module.
5. Predict ROAD-Waymo labels with multi-label heads.
6. Regularize agent-action-location consistency with the t-norm loss.

Because ground-truth boxes are provided at both train and eval time, this experiment is best understood as a structured recognition model, not a full detector.

## Goal

This experiment answers a narrow question:

Can a strong pretrained VLM visual encoder plus lightweight structured heads outperform older handcrafted video detection features for ROAD-Waymo-style label prediction when localization is made easy by giving the model the ground-truth boxes?

That is why the model does not learn agentness or box regression here. Those parts are deferred to `exp1b`.

## Files

- [config.py](/data/repos/ROAD_Reason/experiments/exp1_road_r/config.py): paths, label dimensions, training hyperparameters
- [dataset.py](/data/repos/ROAD_Reason/experiments/exp1_road_r/dataset.py): clip sampling and label tensor construction
- [model.py](/data/repos/ROAD_Reason/experiments/exp1_road_r/model.py): frozen Qwen ViT, ROI pooling, tube linking, classification heads
- [losses.py](/data/repos/ROAD_Reason/experiments/exp1_road_r/losses.py): BCE loss plus t-norm constraint loss
- [train.py](/data/repos/ROAD_Reason/experiments/exp1_road_r/train.py): training loop
- [eval.py](/data/repos/ROAD_Reason/experiments/exp1_road_r/eval.py): evaluation script
- [logs/metrics.jsonl](/data/repos/ROAD_Reason/experiments/exp1_road_r/logs/metrics.jsonl): per-epoch training and validation losses
- [logs/eval_results.json](/data/repos/ROAD_Reason/experiments/exp1_road_r/logs/eval_results.json): saved evaluation summary

## Inputs and Outputs

### Dataset input

Each sample is an 8-frame annotated clip.

Input to the dataset:
- ROAD-Waymo annotation JSON from `road_waymo_trainval_v1.1.json`
- RGB frame directory

Per sample output from the dataset:
- `pil_frames`: list of 8 RGB PIL images
- `frame_targets`: list of 8 per-frame dictionaries, each containing:
  - `boxes`: `[n_t, 4]` normalized bounding boxes
  - `agent`: `[n_t, 10]` multi-hot labels
  - `action`: `[n_t, 22]` multi-hot labels
  - `loc`: `[n_t, 16]` multi-hot labels
  - `duplex`: `[n_t, 49]` multi-hot labels
  - `triplet`: `[n_t, 86]` multi-hot labels

Important detail:
- Triplets are reconstructed from `(agent, action, loc)` names instead of trusting raw `triplet_ids` directly. That logic lives in [dataset.py](/data/repos/ROAD_Reason/experiments/exp1_road_r/dataset.py).

### Model input

The model forward interface is:

```python
preds = model(pixel_values, image_grid_thw, frame_boxes_list)
```

Where:
- `pixel_values` comes from the Qwen image processor
- `image_grid_thw` describes the per-frame token grid
- `frame_boxes_list` is the list of ground-truth boxes for each frame

### Model output

The model returns a dictionary of predictions for all annotated agents in the clip:

- `agent`: `[N, 10]`
- `action`: `[N, 22]`
- `loc`: `[N, 16]`
- `duplex`: `[N, 49]`
- `triplet`: `[N, 86]`

Where `N` is the total number of annotated agents across the clip.

There is no:
- agentness head
- box regression head
- NMS
- learned detection stage

## Data Flow

The full data flow is:

1. `ROADWaymoDataset` samples an 8-frame clip and loads images plus GT labels.
2. `preprocess_clip()` uses the Qwen processor to convert frames into visual patch inputs.
3. `QwenViTExtractor` runs the Qwen2.5-VL visual encoder on each frame independently.
4. `ROIAveragePool` uses each GT box to average the tokens that fall inside that box.
5. `TubeLinkingModule` concatenates all pooled agent features across frames and applies multi-head self-attention.
6. `ClassificationHeads` predicts multi-label probabilities for agent, action, location, duplex, and triplet.
7. `ROADLoss` computes:
   - BCE classification loss across all five heads
   - t-norm consistency loss using agent/action/location probabilities

In compact form:

```text
8 RGB frames
-> Qwen image processor
-> Qwen ViT token maps
-> ROI average pooling using GT boxes
-> per-agent features across frames
-> tube-link self-attention
-> 5 structured sigmoid heads
-> BCE + t-norm loss
```

## Module Breakdown

### 1. QwenViTExtractor

Defined in [model.py](/data/repos/ROAD_Reason/experiments/exp1_road_r/model.py).

What it does:
- Loads `Qwen/Qwen2.5-VL-7B-Instruct`
- Keeps only `full.model.visual`
- Discards the language model
- Freezes the visual encoder in this experiment

Why this matters:
- The experiment is transfer learning, not training a VLM from scratch.
- Most representational power comes from pretrained visual semantics.

### 2. ROI average pooling

Instead of asking the model to localize an agent, the code directly uses the GT box and averages all visual tokens that overlap that region.

This means:
- localization difficulty is removed
- the experiment isolates representation quality for classification
- results should be interpreted as an upper bound relative to a detector that must first find the object

### 3. TubeLinkingModule

This is a small cross-frame reasoning block:
- all agent features from all frames are concatenated
- one self-attention layer lets them exchange context

Why it exists:
- a single frame may not be enough to infer motion-related actions
- nearby frames help disambiguate `Stop`, `MovAway`, `Brake`, turn indicators, etc.

### 4. Classification heads

Five independent sigmoid heads predict:
- agent
- action
- loc
- duplex
- triplet

This is multi-label prediction, not single-class softmax.

That matters because a single instance can legitimately have:
- multiple actions
- multiple valid composites

### 5. Loss

The loss in [losses.py](/data/repos/ROAD_Reason/experiments/exp1_road_r/losses.py) is:

```text
L_total = L_cls + L_tnorm
```

Where:
- `L_cls` is BCE across the five heads
- `L_tnorm` penalizes logically invalid agent-action-location combinations

Important implementation detail:
- the t-norm module expects an `agentness` slot, but this experiment does not predict agentness
- so the code sets `agentness = 1.0` for every instance
- this is reasonable here because every pooled feature already comes from a known GT object

## Training Setup

From [config.py](/data/repos/ROAD_Reason/experiments/exp1_road_r/config.py):

- Backbone: `Qwen/Qwen2.5-VL-7B-Instruct`
- Clip length: 8 frames
- Batch size: 1 clip
- Learning rate: `2e-4`
- Weight decay: `0.01`
- Warmup: 200 steps
- Max epochs: 10
- t-norm: `lukasiewicz`
- `lambda_tnorm = 0.1`

Only the lightweight parts are trained:
- tube-link module
- classification heads

The ViT is frozen.

## Recorded Results

From [logs/metrics.jsonl](/data/repos/ROAD_Reason/experiments/exp1_road_r/logs/metrics.jsonl):

- Best validation total loss occurs at epoch 6: `0.40035`
- Training loss steadily decreases from `0.4390` to `0.3007`
- Validation loss improves early, then plateaus and slightly worsens

Interpretation:
- the small trainable heads fit the task quickly
- the frozen visual features provide a useful starting point
- most gains happen in the first half of training

From [logs/eval_results.json](/data/repos/ROAD_Reason/experiments/exp1_road_r/logs/eval_results.json):

- Agent mAP: `0.3567`
- Action mAP: `0.2223`
- Location mAP: `0.3356`
- Duplex mAP: `0.1228`
- Triplet mAP: `0.0876`
- Constraint violation rate: `0.000211`

Macro-F1:
- Agent: `0.3209`
- Action: `0.1709`
- Location: `0.2652`
- Duplex: `0.0703`
- Triplet: `0.0398`

## What These Results Mean

This experiment is strongest on:
- coarse agent identity
- frequent action classes
- traffic-light states

It struggles most on:
- rare classes
- fine composite labels
- full triplets

That pattern is expected because:
- composites are harder than base labels
- rare classes have fewer positive examples
- the model never learns localization, so errors are purely representational/classification errors

## Comparison to the 3D-RetinaNet + Gödel Baseline

The baseline numbers in [analysis/baseline_val_metrics.csv](/data/repos/ROAD_Reason/analysis/baseline_val_metrics.csv) are:

- 3D-RetinaNet-I3D + Gödel
  - Agent: `17.01`
  - Action: `15.21`
  - Loc: `13.44`
  - Duplex: `13.62`
  - Triplet: `9.37`

Important caveat:
- those baseline values are frame-level detection mAP percentages at IoU 0.5
- this experiment reports macro classification metrics on GT-box-conditioned instances
- so this is not a strict apples-to-apples benchmark comparison

The fair interpretation is:
- this experiment suggests the Qwen visual representation is useful for ROAD-Waymo semantics
- but it does not yet prove better end-to-end detection than the official baseline

## Strengths

- Strong pretrained visual backbone
- Simple and easy to debug
- Good for isolating label prediction quality
- Very low logical violation rate
- Useful warm-start source for later experiments

## Weaknesses

- Uses GT boxes at inference time
- Does not measure full detection performance
- Frozen ViT limits domain adaptation
- Rare classes remain difficult
- Duplex/triplet performance is still modest

## Main Interpretation

Experiment 1 is a representation probe and a training scaffold.

Its real value is not that it solves ROAD-Waymo end to end. Its value is that it shows:

1. Qwen2.5-VL visual features can support structured ROAD-Waymo classification.
2. Tube-level context helps organize per-agent features across time.
3. t-norm constraints can be layered on top of VLM-derived features cleanly.
4. A strong warm-start can be created before introducing learned detection and LoRA.

## Conclusion

`exp1_road_r` is best viewed as the oracle-localization stage of the project.

It removes detection difficulty and asks whether a pretrained VLM vision backbone can support structured ROAD-Waymo label prediction. The answer is yes, but with a ceiling: once GT boxes are removed, the project needs learned localization, stronger class-imbalance handling, and backbone adaptation. Those are exactly the changes introduced in `exp1b_road_r`.

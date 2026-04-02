# ROAD_Reason

Logic-Constrained Scene Understanding and Reasoning for Autonomous Driving

Built on the ROAD-Waymo (ROAD++) dataset. Primary approach: generative VLM with neuro-symbolic t-norm constraints producing structured scene labels and natural language reasoning.

See [`APPROACHES.md`](APPROACHES.md) for the full research roadmap and [`ROAD_plusplus_summary.md`](ROAD_plusplus_summary.md) for dataset documentation.

---

## Setup

```bash
conda env create -f environment.yml
conda activate road_reason
```

## Baselines

Two zero-shot VLM baselines using SmolVLM-500M-Instruct on ROAD-Waymo val frames.

### Zero-shot (flat label lists)

Prompts the model with the raw agent/action/location label sets and asks for JSON output.

```bash
python baseline/smolvlm_inference.py
# options: --n_videos 20 --frames_per_video 10 --model HuggingFaceTB/SmolVLM-Instruct
```

Output: `baseline/results/smolvlm_preds.json`

### Constraint-aware (all 135 valid labels baked in)

Injects all **49 valid duplexes** (agent+action) and **86 valid triplets** (agent+action+location) directly into the prompt. The model must copy triplets verbatim from the constraint list — no invalid combinations possible in compliant output.

```bash
python baseline/smolvlm_constrained.py
# options: --n_videos 20 --frames_per_video 10 --model HuggingFaceTB/SmolVLM-Instruct
```

Output: `baseline/results/constrained_preds.json`

### GT-conditioned reasoning

Feeds verified ground-truth triplets alongside the image and asks the model to produce natural language scene reasoning and intent summary. Isolates the reasoning capability from detection — the model is told what is present and must explain why and what agents are likely to do next. Closest proxy to the Approach 3 thesis contribution.

```bash
python baseline/smolvlm_gt_reasoning.py
# --prefer_ped (default) biases frame selection toward pedestrian scenes
# --no_prefer_ped to sample uniformly
```

Output: `baseline/results/gt_reasoning_preds.json`

### Evaluation

Scores both formats against ground truth. Reports precision/recall/F1 for agent, action, and location label sets, plus **constraint violation rate** for the constrained run.

```bash
python baseline/eval_preds.py --preds baseline/results/smolvlm_preds.json
python baseline/eval_preds.py --preds baseline/results/constrained_preds.json
```

### Dataset paths

| Resource | Path |
|---|---|
| Annotation JSON (~1 GB) | `/data/datasets/ROAD_plusplus/road_waymo_trainval_v1.0.json` |
| Extracted frames | `/data/datasets/ROAD_plusplus/rgb-images/{video_name}/{frame:05d}.jpg` |
| Videos | `/data/datasets/ROAD_plusplus/videos/` |

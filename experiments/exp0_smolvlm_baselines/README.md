# Experiment 0: SmolVLM Prompt Baselines

This folder contains the prompt-only Stage-0 baselines used before the trainable Qwen experiments.

## Files

- [smolvlm_inference.py](/data/repos/ROAD_Reason/experiments/exp0_smolvlm_baselines/smolvlm_inference.py)
  - Zero-shot structured label prediction from flat ROAD-Waymo label lists.
- [smolvlm_constrained.py](/data/repos/ROAD_Reason/experiments/exp0_smolvlm_baselines/smolvlm_constrained.py)
  - Prompt-constrained prediction using only valid ROAD-Waymo duplexes and triplets.
- [smolvlm_gt_reasoning.py](/data/repos/ROAD_Reason/experiments/exp0_smolvlm_baselines/smolvlm_gt_reasoning.py)
  - Ground-truth-conditioned reasoning baseline.
- [eval_preds.py](/data/repos/ROAD_Reason/experiments/exp0_smolvlm_baselines/eval_preds.py)
  - Offline evaluator for saved prompt-baseline JSON outputs.

## Recommended order

1. `smolvlm_inference.py`
2. `smolvlm_constrained.py`
3. `smolvlm_gt_reasoning.py`

## Example commands

```bash
python -u experiments/exp0_smolvlm_baselines/smolvlm_inference.py
python -u experiments/exp0_smolvlm_baselines/smolvlm_constrained.py
python -u experiments/exp0_smolvlm_baselines/smolvlm_gt_reasoning.py
python experiments/exp0_smolvlm_baselines/eval_preds.py --preds experiments/exp0_smolvlm_baselines/results/smolvlm_preds.json
```


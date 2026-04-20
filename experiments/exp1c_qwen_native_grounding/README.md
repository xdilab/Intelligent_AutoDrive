# Experiment 1c: Qwen Native Grounding

This experiment tests Qwen2.5-VL's own prompted grounding behavior directly, instead of adding a custom dense detector head.

It exists to answer the question left open by `exp1b_road_r`:

Is the localization weakness coming from Qwen itself, or from our FCOS-on-token adaptation?

## Files

- [qwen_native_grounding.py](/data/repos/ROAD_Reason/experiments/exp1c_qwen_native_grounding/qwen_native_grounding.py)
  - Prompted Qwen2.5-VL grounding baseline that returns JSON detections with boxes and ROAD-Waymo labels.
- [eval_qwen_grounding.py](/data/repos/ROAD_Reason/experiments/exp1c_qwen_native_grounding/eval_qwen_grounding.py)
  - Lightweight evaluator for parse rate, label quality, and rough box-overlap quality.

## Why it matters

`exp1b_road_r` showed that Qwen features plus a custom FCOS-style detector stack do not beat the official 3D-RetinaNet baseline on the baseline-compatible frame-mAP evaluator.

That still leaves an important question unanswered:

Can Qwen2.5-VL localize ROAD-Waymo agents better when used in the native grounding style described in its technical report?

If the answer is yes, then the FCOS adaptation was the main problem.
If the answer is still no, then moving to OpenMixer becomes even more justified.

## Example commands

```bash
python -u experiments/exp1c_qwen_native_grounding/qwen_native_grounding.py
python experiments/exp1c_qwen_native_grounding/eval_qwen_grounding.py --preds experiments/exp1c_qwen_native_grounding/results/qwen_native_grounding.json
```


# CSEB inference runner

`deployment.py` executes fresh YOLO detection, the verified 1B visual encoder and both trained experts in one process. Models stay loaded between clips; crop/RoI/scene features stay on GPU until scoring finishes. It uses the existing trained weights and fixed blend. This is an inference implementation, not a distilled or retrained architecture.

## Run a video

From this directory on the configured workstation:

```bash
PYTHONPATH=/home/brandon/.local/share/road-reason/cseb-demo-runtime/packages \
CUDA_VISIBLE_DEVICES=0 \
/home/brandon/miniconda3/envs/road_reason/bin/python deployment.py \
  --video /path/to/input.mp4 \
  --checkpoint /home/brandon/.local/share/road-reason/contextual-roi-1b-20260930/attention-global-dcb-seed0/best.pt \
  --encoder-root /home/brandon/.local/share/road-reason/cseb-demo-runtime \
  --yolo /data/repos/ROAD_Reason/experiments/exp11_yolo/runs/v8x_agent_1280_eccv/weights/best.pt \
  --output /path/to/new-inference-output --frames 120 --crop-batch 8
```

The output directory must not exist. Outputs:

- `predictions.jsonl`: one frame per line, containing normalized boxes, detector confidence, agent IDs and all184 detector-weighted scores for classification, contrastive and blend outputs.
- `report.json`: label ordering, source/input hashes, detector configuration, load time, per-frame timings, GPU and peak allocated memory.
- `progress.json`: atomic phase/progress record.
- Optional `--save-features`: per-frame CPU feature tensors for auditing. Disabled by default.

Rendering is separate from inference. Keep using `infer_video.py` when the portable analysis viewer and annotated video are required. The existing demo viewer is unchanged by deployment work.

## Reuse within an application

```python
from pathlib import Path
from deployment import CSEBInference

settings = Path('/path/to/local-settings')
settings.mkdir(parents=True, exist_ok=True)
runner = CSEBInference(checkpoint, encoder_root, yolo, settings)

# images: uint8 RGB array [8,H,W,3], following the trained10Hz sampling.
# key_t: keyframe position in that window, normally3; boundaries use repeated frames.
result = runner.predict_clip(images, key_t=3)
boxes = result['boxes']
scores = result['scores']['blend']   # [N,184], already detector-weighted
```

Retain the runner for subsequent calls. `VideoReader` in `infer_video.py` implements the original video resampling and boundary semantics. The window uses future context; this remains offline/noncausal inference, not a causal live-camera model.

## Numerical contract

The reference checkpoint and source model are checked by hash, with strict state loading and phrase-bank agreement. Visual feature rounding to float16, BF16 expert execution, primitive carry-forward, blend weight and single application of detector confidence are preserved. The deployment encoder returns the original final-block token features directly and omits the unused downstream CLIP pooling computation.

`benchmark_deployment.py` compares identical fresh frames/proposals across original execution and candidate implementations. Its screen requires all expert/blend scores to pass `atol=1e-4, rtol=1e-3` and identical top triplets. Passing a short screen does not establish unchanged detector AP on a dataset.

## Compilation and attention

The installed runtime is PyTorch2.4.1/CUDA12.4 on RTX A6000. Profiling found native PyTorch FlashAttention in all40 visual blocks. The optional external `flash_attn` package is not used; its disabled import guards do not disable native SDPA kernels.

`--compile-blocks` is an experimental compiler path. The installed-runtime test failed the numerical screen: maximum score error0.02167 and some changed top triplets. Keep the eager default for validated inference. The compiled path is retained for controlled investigation only. Initial compilation took90.16s; a new remainder batch added a33.44s first-use frame latency. Timings of previously compiled shapes were about2s/frame, but those outputs failed validation. TensorRT and ONNX runtimes were absent; the research environment was not upgraded.

See [the execution plan](DEPLOYMENT-PLAN.md) and [the benchmark results](deployment-results.json) for measured outcomes. The 1B encoder still processes a temporal crop for each proposal; removing cache files does not remove that computation.

## Executed validation

The default eager runner completed 120 frames / 3,634 proposals in 345.89s, including 13.89s startup. Median frame time was 2.738s; throughput excluding startup was 0.362 frames/s. Visual encoding took 324.53s and both heads took 0.91s. All 2,005,968 expert/blend score values matched the saved reference exactly, including proposal order and temporal sampling. No detector AP was recomputed.

The paired 3-frame comparison found no meaningful acceleration from residency alone and about 1% from batch 16. Batch 8 remains the validated default. Compiler output failed parity and is not accepted. A larger speedup requires another optimization study or a smaller distilled visual encoder; this runner remains offline.

Recheck a completed run with:

```bash
python verify_deployment.py --run /path/to/inference-output \
  --sample /path/to/reference/sample \
  --reference-predictions /path/to/reference/predictions \
  --output /path/to/verification.json
```

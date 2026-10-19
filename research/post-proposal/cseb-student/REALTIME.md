# Real-time CSEB student

The distilled student is the current real-time option. It shares one frame encoder across all detections and carries a causal scene-memory state. The recorded CSEB analysis viewer remains available separately.

## Workstation quick start

The verified runtime bundle is `/home/brandon/.local/share/road-reason/cseb-student-realtime-v1/`. From the repository root:

```bash
PYTHONPATH=/home/brandon/.local/share/road-reason/cseb-demo-runtime/packages \
CUDA_VISIBLE_DEVICES=0 \
/home/brandon/miniconda3/envs/road_reason/bin/python research/post-proposal/cseb-student/live.py \
  --bundle /home/brandon/.local/share/road-reason/cseb-student-realtime-v1 \
  --source /path/to/video.mp4 --display --output /tmp/student-live
```

Use `--source 0` for a camera. Camera capture keeps only the latest frame, so slow inference does not accumulate a queue. Press Q or Escape in the display to stop, or Ctrl+C in the terminal. `--frames 120` bounds a run. Omit `--display` for headless operation. Output directories must be new.

Inference runs at the trained **10 Hz** cadence. Videos can run faster than playback with `--no-pace`, while retaining their10Hz sample timestamps. The model resets temporal state on each invocation; application callers should use `StudentInference.reset()` at video boundaries.

Outputs are `predictions.jsonl` with all184 scores and normalized boxes, and `report.json` with measured loading, inference/processing times, source and label order. The display shows agent-compatible triplet labels; agent classes without valid triplets show the agent label.

## Move to another workstation

Use the repository code plus a CUDA/PyTorch environment matching [requirements.txt](requirements.txt). Choose CUDA wheels appropriate for that workstation. Student inference needs its head, the pretrained ResNet backbone and ROAD YOLO weights; the 1B teacher encoder and research feature caches are not required.

Create a portable bundle from the trusted trained checkpoint:

```bash
python research/post-proposal/cseb-student/export_bundle.py \
  --run /path/to/cseb-student-training-run \
  --yolo /path/to/road-yolov8x/best.pt \
  --output /path/to/new-student-bundle
```

Copy the resulting bundle directory to the other machine and supply it with `--bundle`. It contains `student.pt`, `backbone.pth`, `yolo.pt` and a label/hash manifest. Loading verifies the model source and each asset hash. Weights stay outside Git; there is no public weight-download URL configured.

## Measured behavior

The initial paired RTX A6000 test measured38.4ms/frame for the student pipeline versus1.685s for the teacher, including YOLO and CPU score export but excluding disk decoding/loading. The student has9.65M semantic parameters plus the shared YOLO detector. The live entry point was additionally checked on a120-frame video without display; see [runtime-package-validation.json](runtime-package-validation.json) for complete processing time. Camera/display support has not been exercised against physical camera hardware in this session.

The pilot student was trained on8videos. Its held-out triplet proposal AP was24.1% versus11.6% for supervised-only training, with17/86 classes represented. Full detector AP and long-stream accuracy are unmeasured. [Pilot protocol and results](README.md).

An isolated [expanded-data candidate](EXPANSION.md) is being trained; it does not replace the v1 bundle automatically.

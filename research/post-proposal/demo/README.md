# CSEB prototype demo

Runs the **actual scaled CSEB two-expert checkpoint** end to end from raw video: fresh YOLO proposals → fresh 1B crop/RoI/entire-frame features → both experts → annotated video and interactive viewer. A separate cached-input mode supports reproducibility and quick inspection. This is a qualitative prototype, not a new accuracy experiment or live camera application.

[Architecture and diagram](../../../docs/architectures/cseb-scaled/README.md) · [All architectures](../../../docs/architectures/README.md)

## Included example

[Interactive viewer](../../../docs/assets/cseb-demo.html) · [12-second video](../../../docs/assets/cseb-prototype.mp4) · [Preview](../../../docs/assets/cseb-prototype.png) · [Media provenance](../../../docs/assets/cseb-prototype-provenance.json)

Download `cseb-demo.html` and open it locally in a browser. The self-contained file includes all 120 frames and recorded predictions; it needs no model installation, server, or internet connection for playback. Use the timeline, click detections, switch experts, and adjust the display confidence threshold. Controls stay within the viewport; the scene scales to the remaining space without changing its aspect ratio. On narrow screens, expand “Detection / expert scores” to inspect predictions. Information panels scroll internally when needed.

The viewer and exported video use a dark, minimal blueprint design with monospace typography and plain score rows. Edit `viewer.html` for the interactive layout and `render.py` for video styling, then rebuild from the existing sample and predictions. Regenerating presentation files does not require rerunning inference.

## What runs

The unchanged [trained model](../contextual-roi-1b/model.py) consumes 1408-D crop and RoI features, 16 entire-frame scene tokens, normalized boxes and its embedded phrase bank. Both experts use scene attention and language cross-attention. Both were trained with DCB; one also received corrected all-184 contrastive supervision.

For composition scores: `blend = (1 − weight) × classification + weight × contrastive`. The first 49 scores are carried from the classification expert. The checkpoint supplies the development-selected weight; it is not tuned on the displayed sequence. Action/location/composition scores are multiplied by YOLO confidence exactly as in the official evaluator. Agent labels come from YOLO, not the head's agent logits.

No labels or targets enter inference. Ground truth is not used to choose or annotate the sample. The exporter defaults to the first validation video's first 120 manifest frames. File hashes verify the exact cached inputs and checkpoint's protocol/model compatibility. Use only trusted research checkpoints; PyTorch checkpoint loading is not an upload endpoint.

## Real-time option

Use the [causal student runtime](../cseb-student/REALTIME.md) for video or camera inference at10Hz with optional live overlays. The CSEB viewer remains the detailed recorded analysis interface.

## Inference without analysis exports

Use [the deployment runner](DEPLOYMENT.md) to keep models loaded and pass features directly between GPU stages. It writes per-frame scores without mandatory feature caches or rendering. [Execution plan](DEPLOYMENT-PLAN.md) · [Measured results](deployment-results.json).

## End-to-end video inference

The visual assets and pinned packages are staged locally in `/home/brandon/.local/share/road-reason/cseb-demo-runtime/`. Large weights and run artifacts stay outside Git. No encoder retraining is needed. The fixed 184 phrase bank is already in each expert checkpoint; new videos require no new text encoding.

Run from this directory on the workstation:

```bash
PYTHONPATH=/home/brandon/.local/share/road-reason/cseb-demo-runtime/packages \
CUDA_VISIBLE_DEVICES=0 \
/home/brandon/miniconda3/envs/road_reason/bin/python infer_video.py \
  --video /path/to/input.mp4 \
  --checkpoint /home/brandon/.local/share/road-reason/contextual-roi-1b-20260930/attention-global-dcb-seed0/best.pt \
  --encoder-root /home/brandon/.local/share/road-reason/cseb-demo-runtime \
  --yolo /data/repos/ROAD_Reason/experiments/exp11_yolo/runs/v8x_agent_1280_eccv/weights/best.pt \
  --output /path/to/new-demo-run --frames 120
```

The run directory receives `sample/` (freshly generated features and decoded frames), `predictions/`, `viewer/index.html`, `viewer/cseb-prototype.mp4`, and atomic `progress.json`. Input videos are resampled to the trained 10 Hz cadence. Eight-frame windows span keyframe−3 through keyframe+4; this is offline/noncausal inference with approximately 0.4 seconds of future context. Output FPS describes playback, not processing speed. Arbitrary videos run without ROAD annotation files; that does not establish accuracy on a new domain.

Default YOLO confidence is **0.25** for a practical qualitative demo. Set `--detector-conf 0.001` for the research evaluator's proposal threshold; `imgsz 1280`, NMS IoU 0.7, `max_det 300`, and half precision match the original detector code. Higher demo thresholds change which objects get processed, so do not quote the published detector AP for a thresholded demo. Detection settings and implementation/input hashes are recorded. Fresh video decoding can differ slightly from the JPEG frame extraction used for evaluation.

Three-frame A6000 benchmark (October19): startup 13.9 s; warm frames 2.64–2.79 s each with 30–32 detections. Encoder computation dominates. This is **not real-time processing**. The complete 120-frame run has its own measured timings in its manifest/progress report.

On another workstation, install `requirements-inference.txt` into an isolated CUDA/PyTorch environment with the dependencies below. Obtain the officially gated base and CLIP visual weights with authorized access, then use `configure_assets.py --base BASE.pt --clip CLIP.pth --root RUNTIME_ROOT` to verify exact SHA-256 values. The official `OpenGVLab/InternVideo2_CLIP_S` Python implementation at revision `1f9fca1389fd883defc652634d95a21121c85a8c` must be present in the Hugging Face cache. The unchanged 1B benchmark loader supplies the 1B topology and weights. Optional external `flash_attn` imports are guarded because that package is unused. Native PyTorch SDPA remains enabled; profiling verified FlashAttention kernels in all40 visual blocks.

## Cached-input demo

Dependencies: the existing research Python environment with PyTorch, NumPy, Pillow; FFmpeg with libx264 for MP4. The original tested local environment is `/home/brandon/miniconda3/envs/road_reason/bin/python` (PyTorch 2.4.1). Viewer playback needs only a modern browser, with no internet libraries or service dependencies.

Run the stdlib-only exporter where the completed scaled study is stored (NCShare). This reads data and creates a new output directory; it never modifies the original cache:

```bash
python3 export_sample.py \
  --root /work/bbyrd1/contextual-roi-1b-20260930 \
  --output /work/bbyrd1/cseb-demo-sample \
  --frames 120
```

Copy that sample to the workstation, then run from this directory. Output directories must not exist, to avoid overwriting earlier demo evidence.

```bash
python predict.py \
  --sample /path/to/cseb-demo-sample \
  --checkpoint /path/to/attention-global-dcb-seed0/best.pt \
  --output /path/to/cseb-predictions \
  --device cpu

python render.py \
  --sample /path/to/cseb-demo-sample \
  --predictions /path/to/cseb-predictions/predictions.json \
  --output /path/to/cseb-viewer --mp4

python -m http.server 8765 --bind 127.0.0.1 --directory /path/to/cseb-viewer
```

Open `http://127.0.0.1:8765/`, or open the self-contained `index.html` directly. HTML includes its own images and predictions; copying that one file to another computer is sufficient. `cseb-prototype.mp4` is a 10 fps video, matching the source ROAD-Waymo video's frame rate. Playback speed in the HTML is adjustable.

The viewer supports scrubbing, per-frame box selection, confidence filtering, and switching between expert/blended predictions. Selected composition scores show the exact two expert contributions. Candidate numbers are not temporal track IDs. Low-confidence candidate filtering is display-only; `scores.npz` retains all candidates and all 184 scores. The sidebar crop is the selected keyframe box, not the expanded eight-frame encoder input crop.

## Outputs and interpretation

- `predictions.json`: display predictions, full source/signature metadata, measured inference duration and precision.
- `scores.npz`: all 184 detector-weighted scores from each expert and the blend, in manifest candidate order. Agent-class evaluation additionally uses the separate YOLO class ID.
- `index.html`: portable interactive viewer.
- `cseb-prototype.mp4`, `preview.png`: rendered video and first-frame preview when `--mp4` is used.
- `receipt.json`: output hashes and source identifiers.

CPU head inference uses float32; `--device cuda` uses the evaluator's bfloat16 autocast. Small numerical differences are possible. Head timing excludes encoder feature extraction and YOLO; raw-video timings are separately recorded. Do not present recorded playback as live real-time perception, a new accuracy measurement, or the three-seed aggregate model. The raw-video command executes all visual inference; the cached-input command intentionally reuses existing features.

## Checks

```bash
python -m unittest discover -p 'test_*.py'
```

Checks cover primitive carry-forward, composition blending, single application of detector confidence, invalid blend contracts, portable-array slicing, video timing/boundaries, and exact GPU preprocessing parity with the original trainer. Production inference also checks model/protocol/cache hashes, phrase-bank agreement, feature dimensions, finite values and strict state-dict loading. Render checks frame hashes and prediction-to-sample identity.

Box overlays show the top predicted triplet for the displayed expert/blend. YOLO agent categories with no ROAD-Waymo triplet class (such as traffic lights) are marked `triplet N/A`; raw scores are retained. Longer overlay labels are separated to reduce overlap.

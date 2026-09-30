# InternVideo2-1B contextual DCB scale-up

Authorized September30,2026. Target is model104 with both attention contextual DCB experts and their AP-selected global blend. See protocol.json. Preserve YOLOv8x boxes,184 labels, crop/RoI/entire-frame inputs, geometry, all184 pre-fusion visual-RoI versus adapted-text contrastive loss, and all three seeds.

## Prelaunch dependencies

The released `1B_clip.pth` contains only47 learned tensors (~14MB), not the full encoder. It requires the separately gated Stage2-1B base checkpoint and public InternVL text weights (~25.4GB). Both model access grants are required. Never silently run with randomly initialized missing weights. New phrase embeddings require the official matching tokenizer, whose provenance must also be checked.

The existing checkpoint called InternVideo2_CLIP_S is already24x1024 (L14-sized). This target is40x1408. Pin every asset revision. Full model104 training is still gated on visual/text compatibility, validated new cache manifests, monitoring registration and measured ETA. Prepared files do not imply submitted jobs.

## Benchmark

`benchmark_vision.py` loads the official base visual weights plus CLIP projector into the existing verified InternVideo2 implementation, changes only model dimensions to match the official1B configuration, interpolates temporal positions4→8 with the existing implementation, requires full visual state coverage, checks SDPA versus naive attention numerically, and times real ROAD frame/crop inputs. It preserves all scientific preprocessing and emits memory/throughput results. A benchmark is not a model-quality pilot.

`benchmark.sbatch` was submitted as job744980, dependent on verified asset-download job744979. It requires verified staged model assets, existing cached implementation, and a registered study monitor. Benchmark allocation cap2hours is a safety ceiling, not a measured runtime estimate. Full-study ETA is unknown until the benchmark completes; first useful output will be the compatibility/throughput report, not AP.

Text compatibility uses the exact tokenizer/config linked by the authors (pinned source and hashes in tokenizer-provenance.json), all64 frozen q/v LoRA branches, strict base-weight coverage and the existing184 phrase order. Dedicated text dependencies are isolated from existing runs. No full extraction or head-training job is submitted yet.

Submitted text check:744996, after successful asset job744979, oneH200 with1hour allocation ceiling. Setup uses private text dependencies transformers4.37.2/tokenizers0.15.2/huggingface-hub0.36.0/sentencepiece0.2.1; existing environments are unchanged. User requested one completion ping: research-1b-setup-ping.timer checks all three success receipts and makes at most one desktop notification attempt. Routine notification streams remain muted; no automatic new chat turn is promised.

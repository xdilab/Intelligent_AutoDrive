# CSEB inference deployment plan

## Scope

Keep the current trained model, fixed blend weight, 184-score order, YOLO settings, preprocessing and eight-frame temporal window. Build an inference runner that loads once, keeps visual features on GPU, runs both experts immediately, and exports scores without mandatory feature caches or rendering. No new training or detector AP is claimed.

## Execution

1. Inspect the installed runtime and free GPU capacity. GPU0 is an idle RTX A6000; PyTorch2.4.1/CUDA12.4 and Triton are installed. TensorRT/ONNX runtimes are absent.
2. Implement `deployment.py`, retaining the existing research/demo reference path. Remove only the unused final CLIP pooling work: CSEB consumes the preceding block tokens.
3. Profile the actual attention operator. Paired benchmarks use the same fresh video frames/proposals for reference, resident batch8, resident batch16, and an experimental block-compiled variant.
4. Accept numerical variants only if every expert/blend score passes `atol=1e-4, rtol=1e-3` and top triplets match on the checked proposals. Report failures explicitly. This screen does not substitute for detector AP evaluation.
5. Verify no-candidate handling, score assembly, and original preprocessing. Run the selected configuration end to end on the recorded segment with fresh detection/encoding; compare stored reference predictions.
6. Document measured throughput, startup/compile cost, memory, accuracy checks, remaining bottlenecks and the deployment command. Keep weights/checkpoints outside Git. Do not install TensorRT into the research environment.

## Time and monitoring

Bounded local synchronous benchmarks on GPU0, with atomic progress and stdout per phase/frame; no queued training or unattended experiment. Historical throughput~2.7s/frame gives first comparison1–3min after loading, benchmark pass provisionally5–10min including unknown compilation cost. Each benchmark process is time-limited; inspect logs during execution. Full120-frame inference provisionally6–8min from historical speed, refreshed after benchmarks. Intermediate encoding metrics and qualitative score parity are distinct from detector evaluation; no detector-AP ETA applies because none is launched.

## Follow-on decision

If compilation/batching do not meet latency needs, measure isolated TensorRT support before changing the runtime. A smaller distilled visual encoder is a separate training study with a new accuracy/latency comparison. Current future context remains~0.4s even if computation is accelerated.

## Outcome

Executed the plan through the full 120-frame parity check. The eager GPU-resident runner is the default. All scores matched exactly; native FlashAttention was already active. Batch 16 produced only a small short-sample gain. Block compilation ran after fixing dynamic module registration, but changed scores/top triplets and was rejected. TensorRT was not installed and distillation was not launched. See DEPLOYMENT.md and deployment-results.json for measured times and remaining limits.

# Contextual composition readout study

Authorized September 19, 2026. Six fresh attention-fusion runs: seeds 0–2 with classification alone or corrected all184 adapted-text contrastive loss (0.001). Immutable parent encoder caches are reused.

`flat`: 1208→512 GELU→49; `comp`: [49 primitive sigmoids, 1208 features]→512 ReLU→135. Composition gradients reach the primitive head. Both objectives retain language attention. The primary analysis is the paired objective-by-readout interaction in detector triplet AP, not final arm ranking. Increased capacity is a confound for attributing gains to composition structure alone.

Run `python -m unittest discover -s . -p 'test_*.py'`, then `python preflight.py` before `python pipeline.py`. Preflight verifies production-dimension backbone equality with historical flat implementations whose source hashes match recorded runs. Its GPU benchmark is disposable. Pipeline trains and evaluates one seed per GPU lane, preserves resumable optimizer/RNG checkpoints, and writes comparison.json after all six results. Systemd watchdog and shared repair dispatcher monitor progress. See protocol.json and the wiki direction for full provenance and inference limits.

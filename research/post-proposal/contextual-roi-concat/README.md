# Full-context concatenative skip comparison

Authorized September24,2026. Append the full projected context x=[crop,RoI,scene summary,position] to e=MLP(x), then learn a2560→512 projection before the existing LayerNorm/language head. Scene summary remains8-head attention; the experiment does not flatten all16 raw scene tokens. W starts[I,0,0,0,I] so all common parameters/RNG and initial predictions match the original crop-residual baseline. New paths learn from the first update.

Six new DCB runs: seeds0,1,2 × classification or corrected all184 contrastive λ.001, followed by the same development-AP-selected global probability blend. Three epochs, same420/90/90 split, caches, labels, learning rate, batch order and official36717-frame validation. Compare every seed to immutable DCB102/103/104 results. Checkpoint selection uses development only.

Added1,310,720 weights are an explicit capacity confound. Improvement would support this implementation, not prove concatenation is intrinsically superior. No existing architecture figure or result is relabeled. Results/checkpoints are on home storage through the wiki runs symlink; no cache duplication.

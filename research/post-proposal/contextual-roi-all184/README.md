All184 visual contrastive objective, authorized September18.
Fresh six-run comparison; previous triplet-only study is immutable.
Global184-way multi-positive softmax loss attemperature0.07, lambda0.001; samefocalclassification, seeds0–2, caches, initialization anddevelopmenttripletselection.
Run pipeline.py using contextual-roi-all184-study.service. It waits for original12evaluations to finish before using the two localGPU lanes.

September18 design refinement: contrastive similarity now uses adapted phrase matrix T, with gradients into both visual and text MLPs before language fusion. Frozen P and encoder caches remain fixed. All184 coverage, temperature0.07, weight0.001, splits, seeds and development selection unchanged. Superseded fixed-P queue configuration archived before any training; historical triplet-only runs untouched.

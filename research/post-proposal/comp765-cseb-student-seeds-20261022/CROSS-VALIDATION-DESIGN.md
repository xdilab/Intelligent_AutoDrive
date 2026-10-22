# Optional grouped cross-validation — design only

No cross-validation was launched. The user first selected “add seeds now; prepare a 5-fold design without launching it”, then clarified on October 22 that five folds are not required. This clarification supersedes the planning-only five-fold wording in the frozen extension protocol. It changes no seed-extension condition. Fold count remains undecided.

## What the two studies answer

Additional paired seeds estimate training-randomness sensitivity for the existing fixed split. Grouped cross-validation estimates sensitivity to which videos the student sees. It does not remove teacher/detector exposure or create new labeled data. Finish the fixed ten-seed extension before deciding whether split robustness is worth a separate experiment.

## Conditional student-only design

A possible pool is the existing 76 videos (64 training + 12 development); the 198 official validation videos must not be silently folded into training. Split whole videos into outer folds, never frames or proposals. If several clips share a drive/source recording, group at that larger level where lineage is available. Video IDs are the minimum verified grouping; independence across drives has not been established.

Within each outer training fold, reserve separate grouped development videos for minimum-supervised-BCE checkpoint selection. Never select the checkpoint on the outer test fold. Match supervised/distilled initialization, training order, optimizer, epoch budget, candidate policy and losses within every fold/seed pair. Reset scene state at video boundaries. Generate teacher targets only for that fold's training inputs; do not use teacher predictions as evaluation labels.

A deterministic candidate split rule is to hash a versioned salt plus group ID, sort, and distribute round-robin into K folds; a second fixed salt chooses the inner selection subset. Before freezing, inspect annotation-derived class coverage and group sizes (not model AP), record any absent classes, and decide whether a prespecified grouped multilabel balance rule is necessary. No fold membership is selected for favorable outcomes.

Five folds is only one option. With 76 equal video groups and 12 inner selection videos, K=3 gives roughly 38–39 training,12 selection,25–26 outer-test videos per fold; K=5 gives48–49 training,12 selection,15–16 outer-test. More folds leave more training data per fit but smaller test sets and greater cost. Three paired seeds imply18 models for3 folds or30 for5 folds. These are model counts, not measured runtime estimates; benchmark before any launch.

## Evaluation and uncertainty

Report fold/seed results and all classes, then concatenate out-of-fold predictions within each seed and compute pooled official detector AP using the fixed 86-class ordering. AP is not additive: averaging fold AP is a secondary descriptive summary, not pooled AP. Keep training-derived Tail47/Deep28/Common39 membership fixed. Missing-class support must remain visible. Do not treat folds as independent seeds or apply the paired-seed t interval to all fold×seed cells: training sets overlap. Describe split sensitivity; any inferential interval requires a separately specified dependence-aware method.

The current teacher has seen this pool in its training/development history, and YOLO trained on the600-video training pool. Therefore this is explicitly conditional student validation using an already-exposed teacher/detector, not independent generalization of the complete perception system. Distillation can carry information about outer test videos through teacher weights; disclose that limitation rather than claiming clean teacher–student cross-validation.

## Assets and work needed before approval to run

Existing32-frame caches cover76 videos, but the12 original selection caches intentionally have no teacher targets. They would need training-only target generation whenever their videos enter a CV training fold. Existing proposal caches at confidence0.25 do not replace official full-candidate detector evaluation. Audit available candidate/GT manifests for all prospective outer-test videos and generate verified candidates if missing; validate mapping and score assembly first. Full-video prefix evaluation would require current-image student inference over each outer-test video, not merely crop/proposal AP on the32-frame cache.

A genuinely independent system comparison requires labeled videos unused by teacher/student/detector training and selection, or teacher and detector fitting/selection restricted inside each outer training fold. Neither is authorized or implied by the present fixed-teacher seed extension. No teacher/backbone retraining or new paid compute is started.

Sources: [scikit-learn grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html), [Bengio and Grandvalet on cross-validation variance](https://jmlr.org/papers/v5/grandvalet04a.html). Local exposure evidence: `../comp765-cseb-student-20261021/results/records/detector-history-audit.json` and the original split manifest.

# Targets, state, and score assembly

These details describe the unchanged existing student conventions, now used by both matched conditions.

| Output columns | Meaning | Training |
|---|---|---|
| 0 | Detector agentness | Excluded from both losses; inference uses detector confidence |
| 1–10 | 10 agent labels | Label BCE, plus teacher BCE in the distilled arm |
| 11–32 | 22 observed actions | Same |
| 33–48 | 16 ego-relative locations | Same |
| 49–97 | 49 duplex labels | Same |
| 98–183 | 86 triplet labels | Same |

Ground-truth IDs are resolved through `all_*_labels` strings into the output label order. Each training candidate receives the multi-hot labels of the highest-IoU ground-truth box with the same agent class, provided IoU is at least 0.5. Otherwise it receives all-zero semantic labels. This proposal target assignment permits more than one candidate to match a GT box. Official evaluation separately uses the established one-to-one detection matching and penalizes duplicate detections. No mis-indexed child arrays are used.

The fixed teacher's primitive probabilities come from the classification expert. Duplex and triplet probabilities average the classification and contrastive experts with fixed weight 0.5. Teacher inference applies detector confidence; target construction divides that factor out once, checks the probability range, and clamps only floating-point overshoot. Training uses independent Bernoulli targets, not a softmax distribution. The teacher's eight-frame window spans current minus three through current plus four at 10 Hz. This future information is privileged supervision, never a student input.

Both students use the same cached **ResNet50** RoI and scene features, never the teacher's 1B features. The cache is a training optimization. The portable inference runner recomputes the ResNet50 features from the current raw-video image.

Each training sequence starts with zero scene state and backpropagates through all 32 consecutive steps. Unannotated steps provide current-image scene features and zero actor rows: state updates, but neither loss receives fabricated negative labels. Selection follows the same reset policy. Final evaluation instead processes each entire video prefix from frame 1, resetting only at video boundaries and carrying state through annotation gaps. No tracking-specific or future-frame memory is introduced.

At inference, every semantic probability is multiplied by detector confidence exactly once. Column 0 is replaced with confidence. The established official evaluator uses the detector's own class and confidence for agent AP, so agent/agentness results are shared across student conditions. Action, location, duplex, and triplet results use the corresponding student outputs.

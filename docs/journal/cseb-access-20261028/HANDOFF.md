# CSEB journal revision: laptop Codex handoff

Prepared for Brandon L. Byrd on October 28, 2026.

## Task

Help Brandon revise his **existing CSEB journal manuscript** for IEEE Access and prepare a concrete architecture/figure selection for his next meeting with Dr. Hamidreza Moradi. CSEB (Contextual Scene-Attention Expert Blend) is the selected final architecture.

The advisor's concerns are: the draft currently has only one figure; the architecture and its changes are unclear; the paper needs selected baseline/intermediate comparisons explaining why each design was tried, what worked, and why the final model was retained. He requested IEEE Access format and a draft with figures already placed for discussion. He also requested Brandon's commitment to help address reviewer comments and resubmit after graduation. A proposed email reply is included but has **not been sent**.

Brandon wants a rich visual explanation of the evolution and justification. Use enough diagrams to answer the design questions, with aligned multi-panel figures and supplementary detail. Do not equate more diagrams with more architecture contributions.

This handoff is a plan and evidence packet. It does not certify that the journal manuscript is ready for submission. **No manuscript, thesis or defense slides were edited on the server for this task.**

## Start here on the laptop

1. Read applicable local `AGENTS.md` and repository instructions.
2. Locate the current editable journal manuscript from Brandon's laptop workspace. The server wiki records that a journal draft already exists, but it was not found in the server checkout. Do not start a replacement paper from the old thesis proposal. If several versions exist, identify dates/content and ask Brandon only if the authoritative version remains ambiguous.
3. Make an isolated working copy or branch; preserve user edits and the original thesis/slides. Inspect the manuscript's existing title, authors, references, equations and figure before revising.
4. Open `docs/journal/cseb-access-20261028/index.html` from the extracted bundle. It is an illustrated review packet with eight figure proposals, candidate source diagrams, proposed captions, rationale and evidence. Sources are previews, not finished journal-sized figures.
5. Verify `bundle-manifest.json` hashes. The bundle supplies figures/results locally; `/data/repos/...` and `/home/brandon/...` strings in historical provenance are server identities, **not usable laptop paths**.
6. The repository is `https://github.com/xdilab/Intelligent_AutoDrive`. Find an existing laptop checkout or obtain a separate checkout if needed. Preserve unrelated changes. The bundle contains relevant source snapshots for inspection, not a full installable research environment. Do not overwrite repository files with older bundle files without comparing revisions.

## Scope

- Revise the thesis-derived CSEB journal story, figures, tables, captions, methods and interpretation.
- Keep the separate COMP765 ResNet50/GRU student-distillation and association-rule experiments outside this paper unless Brandon explicitly changes scope. Their ten-seed results are **not** CSEB's three-seed results.
- Do not modify the thesis or defense presentation. Do not make class slides.
- Use existing experimental evidence. Record genuinely missing controls as gaps; do not launch training or invent results to fill them.
- Do not send the email, submit to a journal, upload to SharePoint/Overleaf, or push a remote branch without applicable user authorization. Prepare a reviewable local result first.

## Proposed paper narrative

**Question:** How can actor-specific appearance, scene context and label-conditioned representations support compositional road-event detection, and does combining differently trained contextual experts improve the final predictions?

**Contribution framing:** The pretrained detector, video/text encoders and RoIAlign are inherited. DCB is adopted/adapted from prior work, not a newly invented loss. The work integrates crop, RoI, scene and geometric evidence with adapted phrases; evaluates scene aggregation, readouts and training objectives; and studies a selection-set-chosen blend under an explicitly documented detector protocol.

Keep the existing manuscript title if appropriate. A possible working title, only if needed, is “CSEB: Contextual Scene-Attention Expert Blending for Compositional Road-Event Detection.” Do not claim open-vocabulary recognition, causal prediction, linguistic-semantic attribution, state of the art across unmatched protocols, or real-time CSEB inference.

Suggested section order:

1. Introduction: task, difficulty, research question and supported contributions.
2. Related work: event detection, composition, contextual fusion, contrastive learning and adopted loss.
3. Problem formulation and experimental protocol: ontology, candidate boxes, temporal inputs, splits, metrics and selection.
4. CSEB: overview, context fusion, expert objectives, blending and score assembly.
5. Experiments: baseline families, direct expert comparisons, matched ablations and separately labeled scaling.
6. Analysis: tail/per-class outcomes, negative alternatives, qualitative successes/failures, cost and limitations.
7. Conclusion: what the evidence establishes and what remains unresolved.

## Architecture selection and comparison logic

| Family or comparison | Why include it | Main paper versus supplement |
|---|---|---|
| Published 3D-RetinaNet/FRCB and local detector baseline | Establish recognizable references and the starting point | Main baseline table; clearly separate published protocols from local reevaluation |
| Semantic crop classifier (Stage4), visual composition (Stage5), phrase fusion (Stage6) | Explain actor-local recognition and alternative composition readouts | One aligned three-panel historical figure; full details in supplement |
| Contextual mean pool versus scene attention | Test actor-conditioned scene aggregation under a matched study | Main method comparison and ablation table |
| Classification expert versus contrastive expert versus CSEB | Direct parents of the final model, using the same scaled system | Main architecture and headline comparison |
| Focal versus DCB; original versus scaled encoder system | Explain training-objective and representation choices | Separate result blocks; these are variants, not wholly new architectures |
| Global blend versus classwise/generic/factorized gates | Explain retaining the simpler fixed blend | Compact negative/alternative comparison; detailed router diagrams supplementary |
| Contextual composition MLP and language controls | Bound claims and document alternatives that did not clearly help | Main limitations/ablation summary, full results supplementary |

Do not turn the entire chronological experiment history into the main table. Preserve the broader record in supplementary material. A historical improvement is not an isolated component effect when encoders, data or recipes also change.

## Eight proposed main figures

The count is provisional and editorial, not an IEEE requirement. Slots1–6 explain architecture/process; slots7–8 present evidence. Each figure needs a specific question, a caption, an in-text interpretation and an associated comparison where available.

| Figure | Proposed content and placement | Why it is needed |
|---|---|---|
| 1 | Task, candidate box, eight-frame window and evaluation/selection protocol; problem/setup | Define exactly what the model observes and predicts |
| 2 | Stage4/5/6 crop baselines, three aligned panels; baseline subsection | Show what local scoring and explicit composition do before contextual modeling |
| 3 | Complete scaled CSEB architecture; opening of proposed method | Establish the definitive final model and inherited/trainable components |
| 4 | Mean pooling versus actor-conditioned scene attention, two aligned close-ups; fusion subsection | Explain the query, keys/values and motivation for scene selection |
| 5 | Classification versus contrastive expert training, with correct gradient paths; objectives subsection | Explain the actual difference between experts and frozen/trainable boundaries |
| 6 | Composition-only blend and held-out selection, with primitive bypass; aggregation subsection | Explain the simple retained inference rule and its selection procedure |
| 7 | Controlled comparisons plus overall/tail/deep/common and per-class outcomes; results | Connect diagrams to evidence, including tradeoffs and negative findings |
| 8 | Qualitative improvements and failures, with documented example selection; discussion | Make behavior and limitations visible; report runtime scope in an adjacent table |

Supplement candidates: detailed Stage0–3 evolution, full readout/router diagrams, all per-class AP, exact loss/selection details and encoder adaptation branches. Merge a main figure only when it preserves readability and scientific purpose.

Detailed captions, preparation notes and asset names are in `figure-plan.json` and the illustrated HTML packet. Figures1/7/8 still require construction or example selection; existing sources for the other slots require journal-layout adaptation. Do not describe those tasks as already completed.

## Verified result anchors

ROAD-Waymo detector frame-mAP at IoU0.5 (%), mean ± sample standard deviation across seeds0/1/2, on36,717 scored validation frames:

| Scaled configuration | Triplet | Tail47 | Deep28 |
|---|---:|---:|---:|
| Classification expert, DCB | 15.026 ± 0.208 | 8.396 ± 0.582 | 6.052 ± 0.540 |
| Contrastive expert, DCB + all184 auxiliary | 15.081 ± 0.297 | 8.533 ± 0.628 | 5.992 ± 0.532 |
| CSEB, AP-selected composition blend | **15.748 ± 0.274** | **9.000 ± 0.644** | **6.354 ± 0.571** |

Raw source: `research/post-proposal/results-snapshots/20261019-headline/scaled-cseb/`. All nine raw-result hashes were verified against `full-summary.json`; the corresponding local workbook headline cells agree. Recompute any new statistics from raw per-seed results.

Useful comparisons from the local workbook (unrounded values and source references are in `selected-results.csv`):

- Mean pooling → scene attention, matched focal classification: triplet11.740→12.024; Tail47 5.473→6.189. Report paired variability before interpreting significance.
- Scaled CSEB versus classification expert: +0.722 triplet and +0.604 Tail47 points. Versus contrastive expert: +0.667 and +0.468 points.
- Small-encoder global/classwise/generic/factorized routing: triplet12.584/12.590/12.588/12.593. This does not establish a practically meaningful advantage for complex routing.
- DCB versus focal global blend: triplet12.584→13.428, while Tail47 6.488→6.234. Do not describe DCB as improving every group.
- Original-encoder DCB blend → scaled CSEB: triplet13.428→15.748; Tail47 6.234→9.000. This changes the encoder system and extraction/projection details, not only parameter count.

The included local workbook is `evidence/ROAD-Waymo Results(1).xlsx`, tab `Clean Metrics Fall 2026`. ColumnA is model ID; its values are not worksheet row numbers. `selected-results.csv` retains both. The supplied SharePoint workbook was inaccessible from the server's web tool, so cloud/local identity was **not** established. Reconcile a newer laptop/cloud copy rather than overwriting it.

## Scientific details every diagram and methods section must preserve

- CSEB uses frozen InternVideo2-CLIP-1B visual assets and matching text assets originating from InternVL; the text display name is a system name, not a1B text parameter count. Scaling changes both encoder checkpoints and required projection/extraction conditions.
- Scaled visual width1408; phrase bank184×768; trainable projection to512; contextual head width512.
- One padded eight-frame crop produces one feature vector. The CSEB temporal window is t−3…t+4 at the recorded cadence; it uses future context and is not a causal streaming model.
- Full-frame features provide actor RoI features and16 scene tokens. Geometry is an8-value box representation embedded after encoding.
- Scene attention: query = projected crop + geometry; keys/values = scene tokens. Language attention: query = pre-language visual representation; keys/values = adapted phrases. They are distinct operations.
- Both experts contain scene and language attention. Classification-only means no contrastive auxiliary, not no language.
- The corrected multi-positive contrastive auxiliary aligns pre-language visual representations with adapted text across184 outputs, with the documented weight0.001. It does not contrast final detector probabilities. Frozen encoder weights are not updated by head training.
- There are184 outputs: agentness1, agents10, actions22, locations16, duplexes49, triplets86. The first49 include agentness; there are183 ordinary semantic labels.
- CSEB experts use a flat184 readout, not the earlier Composition MLP. The standard residual around visual fusion carries crop features only.
- Blend135 composition probabilities: `(1-alpha)*p_classification + alpha*p_contrastive`. Retain classification-expert primitives before detector assembly; YOLO supplies agentness/agent labels and candidate confidence. Apply semantic confidence weighting exactly once.
- Frozen features are shared between expert heads; repeated encoder drawings must not imply independently trained backbones.
- Training, checkpoint selection and blend selection follow the exact study protocol. The recorded partition uses420 expert-training videos,90 development videos,90 reserved gate videos with a45-video selector subset. Do not collapse selection into final validation evaluation.
- Tail47 z<0 and Deep28 z<−0.5 are fixed training-derived subsets of86 triplets; Common39 is the complement of Tail47. Never use the legacy mis-indexed `duplex_childs`/`triplet_childs` arrays; use label strings/verified mappings.
- Separate detector AP, development crop AP, proposal AP and retrieval. Separate published results, local reevaluation, seed means and best runs.

Authoritative starting points: `docs/architectures/CONTRACT.md`, relevant architecture READMEs, the model/protocol/evaluator source snapshots, and `docs/architectures/assets/provenance.json`.

## Claims and missing evidence

1. Current evidence supports the measured contextual/blend system performance, not every proposed causal explanation.
2. A same-objective ensemble at comparable budget is needed before claiming that distinct training objectives uniquely explain ensemble gains. Inspect existing records before declaring that control absent or running anything.
3. Real/random/bypass language controls have limitations. Random prototypes change geometry and semantics together; inconclusive differences are not equivalence. Do not claim semantic understanding from phrase-branch presence alone. Scaled language-control evidence is included separately.
4. Historical stage comparisons do not isolate crop, RoI, scene or geometry effects. Identify genuinely matched cells and retain missing-component attribution as a limitation if necessary.
5. The official validation benchmark has prior research use and detector-selection exposure. It is not an untouched end-to-end test. Three seeds measure training randomness conditional on that benchmark.
6. Include negative results and all class outcomes. Do not select a favorable subgroup or seed after seeing results.
7. The existing raw-video CSEB demo is offline and encoding-heavy. Runtime measurements are tied to hardware, candidate count, window and export scope. The separate fast student does not establish fast CSEB inference.

## Figure production and manuscript integration

- Reuse the editable `.drawio` ancestors in the bundle; preserve the established teal video encoder, purple text/phrase branch, blue features, trained/frozen badge grammar and rounded black arrows. Read `evidence/defense-diagram-reference.md` and applicable laptop diagram skills.
- Build journal derivatives in a new directory, leaving historical figures immutable. Keep repeated components at identical stations across comparison panels. Avoid independently cropping/recentering each panel.
- Simplify for paper size without changing topology or hiding important paths. Put detailed tensor dimensions in focused figures or a table; ensure all text is readable at final one/two-column dimensions.
- Inspect source code against every scientific label, then inspect the rendered figures and their placement in the compiled manuscript. Keep an asset manifest with source hash, derivative path, figure number and caption.
- Every figure must be referenced and interpreted in the surrounding prose. Each architectural modification should state: limitation → proposed change → controlled comparison → observed outcome → retained/rejected decision.
- Use the official IEEE Access template, double-column and single-spaced, with matching editable source and PDF. IEEE Access recommends staying under20pages and may request reduction/justification above20. This is guidance, not a fixed figure quota. Check current guidance at https://ieeeaccess.ieee.org/authors/submission-guidelines/ before final packaging.
- Preserve existing authorship/reference metadata; do not invent affiliations, ORCIDs, biographies or submission identifiers. Handle required disclosures during final author review.

## Execution order and acceptance checks

1. Inventory manuscript and evidence; resolve the authoritative draft.
2. Produce a section-to-claim-to-table/figure matrix using this plan; flag missing comparisons or stale claims.
3. Convert a copy of the existing draft to the official Access format without silently dropping content/citations.
4. Prepare the CSEB overview and selected comparison figures first, then expert training/blend detail. Place actual figures and captions in the manuscript, not only placeholders in a plan.
5. Build result tables/plots from verified sources and write explanations of what worked and what did not. Reconcile the current workbook while preserving old data and source identity.
6. Add qualitative examples and failure cases using explicit provenance; finish the protocol/runtime/limitations sections.
7. Compile/export; check references, cross-references, figure text/readability, page breaks, equations and table rounding. Confirm no unsupported comparison or missing-class treatment changed.
8. Deliver an editable revised manuscript, PDF, editable figure sources and exports, figure/claim map, provenance manifest, concise unresolved-gaps list and an unsent reply draft. Summarize changes for the meeting. Do not claim the advisor approved or that submission occurred.

The advisor's graduation/publication statement is a requirement he communicated in the supplied email; this packet does not independently verify university policy. Brandon's personal commitment should be reviewed by Brandon before any email is sent.

## Packet status and portability

- Existing figure previews were visually inspected;16 PNG/draw.io source hashes were checked. They remain historical exports requiring journal adaptation.
- Selected results:29 workbook records, including direct comparisons and negative alternatives.
- CSEB headline evidence:9 raw detector results with verified hashes; original study has3 seeds, not10.
- No new experiment, manuscript edit, cloud upload, email or remote push was performed for this handoff.
- The105-slide SharePoint deck from the prior discussion was not accessible and is not included. Do not assume an older74/77-slide server copy is that deck. Use a newer laptop copy if Brandon provides it.
- Start with this handoff and the HTML review packet; use bundle-relative paths. Additional raw datasets, encoder weights, caches and training checkpoints are intentionally not included and are not needed for drafting.

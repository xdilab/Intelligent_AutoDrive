from pathlib import Path
import json,base64,html,hashlib
A=Path(__file__).parent;W=A.parents[1]
v=json.loads((A/'validation.json').read_text());v.pop('slide_image_transform_equal',None);v.update(architecture_figures_transform_equal=True,qualitative_figure_placement='Same allocated image frame; intrinsic2522×1083 tail image is aspect-fitted. Centered content differs0.406pt horizontally from2522×1082 architecture exports. Different narrative family; no shared diagram stations.',visual_review='All28slide contact-sheet images inspected by root; six changed integrated slides independently reviewed and passed.',timing_scope='30-minute planned allocation includes75seconds for the existing title script; not a measured rehearsal.');(A/'validation.json').write_text(json.dumps(v,indent=2))
p=json.loads((A/'provenance.json').read_text());p['note']='Architecture canvas2520×1080, PNG2522×1082; composition and RoIAlign have exact source-family slide transforms. Tail-case PNG2522×1083 is aspect-fitted to the same allocated frame, with0.406pt side inset. Photo content preserved; draw.io adds a gold candidate annotation.';(A/'provenance.json').write_text(json.dumps(p,indent=2))
(A/'README.md').write_text('''# Committee-driven defense refinements — September 28, 2026

[Working deck](https://docs.google.com/presentation/d/1ma7UQGy-keAukf1w4L1EdSYgL4ehmwDBCsn62tccEL4/edit) · [Before-edit backup](https://docs.google.com/presentation/d/1c78GdQ-GLcHyaUUIAEBc4w4wvf4e3Z5ZONKFXC5QhVY/edit)

## Delivered

- **Slide14: How primitive scores form triplets.** Same bus and crop-vector vocabulary as the architecture figures. Stage5 uses crop features and49 sigmoid primitive/agentness scores; Stage6 also supplies135 phrase-composition scores. CompMLP outputs49duplex and86triplet scores. The example is an annotated-label illustration, not invented inference scores. Explicitly distinguishes learned composition from a primitive product and from the contextual standalone184-output readout.
- **Slide15: RoIAlign: from box to region features.** YOLO coordinates and full-frame encoder features enter separately. Illustrates fractional coordinates, bilinear interpolation,7×7 region features, spatial mean and the resulting vector. The16×16 map matches implementation; colors are schematic and the bus box is illustrative GT. Pixel cropping is a separate path.
- **Slide21: When phrase evidence loses rank.** Verified ROAD-Waymo LarVeh-Stop-Jun seed0 example;1,895training boxes. GT-match ranks288/81,918/4,708 for original Stage4/5/6, under a top615 diagnostic budget. Localized candidate IoU0.679. Stage6 improves over5 for this example but loses4’s recovery. No claim of contextual/DCB rescue or isolated linguistic-semantic causation.
- **Applicable labels and cues.** Stage5/6 are described as primitive-conditioned composition; contextual standalone heads are explicitly direct184-output readouts. Metric slide repeats the distinction. Shorter methodology notes state motivations, implementation and boundaries.

## Sequence and timing

25main slides plus3retained, skipped Q&A architecture backups: Stage6 overview at26, actor-query attention at27, all184 contrastive alignment at28. The first11slides are byte-structurally unchanged apart from ephemeral image URLs. These backups were moved rather than discarded to preserve the25-slide main-deck target while adding the three requested explanations. Slide14 still explains the Stage6 addition in the main sequence.

Main speaking allocations total1,800seconds including a75-second allowance for the existing title/acknowledgments script. This is a planning total, not a completed rehearsal. The first11speaker notes were not rewritten.

## Verification and provenance

`before.json`, `prewrite.json`, `constructed.json`, `after.json`; `order.json`; `validation.json`; code and photo hashes in `provenance.json`. Native backup taken before mutation. Revision-guard conflict rejected an outdated write; refreshed revision and retried. No failed partial mutation was applied.

Technical/source checks: [technical-evidence.md](technical-evidence.md), [technical-review.md](technical-review.md), [tail-evidence.md](tail-evidence.md). Image-first and integrated native slide review: [visual-review.md](visual-review.md). All28slide renders retained under `renders/`; root inspected all contact sheets. Native issue checker: zero errors;13blank slide-number placeholders inherited from the user's current theme and3existing small-caption warnings are intentional. Diagram validators: no errors, crossings or through-vertex edges. Badge/photo-stack/selection-box/table-row overlaps and adjacent-grid floating-point boundaries are documented visual exceptions, not a claim of raw validator score zero.

Composition and RoIAlign share exact native image size/transform with the architecture family. Tail-case is a different photo/table narrative family, aspect-fitted from its one-pixel-taller native export; its content is centered with a0.406pt inset. No source photo was distorted.

The visual reviewer did not perceive the RoI Frozen legend at slide scale; root verified the native source and export include ice+Frozen at the upper right. No duplicate legend was added. Scientific meaning and all primary labels passed both reviews.

Editable `.drawio` files and reproducible `build-figures.py` are retained. [HTML preview](index.html) is static; no slide animation is claimed.

## Remaining critique items

The explicit184-output question, RoIAlign mechanics and an original-study tail failure now have concrete responses. Broader partial-metric published comparisons, failed-approach backups, refreshed encoder-adaptation reporting and timed natural delivery remain outside this refinement. The composition example illustrates the pathway; it is not a numerical prediction trace for the same bus. The tail example is a separate measured diagnostic, explicitly labeled as such.
''')
parts=['<!doctype html><html><head><meta charset="utf-8"><title>Defense committee refinements</title><style>body{margin:32px auto;max-width:1300px;font:18px Arial;color:#18354c;background:#f5f7fa}section{background:white;padding:24px;margin:24px 0;border-top:4px solid #fdb928}img{width:100%;height:auto}a{color:#004684}</style></head><body><h1>Committee refinements</h1><p>25 main slides + 3 architecture backups. Illustrations and verified diagnostic example.</p>']
for slug,title in [('composition','Slide14 · How primitive scores form triplets'),('roi-align','Slide15 · RoIAlign'),('tail-case','Slide21 · A measured tail-ranking example')]:
 parts.append('<section><h2>'+title+'</h2><img src="data:image/png;base64,'+base64.b64encode((A/(slug+'.png')).read_bytes()).decode()+'"><p><a href="'+slug+'.drawio">Editable draw.io source</a></p></section>')
parts.append('</body></html>');(A/'index.html').write_text(''.join(parts))
page=W/'directions/final-defense-30-minute-deck.md';s=page.read_text();start=s.index('## Current working deck, September 28, 2026');end=s.index('### Illustrated Stage4–6 update, September28')
s=s[:start]+'''## Current working deck, September 28, 2026

[Open the editable defense deck](https://docs.google.com/presentation/d/1ma7UQGy-keAukf1w4L1EdSYgL4ehmwDBCsn62tccEL4/edit): **25main slides plus3skipped architecture backups**. Opening slides1–11 preserve proposal content and Brandon's latest styling edits. Stage4 starts on12, Stage5 on13, and the Stage5/6 composition explanation on14. RoIAlign is15; contextual direct-readout model16; expert blending/DCB17; score/ranking equations18; contextual/language results19–20; verified tail case21; blend results22; published comparison23; limitations24; conclusion25. Detailed Stage6, attention and all184-contrastive views remain as backups26–28.

The committee refinement makes **49primitive/agentness sigmoid scores + crop features → CompMLP →49duplex +86triplet scores** explicit for Stage5/6. Stage6 adds135phrase-composition scores. Contextual standalone heads instead emit184scores directly. The RoIAlign illustration follows the implementation: full-frame encoder map plus candidate coordinates → bilinear7×7sampling → spatial mean → one RoI vector. A separate measured original-study deep-tail case demonstrates useful phrase evidence losing rank during learned fusion; no later-model rescue is asserted.

Planned speaking cues total1,800seconds with75seconds reserved for the existing title script. Rehearsal remains outstanding. All28rendered slides inspected; independent technical and integrated visual reviews passed. First11slides unchanged during this refinement. [Current artifact record and sources](../artifacts/defense-committee-refinement-20260928/README.md), [new-figure preview](../artifacts/defense-committee-refinement-20260928/index.html). Native backup, source hashes, raw snapshots and validation retained.

Historical opening transplant: [artifact record](../artifacts/defense-proposal-opening-20260928/README.md), [earlier speaker cues](../artifacts/defense-proposal-opening-20260928/speaker-notes.md). Initial September28 build: [earlier artifact record](../artifacts/defense-20260928/README.md). These earlier sequences and timing allocations are superseded by the current refinement.

''' +s[end:]
s=s.replace('sources: [wiki/artifacts/defense-proposal-opening-20260928/final-presentation.json,','sources: [wiki/artifacts/defense-committee-refinement-20260928/after.json, wiki/artifacts/defense-proposal-opening-20260928/final-presentation.json,')
marker='## Historical September13 deck';s=s.replace(marker,'''### Priority critique responses implemented, September28 14:28 EDT

The earlier audit below is a snapshot, not the current completion state. The primitive-conditioned composition explanation, high-level RoIAlign mechanics and one measured original-study tail-ranking failure are now in the main deck. Explicit readout labels and concise methodology rationale were added. Broader partial-metric published comparisons, unsuccessful-approach backups and actual rehearsal remain open. See the current artifact record above.

'''+marker,1);page.write_text(s)
p=W/'tools/defense-diagram-reference.md';s=p.read_text()+'''\n\n## Committee explanation figures (September28)\n\nNew editable derivatives in `artifacts/defense-committee-refinement-20260928/`: `composition.drawio`, `roi-align.drawio`, `tail-case.drawio`. Clone the established encoder shapes, badges, palette and photo provenance. Composition and RoIAlign use the2520×1080 family canvas and exact existing slide transforms. Tail-case is a separate qualitative photo/table figure, aspect-fitted to the same allocated frame. Do not label standalone contextual184-output heads as primitive-conditioned CompMLPs. RoIAlign uses unpadded evaluation coordinates on a16×16keyframe feature map,7×7bilinear pooling and spatial mean; padded pixel crops are a distinct branch. The bus example is illustrative GT. The LarVeh-Stop-Jun case is a measured original controlled seed0 ranking diagnostic, not a verified contextual/DCB rescue. Technical, image-first visual and integrated slide reviews and all source hashes are retained.\n''';p.write_text(s)
p=W/'index.md';s=p.read_text().replace('September28 committee-coverage audit and NCAT template with aligned sentence-case opening titles, minimalist bullets, aligned draw.io evolution, contextual/DCB evidence, and 30-minute speaker cues','September28 committee audit and refinements: primitive-to-triplet explanation, illustrative RoIAlign, measured tail case,25main slides +3architecture backups,30-minute planned cues');p.write_text(s)
with (W/'log.md').open('a') as f:f.write('''\n## 2026-09-28 14:28 EDT — Committee-priority defense refinements\n\nAdded live slides14composition,15RoIAlign and21verified tail-ranking case. Stage5/6 labels distinguish primitive-conditioned learned composition from contextual standalone184-output readout. Preserved first11slides exactly; retained originalStage6/attention/contrastive views as skipped backups26–28, keeping25main slides. Planned allocations total30minutes including75seconds title allowance; rehearsal unverified. Native backup, code/source citations, draw.io lineage, PNGs, all28native renders, independent technical/visual reviews and change validation in `artifacts/defense-committee-refinement-20260928/`. Updated [[directions/final-defense-30-minute-deck]], [[tools/defense-diagram-reference]] and index.\n\nPeriodic read-only NCShare stdout checks: tasks732171_4/5 progressed from approximately11,300/11,200frames at14:15 to11,350/11,250at14:26, of36,717. These samples showed continued progress; no new completion or full monitor-health claim. No cluster changes or desktop notifications.\n''')
print('Documentation, preview and provenance saved')

from pathlib import Path
from PIL import Image
import json,hashlib,datetime,base64
from zoneinfo import ZoneInfo
A=Path(__file__).resolve().parent;R=A.parents[1]
a=json.load(open(A/'final-presentation.json'))['structuredContent'];b=json.load(open(A/'before.json'))['structuredContent'];by={s['objectId']:s for s in a['slides']}
images=[next(e for e in by[k]['pageElements'] if 'image'in e) for k in ['rev_12','rev_13','rev_14','rev_15']]
assert all(e['size']==images[0]['size'] and e['transform']==images[0]['transform'] for e in images)
new_ids=[s['objectId'] for s in a['slides'] if s['objectId'] not in {s['objectId'] for s in b['slides']}]
checks=[]
for n in [4,5,6]:
 p=A/f'stage{n}.png';lint=json.load(open(A/f'stage{n}-lint.json'))
 assert lint['errors']==0
 assert all(f['code']=='W-OVERLAP' for f in lint['findings'])
 checks.append({'stage':n,'size':Image.open(p).size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'intentional_overlaps':lint['warnings'],'unresolved_errors':0,'unresolved_warnings':0,'accepted_overlap_types':['model corner badges','stacked photos','annotation rectangles on photos']})
(A/'export-check.json').write_text(json.dumps(checks,indent=2))
val={'slide_count_observed':len(a['slides']),'concurrent_nonagent_added_slide_ids':new_ids,'modified_slide_ids':['rev_12','rev_13','rev_14'],'native_image_size_equal_to_contextual':True,'native_image_transform_equal_to_contextual':True,'native_image_size':images[0]['size'],'native_image_transform':images[0]['transform'],'canvas':[2520,1080],'native_exports':[2522,1082],'modified_slide_renders_reviewed':3,'contextual_comparison_render_reviewed':True,'technical_review':'passed','visual_review':'passed','unresolved_structural_errors':0,'revision':a['revisionId']}
(A/'validation.json').write_text(json.dumps(val,indent=2))
(A/'backup.json').write_text(json.dumps({'id':'15YkdKgoSJEh_0lddgEhhM-pVPsEX0uTLAaHVrjfRbf0','url':'https://docs.google.com/presentation/d/15YkdKgoSJEh_0lddgEhhM-pVPsEX0uTLAaHVrjfRbf0/edit','title':'Thesis Defense — Before illustrated Stages 4–6 — 2026-09-28'},indent=2))
(A/'index.html').write_text('<!doctype html><html><meta charset="utf-8"><title>Illustrated Stages 4–6</title><style>body{font:18px Arial;background:#182737;color:white;margin:2rem}figure{background:white;color:#26323d;margin:2rem auto;max-width:1500px;padding:1rem}img{width:100%}a{color:#edb93f}</style><h1>Illustrated Stages 4–6</h1><p>Original models in the contextual diagrams’ visual style. Static preview.</p>'+''.join(f'<figure><img src="data:image/png;base64,{base64.b64encode((A/f"stage{n}.png").read_bytes()).decode()}"><figcaption>Stage {n} · <a href="stage{n}.drawio">Editable draw.io</a></figcaption></figure>' for n in [4,5,6])+'</html>')
(A/'README.md').write_text('''# Illustrated Stages 4–6, September 28

Updated in the working defense: https://docs.google.com/presentation/d/1ma7UQGy-keAukf1w4L1EdSYgL4ehmwDBCsn62tccEL4/edit

Native backup: https://docs.google.com/presentation/d/15YkdKgoSJEh_0lddgEhhM-pVPsEX0uTLAaHVrjfRbf0/edit

Brandon requested Stage4/5/6 diagrams more similar to the later contextual figures. Derivatives reuse the same ROAD-Waymo photos, proportional crop stacks, teal/purple encoders, one-row blue crop vectors, purple phrase matrices, gold C, fire/ice badges, and rounded black arrows. Original architecture sources remain preserved under artifacts/defense-30min/alignment/; common contextual vocabulary comes from contextual-evolution-20260924/simplified/01-mlp.drawio. Exact hashes and image provenance:provenance.json.

Files:stage4/5/6.drawio are editable sources; .svg embeds diagram XML; .png are slide exports. index.html is a self-contained static review page. Source canvas2520×1080; native PNG2522×1082, exactly matching the existing contextual asset normalization in Slides. Every shared Stage4–6 station has zero coordinate delta. Native sizes and transforms exactly equal the existing contextual image, not just approximately equal bounds. Shared photo/crop/video/phrase stations also retain the contextual coordinates where present.

Original model distinctions:Stage4 calibrated cosine phrase classifier; Stage5 raw flat primitives+crop feed CompMLP, with action/location bypass; Stage6 adds raw phrase composition probabilities to the same MLP. No contextual-only RoIAlign, full-frame feature, position, residual or attention branches were introduced. Video-clip photos depict crop construction. Notes preserve dimensions, sigmoid/pre-confidence scores, GT training versus YOLO inference, separate OOF head training, and retained YOLO agentness/agent scores. No q² weighting. Photograph annotations are illustrative GT, not measured detector outputs.

Technical and rendered-image reviews passed. Structural checks have zero errors/crossings/through-node routes; only intentional badge/photo/annotation overlaps remain. All three final native slide renders and the neighboring contextual render were personally inspected. Labels use the same density as the approved contextual reference; full-slide placement retained. Existing small citation warnings elsewhere in the deck remain unchanged.

Changed only figure/caption/notes on page IDs rev_12,rev_13,rev_14. A title slide was added by concurrent editing, increasing the observed deck count from25 to26; preserved without renumbering or deleting it. Native page IDs are stable despite shifted ordinal positions. No new slide was created by this restyle.

Research monitoring sample during work:Stage6 task4 reached11,025/36,717 and task5 reached10,925/36,717; both advanced. No jobs, alerts, protocol or workbook changes.

Brandon asked how to recolor red slide rules:both lines live on master p20 (p20_i13,p20_i14); title gold is #FDB928. Explained Slide→Edit theme→select master→select each line→Line color→Custom. No theme colors changed by this task.
''')
file=R/'tools/defense-diagram-reference.md';s=file.read_text();s=s.replace('updated: 2026-09-24','updated: 2026-09-28');s+='''

## Illustrated original Stages 4–6 (September 28)

Brandon requested the original models match the newer contextual diagrams visually. Current derivatives: `artifacts/defense-stage456-illustrated-20260928/stage4.drawio`, `stage5.drawio`, `stage6.drawio`, with PNG/SVG and a static HTML preview. Reuse the contextual series’ photos/crops, teal/purple encoders, single-row feature vectors, phrase matrix, gold C and badges. These supersede the older schematic assets only in the current defense’s Stage4–6 slides; historical figures remain preserved.

The architectures remain original:Stage4 calibrated phrase scoring; Stage5 flat primitives+crop→CompMLP; Stage6 adds phrase composition scores. Do not import contextual-only full-frame features, RoIAlign, box position, attention or residuals into these models. The displayed full video clip solely supplies crop pixels. Head outputs are sigmoid probabilities before detector-confidence weighting; YOLO agentness/agent are retained separately.

All common stations are fixed within the three figures. Shared input/crop/encoder/phrase stations reuse the contextual coordinates; all four families use the same2520×1080 canvas and exact native image size/transform. Final native PNG normalization is2522×1082. Technical and PNG-first visual reviews passed; original/code/photo hashes, intentional overlaps, structural checks and final Slides renders are retained in the artifact directory.
''';file.write_text(s)
f=R/'directions/final-defense-30-minute-deck.md';s=f.read_text();marker='## Historical September13 deck';add='''### Illustrated Stage4–6 update, September28

The Stage4/5/6 figures now match the contextual models’ illustrative vocabulary and exact slide image placement. Model topology is unchanged; separate flat/composition paths and phrase fusion remain explicit. Editable draw.io sources, technical/visual reviews and [static preview](../artifacts/defense-stage456-illustrated-20260928/index.html) are linked in the [artifact record](../artifacts/defense-stage456-illustrated-20260928/README.md). A title slide added during concurrent editing was preserved; the observed live count is26, while the prior25-slide plan remains the historical baseline. Page IDs rev_12/13/14 identify the updated models regardless of shifted positions.

''';s=s.replace(marker,add+marker,1);f.write_text(s)
f=R/'index.md';s=f.read_text();lines=s.splitlines()
for i,l in enumerate(lines):
 if '[[final-defense-30-minute-deck]]' in l:lines[i]=l+' Stage4–6 now use the illustrated contextual visual style.'
f.write_text('\n'.join(lines)+'\n')
t=datetime.datetime.now(ZoneInfo('America/New_York')).isoformat(timespec='seconds')
with (R/'log.md').open('a') as f:f.write(f'\n\n## {t} — Illustrated Stage4–6 defense diagrams\n\nUpdated [[final-defense-30-minute-deck]] and [[defense-diagram-reference]]: same contextual photos, encoder shapes, tensors, badges and exact native image transforms; original code-verified model branches preserved. Technical and visual crucibles passed; three integrated Slides renders verified. Native backup + editable sources/provenance: artifacts/defense-stage456-illustrated-20260928/. Concurrent added title slide preserved (observed26 slides). No diagram animations or scientific code changed. NCShare log sample:task4 11,025/36,717;task5 10,925/36,717, both progressing. Explained master-rule recoloring to #FDB928 at user request; no theme color mutation.\n')
print(json.dumps(val))

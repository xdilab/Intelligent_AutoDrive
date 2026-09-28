import json,pathlib,re,datetime,html
from PIL import Image,ImageDraw
w=pathlib.Path('/data/repos/wiki/artifacts/defense-20260928');root=w.parent.parent
p=json.load(open(w/'final-presentation.json'))['structuredContent'];plan=json.load(open(w/'slide-plan.json'));slides=p['slides']
assert [s['objectId'] for s in slides]==[f'def_{i:02d}' for i in range(1,26)]
for s in slides:
 assert not s['slideProperties'].get('isSkipped',False)
 notes=s['slideProperties']['notesPage'];oid=notes['notesProperties']['speakerNotesObjectId'];e=next(e for e in notes['pageElements'] if e['objectId']==oid)
 text=''.join(x.get('textRun',{}).get('content','') for x in e['shape']['text']['textElements']);assert len(text.strip())>20
for group in [[9,10],[11,12,13,14,15]]:
 es=[next(e for e in slides[i-1]['pageElements'] if e['objectId']==f'def_{i:02d}_image') for i in group]
 assert all(e['transform']==es[0]['transform'] and e['size']==es[0]['size'] for e in es)
sec=sum(int(re.search(r'(\d+) seconds',s['notes']).group(1)) for s in plan);assert sec==1800
(w/'validation.json').write_text(json.dumps({'slide_count':25,'populated_notes':25,'planned_seconds':sec,'ordered_ids':True,'all_unskipped':True,'diagram_family_transforms_equal':True,'native_issue_count':0,'renders_reviewed':25,'technical_review':'passed with corrections','visual_review':'passed after corrections','render_fallback':'native Google Slides thumbnails; Drive PDF helper could not materialize export'},indent=2))
# Minimal offline visual preview; native Slides remains the editable source.
body=''.join(f'<section><h2>{i}. {html.escape(plan[i-1]["title"])}</h2><img src="renders/slide-{i:02d}.png" alt="Slide {i}"></section>' for i in range(1,26))
(w/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Thesis defense · 25 slides</title><style>body{font:16px Arial;background:#eee;max-width:1100px;margin:30px auto}section{margin:35px 0}img{width:100%;box-shadow:0 3px 16px #aaa}h2{font-size:18px}</style><h1>Thesis defense · September 28, 2026</h1>'+body)
(w/'speaker-notes.md').write_text('# Defense speaker cues · 25 slides · 30-minute target\n\n'+'\n\n'.join(f'## {s["number"]}. {s["title"]}\n\n{s["notes"]}' for s in plan))
for start in [1,9,17,25]:
 ns=range(start,min(start+8,26));im=Image.new('RGB',(1600,470*((len(ns)+1)//2)),'#dedede');d=ImageDraw.Draw(im)
 for j,n in enumerate(ns):
  page=Image.open(w/'renders'/f'slide-{n:02d}.png').convert('RGB');page.thumbnail((790,445));x=(j%2)*800;y=(j//2)*470;im.paste(page,(x,y+20));d.text((x+5,y+3),str(n),fill='black')
 im.save(w/f'contact-{start:02d}.png')
(w/'README.md').write_text('''# Thesis defense, September 28, 2026

Working deck: https://docs.google.com/presentation/d/1ma7UQGy-keAukf1w4L1EdSYgL4ehmwDBCsn62tccEL4/edit

Backup: https://docs.google.com/presentation/d/1hC3JCpZBOOErHRcnDgq6doAOnE2DjdVfcEKH_7eiqZI/edit

25 slides, including title, outline and closing. Requested sequence: Motivation; Related Works & Limitations; Methodology; Results; Conclusion. Native text/tables and 30-minute speaker-cue pacing. Rehearsal and advisor review remain open.

Style reference: downloaded `AAAI_Ehsan_CL Presntation_v3_HM.pptx`; linked native template was backed up before edits. Native exemplar mappings and media disposition recorded before mutations in slide-plan.json. Original native runs retained in source-style-snapshot.json. Retained NCAT/lab branding; removed unrelated medical figures, authors and sponsor logos. Architecture PNGs reuse approved draw.io ancestors, hashes in plan and image alt text; exact shared image transforms verified for Stage5/6 and contextual families. Narrative remains editable native text, not screenshot slides.

All 25 native renders inspected; independent technical and visual reviews completed. Corrected title color/font, rendered math, citations, Stage2 RoIAlign caption and scientific qualifications. Final issue checker reports zero issues; custom verification confirms count/order, populated notes, 1800-second pacing and equal family transforms. Native PDF export helper failed to materialize bytes; used native LARGE Slides thumbnails for complete final visual QA. index.html is a static preview, not the editable source.

Claims: DCB has strongest completed contextual mean triplet AP (13.43), focal blend retains better tail (6.49 vs6.23); DCB primary standalone-tail endpoint was not met. Language tail control evidence is exploratory and semantic/geometry-confounded. FRCB selected result is distinct from our three-seed mean and not a matched superiority comparison. Stage5 selected epoch3 and Stage6 interim epoch1 detector rows are single-seed; selected Stage6 detector evaluation remains pending. Historical frozen86 recipe not in main narrative.

Sources, full numbers and notes: slide-plan.json, speaker-notes.md, selection.json references, and final-presentation.json. No workbook, experiment, repository or notification-service mutation performed.
''')
f=root/'directions/final-defense-30-minute-deck.md';t=f.read_text().replace('title: "Final defense: 30-slide working deck"','title: "Final defense: 25-slide working deck"').replace('updated: 2026-09-14','updated: 2026-09-28');t=t.replace('# Final defense: 30-slide working deck','# Final defense: 25-slide working deck',1)
marker='\n## Current working deck, September 28, 2026\n\n'
section='''Brandon selected **25 slides total**, following Motivation → Related Works & Limitations → Methodology → Results → Conclusion, with minimalist native bullets and a 30-minute speaker-cue plan. [Open the current editable deck](https://docs.google.com/presentation/d/1ma7UQGy-keAukf1w4L1EdSYgL4ehmwDBCsn62tccEL4/edit). The downloaded NCAT template is retained; a native backup was created before edits. Prior September13 deck below remains historical.

Includes approved draw.io Stage5/6 and contextual evolution assets, Stage2 RoIAlign, ground-truth training versus YOLO evaluation, 184 outputs and frame-mAP equations, original/contextual/language-control/blend/DCB tables, a qualified published comparison, and dated interim ROAD adaptation results. Final selected Stage6 evaluation remains pending. DCB leads completed contextual triplet means, but its original standalone-tail target was not met; focal blend retains higher tail AP. Language evidence remains exploratory and does not isolate semantics from prototype geometry.

Validation: 25 native slides, 25 note sections, all rendered and inspected, independent technical/visual reviews, identical family image transforms, zero automated output issues. [Artifact and provenance record](../artifacts/defense-20260928/README.md), [speaker cues](../artifacts/defense-20260928/speaker-notes.md), [static preview](../artifacts/defense-20260928/index.html). Timed rehearsal and advisor review remain outstanding.

## Historical September13 deck
'''
pos=t.index('\n**Current deadlines');t=t[:pos]+marker+section+t[pos:];f.write_text(t)
f=root/'index.md';t=f.read_text();t=t.replace('[[directions/final-defense-30-minute-deck|30-minute defense working deck]] — copied original into 30 slides with short timed notes, Stage 4–6 figures, and one headline model per paper in comparison tables','[[directions/final-defense-30-minute-deck|25-slide defense working deck]] — September28 NCAT template, minimalist bullets, aligned draw.io evolution, contextual/DCB evidence, and 30-minute speaker cues');f.write_text(t)
now=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
with open(root/'log.md','a') as f:f.write(f'\n\n## {now} — 25-slide thesis defense built and reviewed\n\nUpdated Brandon’s supplied native Slides working deck in place after a native backup. Used downloaded NCAT template; 25 slides follow requested five sections, concise native bullets/tables, approved draw.io figures and source hashes, short cues totaling1800seconds. Completed independent technical and rendered-image reviews; verified count/order/notes, exact family transforms, and zero automated issues. All25 native thumbnails reviewed because PDF helper could not materialize export bytes. Preserved overall-versus-tail tradeoffs, DCB failed primary tail target, exploratory language/geometry caveat, published protocol differences and pending selected Stage6 detector evaluation. No experiments/workbook/service changes. Live log check during work observed selected Stage6 progress10400/10225 of36717. Evidence: artifacts/defense-20260928/; current deck link and historical distinction updated in directions/final-defense-30-minute-deck.md and index.md. Rehearsal/advisor review pending.\n')
print('Verified 25 slides, 25 notes, 1800s, family alignment; documented artifacts and wiki.')

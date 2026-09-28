import json,pathlib,datetime,html
from zoneinfo import ZoneInfo
W=pathlib.Path('/data/repos/wiki/artifacts/defense-proposal-opening-20260928');root=W.parents[1];d=json.load(open(W/'final-presentation.json'))['structuredContent'];p=json.load(open(W/'proposal.json'))['structuredContent'];notes=json.load(open(W/'speaker-notes.json'))
def strings(s):return [''.join(t.get('textRun',{}).get('content','') for t in e.get('shape',{}).get('text',{}).get('textElements',[])).strip() for e in s['pageElements'] if 'shape'in e]
def norm(t):return ' '.join(t.split())
missing=[]
for i in range(11):
 target=norm(' '.join(strings(d['slides'][i])))
 for t in strings(p['slides'][i]):
  if norm(t) not in target:missing.append([i+1,t])
assert not missing
assert len(d['slides'])==25
assert sum(x['seconds'] for x in notes)==1800
ims=[next(e for e in d['slides'][i]['pageElements'] if 'image'in e) for i in [11,12,13]]
assert all(e['transform']==ims[0]['transform'] and e['size']==ims[0]['size'] for e in ims)
validation={'slides':25,'proposal_opening_slides':11,'missing_source_text':missing,'stage4_starts_at':12,'stage4_5_6_identical_image_transforms':True,'speaker_seconds':1800,'native_slides_visually_reviewed':25,'revision':d['revisionId'],'review':'Local visual and source-text review; inherited scientific evidence retained, no new experiment conclusions'}
(W/'validation.json').write_text(json.dumps(validation,indent=2))
(W/'speaker-notes.md').write_text('# Revised 25-slide defense cues\n\n'+ '\n\n'.join(f"## Slide {n['number']}\n\n{n['notes']}" for n in notes))
(W/'index.html').write_text('<!doctype html><html><meta charset="utf-8"><title>Defense: proposal opening revision</title><style>body{background:#142131;color:white;font:18px Arial;margin:2rem}figure{max-width:1200px;margin:2rem auto}img{width:100%}figcaption{padding:1rem}</style><h1>Thesis defense — 25 slides</h1><p>Proposal content on slides 1–11; architecture progression starts at Stage 4.</p>'+''.join(f'<figure><img src="renders/slide-{i:02}.png"><figcaption>Slide {i}</figcaption></figure>' for i in range(1,26))+'</html>')
(W/'README.md').write_text('''# Defense revision: proposal opening, September 28

Working deck: https://docs.google.com/presentation/d/1ma7UQGy-keAukf1w4L1EdSYgL4ehmwDBCsn62tccEL4/edit

Backup before this revision: https://docs.google.com/presentation/d/1Md92eF-_-xs-lK1Hvq-4TbtJ_YslJ5q1P5bO0ob6DH4/edit

Brandon asked for identical **content** to proposal slides 1–11, separate from styling, and methodology beginning at Stage 4. All nonempty source text blocks are preserved, verified after whitespace normalization, including original proposal outline and wording. Native editable text uses the defense template’s type and branding. Existing meaning-bearing source images were reused with aspect-preserving placement; the benchmark asset is fully fitted rather than retaining its former crop. Source title background was decorative and omitted; source logos remain. Source JSON, media hashes, slide mapping and pre-edit deck are retained here.

Final order: proposal opening 1–11; Stage4/5/6 12–14; contextual MLP/attention/contrastive 15–17; blend+DCB18; evaluation19; contextual results20; language controls21; blend results22; published comparison23; limitations24; conclusion/questions25. Original baseline triplet means are retained on20; interim ROAD adaptation details are in24’s notes, with pending status visible. Separate Stage2 and intermediate baseline slides are no longer in the main sequence; the pre-edit native backup preserves them.

Validation:25 native slides,25 speaker cues totaling1,800seconds, zero missing source text blocks, exactly equal Stage4/5/6 image sizes/transforms. All25 native renders inspected; title contrast, source-footer clearance and input/output spacing repaired. Three automated warnings identify small source/footer text; no errors. No new independent sub-agent review was requested for this content transplant. Static preview:index.html. Native diagrams are existing approved draw.io exports; no new architecture was drawn.

Research logs checked during work: NCShare Stage6 task4 reached10,750/36,717 frames; task5 reached10,650/36,717. Both advanced beyond the prior snapshot; last sampled lines contain progress, not failures. No jobs, monitors, alerts or scientific protocols changed.
''')
f=root/'directions/final-defense-30-minute-deck.md';s=f.read_text();a=s.index('## Current working deck, September 28, 2026');b=s.index('## Historical September13 deck',a)
section='''## Current working deck, September 28, 2026

[Open the current editable 25-slide deck](https://docs.google.com/presentation/d/1ma7UQGy-keAukf1w4L1EdSYgL4ehmwDBCsn62tccEL4/edit). Brandon’s latest revision preserves **proposal slides1–11 in content**, separate from styling, and starts the architecture progression at **Stage4 on slide12**, followed by Stage5/6 on13/14. Existing NCAT defense styling and source citations remain. The original proposal is untouched.

The remaining sequence covers contextual MLP, attention, all184 contrastive learning, expert blending/DCB, evaluation, contextual and language-control results, published comparison, limitations and conclusion. To stay at25 slides, baseline triplet means are on slide20 and interim adaptation details are in slide24’s notes. Selected Stage6 detector evaluation remains visibly pending. The separate Stage2 RoIAlign slide is retained in the pre-edit backup; contextual figures still illustrate RoIAlign. Proposal wording on the opening outline is intentionally retained verbatim.

Validation:25 native slides and25 speaker cues totaling1,800seconds; no missing proposal text blocks; identical Stage4–6 image transforms; all25 renders reviewed. Small citation text produces three accepted automated warnings, with no errors. This content transplant received local visual/source checks; the initial draft’s independent technical/visual reviews remain historical, not a claim of a second independent review. [Current artifact record](../artifacts/defense-proposal-opening-20260928/README.md), [speaker cues](../artifacts/defense-proposal-opening-20260928/speaker-notes.md), [static preview](../artifacts/defense-proposal-opening-20260928/index.html). Timed rehearsal and advisor review remain outstanding.

Initial September28 build and its earlier25-slide sequence are preserved in [the earlier artifact record](../artifacts/defense-20260928/README.md). Native backups are linked in each record. Research evidence has not changed: DCB leads completed contextual triplet means, focal blend retains better tail AP, and language controls do not isolate semantics from geometry.

'''
s=s[:a]+section+s[b:];s=s.replace('sources: [wiki/artifacts/defense-30min/final-presentation.json, wiki/artifacts/defense-30min/comparison-sources.md]','sources: [wiki/artifacts/defense-proposal-opening-20260928/final-presentation.json, wiki/artifacts/defense-proposal-opening-20260928/proposal.json, wiki/artifacts/defense-30min/comparison-sources.md]');f.write_text(s)
t=datetime.datetime.now(ZoneInfo('America/New_York')).isoformat(timespec='seconds')
with (root/'log.md').open('a') as f:f.write(f'\n\n## {t} — Defense opening restored to proposal content\n\nUpdated [[final-defense-30-minute-deck]] per Brandon: proposal slides1–11 content preserved in current styling; methodology begins Stage4 on12, then Stage5/6. Kept25 slides and1,800-second cues. Backed up native deck, preserved raw source/readback/media hashes, verified no source text omissions and identical Stage4–6 figure transforms, inspected25 native renders, repaired contrast/spacing. Records: artifacts/defense-proposal-opening-20260928/. No research jobs or workbook changes. During-session log sample: NCShare Stage6 task4 10,750/36,717, task5 10,650/36,717; both advancing.\n')
f=root/'index.md';s=f.read_text();needle='[[final-defense-30-minute-deck]]';lines=s.splitlines()
for i,line in enumerate(lines):
 if needle in line and 'proposal opening' not in line:lines[i]=line+' Latest revision: proposal opening slides1–11; methodology starts at Stage4.'
f.write_text('\n'.join(lines)+'\n')
print(json.dumps(validation))

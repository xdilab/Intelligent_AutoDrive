"""Build a standalone, offline evidence report from verified diagnostic artifacts."""
from pathlib import Path
import base64,csv,json,hashlib,datetime
from zoneinfo import ZoneInfo
A=Path(__file__).resolve().parent.parent;H=A/'html';D=A/'collected'
def readcsv(p):return list(csv.DictReader(p.open()))
def b64(p):return 'data:'+('image/png' if p.suffix=='.png' else 'image/jpeg')+';base64,'+base64.b64encode(p.read_bytes()).decode()
report=json.loads((D/'report.json').read_text());assert report['parity_pass']
context=readcsv(A/'all-triplet-context.csv')
for r in context:
 for k in r:
  if k=='class':continue
  r[k]=r[k]=='True' if k in ['tail47','deep28'] else float(r[k])
budgets=readcsv(D/'budget-summary.csv')
for r in budgets:
 for k in r:
  if k!='class':r[k]=float(r[k])
allcases=json.loads((D/'examples.json').read_text());selected={}
# Stored examples are already selected by rank gap; retain first representative per group.
for r in allcases:selected.setdefault((r['class'],r['seed'],r['category']),r)
cases=list(selected.values());images={};image_sources=[]
for r in cases:
 stem=r['frame']
 if stem in images:continue
 video,fid=stem.rsplit('_',1);p=Path('/data/datasets/ROAD_plusplus/rgb-images')/video/f'{int(fid):05d}.jpg';images[stem]=b64(p);image_sources.append({'frame':stem,'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
# Keep the two previously published visual cases in the explorer as the exact featured examples.
featured=json.loads((A/'selected-visual-cases.json').read_text())
for f in featured:
 if not any(r['frame']==f['frame'] and r['seed']==f['seed'] and r['class']==f['class'] and r['category']==f['category'] for r in cases):cases.append(f)
 if f['frame'] not in images:images[f['frame']]=b64(Path(f['source_frame']));image_sources.append({'frame':f['frame'],'path':f['source_frame'],'sha256':f['source_frame_sha256']})
raw_ap=[r for r in readcsv(A.parent/'post-proposal-experiments/per-class-all-runs.csv') if r['head']=='triplet' and r['run'].startswith('seed')]
payload={'report':report,'context':context,'budgets':budgets,'cases':cases,'images':images,'ap':raw_ap,'allClassSummary':json.loads((A/'all-triplet-context.json').read_text()),'parity':readcsv(D/'ap-parity.csv'),'image_sources':image_sources,'featured':featured}
downloads={name:(D/name).read_text() for name in ['budget-summary.csv','ap-parity.csv','report.json','examples.json']};downloads['all-triplet-context.csv']=(A/'all-triplet-context.csv').read_text();payload['downloads']=downloads
stamp=datetime.datetime.now(ZoneInfo('America/New_York')).strftime('%B %d, %Y · %I:%M %p EDT')
template=(H/'template.html').read_text();template=template.replace('__BUILT__',stamp).replace('__DATA__',json.dumps(payload,separators=(',',':')).replace('</','<\\/'))
template=template.replace('__LOST_IMAGE__',b64(A/'case-larveh-stop-jun-phrase_help_lost.png')).replace('__KEPT_IMAGE__',b64(A/'case-bus-stop-vehlane-phrase_help_preserved.png'))
out=A/'language-fusion-analysis.html';out.write_text(template)
(H/'build-provenance.json').write_text(json.dumps({'built':stamp,'output':str(out),'output_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'embedded_case_count':len(cases),'embedded_source_frames':len(images),'source_frame_hashes':image_sources,'sources':{str(p.relative_to(A)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [D/'report.json',D/'budget-summary.csv',D/'examples.json',D/'ap-parity.csv',A/'all-triplet-context.csv',H/'template.html']}},indent=2)+'\n')
print(json.dumps({'html':str(out),'bytes':out.stat().st_size,'cases':len(cases),'photos':len(images)},indent=2))

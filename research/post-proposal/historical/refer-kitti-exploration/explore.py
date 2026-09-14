from pathlib import Path
import json,collections,hashlib,html
root=Path('/data/datasets/Refer-KITTI');out=Path(__file__).resolve().parent
cache={};missing=[];refs=0;expr=[];pairs=set();keywords=['turning','braking','parking','moving','walking','standing','left','right','direction']
for p in sorted((root/'expression').rglob('*.json')):
 d=json.loads(p.read_text());v=p.parent.name;s=d['sentence']
 for f,ids in d['label'].items():
  key=(v,int(f));pairs.add(key)
  if key not in cache:
   q=root/'labels_with_ids/image_02'/v/f'{int(f):06d}.txt'
   cache[key]={int(float(line.split()[1])) for line in q.read_text().splitlines() if line.strip()} if q.exists() else set()
  for oid in ids:
   refs+=1
   if int(oid) not in cache[key]:missing.append([v,f,oid,s])
 expr.append({'video':v,'sentence':s,'frames':len(d['label']),'path':str(p.relative_to(root))})
raw=[(p.parent.name,json.loads(p.read_text())) for p in (root/'expression').rglob('*.json')]
counts={w:{'expressions':sum(w in e['sentence'].split() for e in expr),'videos':len({e['video'] for e in expr if w in e['sentence'].split()})} for w in keywords}
a={'unique_sentences':len({e['sentence'] for e in expr}),'unique_video_target_maps':len({(v,json.dumps(d['label'],sort_keys=True)) for v,d in raw}),'six_behavior_token_union':sum(any(w in d['sentence'].split() for w in ['turning','braking','parking','moving','walking','standing']) for _,d in raw),'expressions':len(expr),'videos':len({e['video'] for e in expr}),'unique_labeled_video_frames':len(pairs),'expression_frame_object_references':refs,'missing_box_references':len(missing),'missing_examples':missing[:10],'keyword_counts':counts,'archive_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'downloads').glob('*.zip') if p.stat().st_size<10000000},'expressions_index':expr}
(out/'annotation-audit.json').write_text(json.dumps(a,indent=2))
# Select one actor with a large box in each behavior category; lexical selection is not semantic verification.
samples=[]
for word in ['turning','braking','parking','walking','standing','direction']:
 candidates=[]
 for e in expr:
  if word not in e['sentence'].split():continue
  d=json.loads((root/e['path']).read_text());fs=sorted(d['label'],key=int)
  f=fs[len(fs)//2];p=root/'labels_with_ids/image_02'/e['video']/f'{int(f):06d}.txt'
  boxes=[list(map(float,line.split())) for line in p.read_text().splitlines() if line.strip()]
  for b in boxes:
   if int(b[1]) in d['label'][f]:candidates.append((b[4]*b[5],e,f,b,d))
 _,e,f,b,d=max(candidates,key=lambda x:x[0]);oid=int(b[1]);frames=sorted(int(k) for k,ids in d['label'].items() if oid in ids)
 for frac in [.25,.5,.75]:
  frame=frames[min(len(frames)-1,int((len(frames)-1)*frac))]
  rows=[list(map(float,l.split())) for l in (root/'labels_with_ids/image_02'/e['video']/f'{frame:06d}.txt').read_text().splitlines() if l.strip()]
  box=next(r for r in rows if int(r[1])==oid)
  samples.append({'keyword':word,'sentence':e['sentence'],'video':e['video'],'frame':frame,'object_id':oid,'box':box,'annotation':e['path'],'image_member':f'training/image_02/{e["video"]}/{frame:06d}.png'})
(out/'samples.json').write_text(json.dumps(samples,indent=2));print({k:v for k,v in a.items() if k not in ['expressions_index','archive_sha256']})

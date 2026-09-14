import json,hashlib,time
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path('/work/bbyrd1/road-contrastive-pilot-20260914');cfg=json.loads((ROOT/'code/protocol.json').read_text());out=ROOT/'results';out.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def iou(b,g):
 if not len(g):return 0.
 inter=np.maximum(0,np.minimum(b[2:],g[:,2:])-np.maximum(b[:2],g[:,:2])).prod(1);union=(b[2:]-b[:2]).prod()+(g[:,2:]-g[:,:2]).prod(1)-inter
 return float((inter/np.maximum(union,1e-9)).max())
print('Loading source annotations',flush=True);d=json.loads(Path(cfg['annotation']).read_text());prep=json.loads((Path(cfg['expert_study'])/'results/preparation.json').read_text());manifest=json.loads((Path(cfg['original_study'])/'code/shared-frames.json').read_text());groups=['agent','action','loc','duplex','triplet'];labels={h:d[h+'_labels'] for h in groups};assert labels==manifest['labels'];remap={h:{i:labels[h].index(n) for i,n in enumerate(d['all_'+h+'_labels']) if n in labels[h]} for h in groups};sizes=[len(labels[h]) for h in groups];assert sizes==[10,22,16,49,86]
selected={'train':prep['videos']['expert'][:cfg['train_videos']],'dev':prep['videos']['development'][:cfg['dev_videos']]};official={s.rsplit('_',1)[0] for s in manifest['frames']};assert not set(selected['train'])&set(selected['dev']);assert not (set(selected['train'])|set(selected['dev']))&official
result={'protocol':cfg,'videos':selected,'labels':labels,'train':[],'dev':[]};counts={};file_count=0
for split,vids in selected.items():
 for v in vids:
  vid=d['db'][v];assert 'train' in vid['split_ids'];fs=sorted(int(k) for k,f in vid['frames'].items() if f.get('annotated') and f.get('annos'));assert len(fs)>=cfg['frames_per_video'];chosen=[fs[j] for j in np.linspace(0,len(fs)-1,cfg['frames_per_video']).round().astype(int)]
  for fid in chosen:
   fr=vid['frames'][str(fid)];gt=[];ys=[];ids=[]
   for aid,ann in sorted(fr['annos'].items()):
    b=np.clip(np.array(ann['box'],float),0,1)
    if not (b[2]>b[0] and b[3]>b[1]):continue
    y=np.zeros(184);y[0]=1;off=1
    for h in groups:
     for j in ann[h+'_ids']:
      if j in remap[h]:y[off+remap[h][j]]=1
     off+=len(labels[h])
    gt.append(b);ys.append(y);ids.append(aid)
   assert gt;allgt=np.array(gt);order=sorted(range(len(gt)),key=lambda j:hashlib.sha256(f'road-pilot:{v}:{fid}:{ids[j]}'.encode()).hexdigest())[:cfg['max_gt_per_frame']];boxes=[gt[j] for j in order];targets=[ys[j] for j in order];annids=[ids[j] for j in order];rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'{v}:{fid}'.encode()).digest()[:8],'little'))
   for _ in range(200):
    if len(boxes)==len(order)+cfg['background_per_frame']:break
    w,h=rng.uniform(.02,.35,2);x,y=rng.uniform([0,0],[1-w,1-h]);b=np.array([x,y,x+w,y+h])
    if iou(b,allgt)<=.3:boxes.append(b);targets.append(np.zeros(184));annids.append(None)
   fids=[min(max(fid-3+j,1),vid['numf']) for j in range(8)];key_t=fids.index(fid);digests=[];dimensions=[]
   for f in fids:
    path=Path(cfg['frames'])/v/f'{f:05d}.jpg';raw=path.read_bytes();digests.append(hashlib.sha256(raw).hexdigest())
    with Image.open(path) as im:dimensions.append(list(im.size));im.verify()
    file_count+=1
   assert len(set(map(tuple,dimensions)))==1
   result[split].append({'video':v,'fid':fid,'fids':fids,'key_t':key_t,'boxes':np.array(boxes).tolist(),'targets':np.array(targets).tolist(),'annotation_ids':annids,'frame_sha256':digests,'frame_size':dimensions[0]})
 rows=np.concatenate([r['targets'] for r in result[split]]);counts[split]={'videos':len(vids),'keyframes':len(result[split]),'crop_rows':len(rows),'positive_rows':int(rows[:,0].sum()),'background_rows':int((rows[:,0]==0).sum()),'triplet_positive_counts':rows[:,98:].sum(0).astype(int).tolist(),'triplet_classes_present':int((rows[:,98:].sum(0)>0).sum())};assert rows[:,98:].sum()>0
result['source_sha256']={'annotation':sha(cfg['annotation']),'preparation':sha(Path(cfg['expert_study'])/'results/preparation.json'),'phrase_embeds':sha(Path(cfg['original_study'])/'data/phrase_embeds.pt'),'alphas':sha(Path(cfg['original_study'])/'data/flat_alphas.pt')};result['counts']=counts;result['image_reads_verified']=file_count;result['completed_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());(ROOT/'manifest.json').write_text(json.dumps(result,indent=2));(out/'preparation.json').write_text(json.dumps({k:v for k,v in result.items() if k not in ['train','dev']},indent=2));print(json.dumps(counts),flush=True)

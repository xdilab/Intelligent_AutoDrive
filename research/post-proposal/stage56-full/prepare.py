import json,pickle,hashlib,time
from pathlib import Path
import numpy as np
from PIL import Image
from base import sha
root=Path('/work/bbyrd1/stage56-full-20260914');cfg=json.loads((root/'code/protocol.json').read_text());prep=json.loads((Path(cfg['expert_study'])/'results/preparation.json').read_text());ann=json.loads(Path(cfg['annotation']).read_text());official=json.loads((Path(cfg['original_study'])/'code/shared-frames.json').read_text());splits={'train':prep['videos']['expert'],'dev':prep['videos']['development']};assert len(splits['train'])==420 and len(splits['dev'])==90;assert not set(splits['train'])&set(splits['dev']);assert not (set(splits['train'])|set(splits['dev']))&{k.rsplit('_',1)[0] for k in official['frames']}
lookup={v:s for s,vs in splits.items() for v in vs};(root/'data').mkdir(exist_ok=True);files={s:(root/f'data/{s}.jsonl').open('w') for s in splits};seen=set();hashes={};counts={s:{'frames':0,'rows':0,'positive_rows':0,'triplet_counts':np.zeros(86,dtype=np.int64)} for s in splits}
for shard in range(8):
 p=Path(cfg['source_cache'])/f'crop_feats_train.shard{shard}of8.pkl';print('SHARD',p,flush=True);d=pickle.load(p.open('rb'));assert not d['meta']['partial'];check=hashlib.sha256()
 for key in sorted(d['boxes']):
  b=d['boxes'][key];y=d['targets'][key];check.update(key.encode());check.update(b.astype(np.float32).tobytes());check.update(y.tobytes());v,f=key.rsplit('_',1)
  if v not in lookup:continue
  assert key not in seen;seen.add(key);split=lookup[v];fid=int(f);nf=ann['db'][v]['numf'];fids=[min(max(fid-3+j,1),nf) for j in range(8)];assert b.shape==(len(y),4) and y.shape[1]==184
  digests=[]
  for fj in fids:
   rel=f'{v}/{fj:05d}.jpg'
   if rel not in hashes:
    path=Path(cfg['frames'])/rel;raw=path.read_bytes();hashes[rel]=hashlib.sha256(raw).hexdigest()
    with Image.open(path) as im:im.verify()
   digests.append(hashes[rel])
  row={'video':v,'fid':fid,'fids':fids,'key_t':fids.index(fid),'boxes':b.tolist(),'positive_indices':[np.flatnonzero(t).tolist() for t in y],'frame_sha256':digests}
  files[split].write(json.dumps(row,separators=(',',':'))+'\n');c=counts[split];c['frames']+=1;c['rows']+=len(y);c['positive_rows']+=int(y[:,0].sum());c['triplet_counts']+=y[:,98:].sum(0).astype(np.int64)
 assert check.hexdigest()==d['meta']['row_sha256'];del d;print('SHARD_DONE',shard,flush=True)
for f in files.values():f.close()
for s in counts:counts[s]['triplet_counts']=counts[s]['triplet_counts'].tolist()
report={'protocol':cfg,'videos':splits,'counts':counts,'frame_files_verified':len(hashes),'data_sha256':{s:sha(root/f'data/{s}.jsonl') for s in splits},'source_sha256':{'annotation':sha(cfg['annotation']),'split':sha(Path(cfg['expert_study'])/'results/preparation.json')},'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};(root/'results/preparation.json').write_text(json.dumps(report,indent=2));print(json.dumps(counts),flush=True)

"""NCShare CPU-only export of the reserved gate videos; original caches read-only."""
from pathlib import Path
import hashlib,json,pickle,time
import numpy as np
ROOT=Path('/work/bbyrd1/contextual-gate-20260919');ROOT.mkdir(exist_ok=True)
split=json.loads(Path('/work/bbyrd1/class-gate-study-20260914/results/preparation.json').read_text())['videos'];gate=set(split['gate']);assert len(gate)==90 and not gate&set(split['expert']) and not gate&set(split['development'])
report={'passed':False,'videos':split,'shards':[]};seen=set()
for shard in range(8):
 dest=ROOT/f'gate-{shard}.npz';meta=ROOT/f'gate-{shard}.json'
 if dest.exists() and meta.exists():
  info=json.loads(meta.read_text());report['shards'].append(info);seen.update(r['video'] for r in info['rows']);continue
 source=Path('/work/bbyrd1/adapted-road-20260911/cache')/f'crop_feats_train.shard{shard}of8.pkl';d=pickle.load(source.open('rb'));assert not d['meta']['partial'];assert d['meta']['revision']=='1f9fca1389fd883defc652634d95a21121c85a8c' and d['meta']['pad']==2
 h=hashlib.sha256();xs=[];ys=[];bs=[];rows=[];pos=0
 for key in sorted(d['boxes']):
  b=d['boxes'][key].astype(np.float32);y=d['targets'][key];h.update(key.encode());h.update(b.tobytes());h.update(y.tobytes());v,f=key.rsplit('_',1)
  if v not in gate:continue
  x=np.asarray(d['feats'][key],np.float16);assert x.shape==(len(b),1024) and y.shape==(len(b),184) and np.isfinite(x).all();xs.append(x);ys.append(y.astype(np.uint8));bs.append(b);rows.append({'video':v,'fid':int(f),'start':pos,'end':pos+len(b)});pos+=len(b);seen.add(v)
 assert h.hexdigest()==d['meta']['row_sha256']
 tmp=dest.with_suffix('.tmp')
 with tmp.open('wb') as out:np.savez(out,crop=np.concatenate(xs),targets=np.concatenate(ys),boxes=np.concatenate(bs))
 tmp.replace(dest);info={'file':dest.name,'source':str(source),'source_row_sha256':h.hexdigest(),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'rows':rows,'n':pos};tmp=meta.with_suffix('.tmp');tmp.write_text(json.dumps(info));tmp.replace(meta);report['shards'].append(info);print('EXPORTED',shard,pos,flush=True);del d,xs,ys,bs
assert seen==gate
report.update(passed=True,time=time.time(),rows=sum(s['n'] for s in report['shards']));tmp=ROOT/'export.tmp';tmp.write_text(json.dumps(report));tmp.replace(ROOT/'export.json');print('COMPLETE',report['rows'],flush=True)

import hashlib,json,os,pickle
from pathlib import Path
import numpy as np
root=Path(os.environ['COMPARISON_ROOT']);ckhash=hashlib.sha256(Path(os.environ['ADAPTED_CHECKPOINT']).read_bytes()).hexdigest();manifest=json.loads((root/'code/shared-frames.json').read_text())
report={}
for split,n in [('train',8),('val',4)]:
 outputs={c:{'feats':{},'targets':{},'n_boxes':{}} for c in ['original','adapted']};seen=set();hashes={}
 for i in range(n):
  p=root/'cache'/f'crop_feats_{split}.shard{i}of{n}.pkl';d=pickle.load(p.open('rb'))
  assert not d['meta']['partial'] and d['checkpoint_sha256']==ckhash
  keys=set(d['feats']);assert keys==set(d['adapted_feats'])==set(d['boxes']) and not seen&keys;seen|=keys
  rowhash=hashlib.sha256()
  for k in sorted(keys):
   b=d['boxes'][k];rowhash.update(k.encode());rowhash.update(b.astype(np.float32).tobytes())
   if d['targets'] is not None:rowhash.update(d['targets'][k].tobytes())
   for f in [d['feats'][k],d['adapted_feats'][k]]:assert f.shape==(len(b),1024) and np.isfinite(f).all()
  assert rowhash.hexdigest()==d['meta']['row_sha256']
  hashes[p.name]=d['meta']['row_sha256']
  for c,field in [('original','feats'),('adapted','adapted_feats')]:
   outputs[c]['feats'].update(d[field]);outputs[c]['n_boxes'].update(d['n_boxes']);outputs[c]['targets'].update(d['targets'] or {})
  del d
 if split=='val':assert set(manifest['frames'])<=seen
 for c,d in outputs.items():
  d['targets']=d['targets'] or None;d['meta']={'condition':c,'row_hashes':hashes,'adapted_checkpoint_sha256':ckhash,'split':split}
  p=root/'data'/c/f'crop_feats_{split}.pkl'
  with p.with_suffix('.partial').open('wb') as f:pickle.dump(d,f,protocol=pickle.HIGHEST_PROTOCOL)
  p.with_suffix('.partial').replace(p)
 report[split]={'frames':len(seen),'row_hashes':hashes}
 print(split,len(seen),flush=True)
 del outputs
(root/'results/merge.json').write_text(json.dumps(report,indent=2)+'\n')

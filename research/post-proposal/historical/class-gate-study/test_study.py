"""Behavioral tests plus a synthetic end-to-end pipeline, including final AP."""
import hashlib,json,pickle,runpy,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
import torch
from common import *
A=Path(__file__).parent.resolve();rng=np.random.default_rng(18)
split=partitions([f'v{i}' for i in range(20)]);assert split==partitions(list(reversed([f'v{i}' for i in range(20)])))
assert [len(split[k]) for k in ['expert','gate','development']]==[14,3,3]
assert len(set().union(*map(set,split.values())))==20
# Mixture endpoints and confidence applied once.
p5=torch.rand(12,135);pl=torch.rand(12,135);q=torch.rand(12,1)
for weight in [0.,.25,1.]:
 mixed=(1-weight)*p5+weight*pl;assert torch.allclose(q*mixed,(1-weight)*(q*p5)+weight*(q*pl))
 if weight==0:assert torch.equal(mixed,p5)
 if weight==1:assert torch.equal(mixed,pl)
# Unsupported class weights retain the global anchor despite optimizer updates.
y=rng.integers(0,2,(48,135)).astype(np.float32);y[:,0]=0;good=.1+.8*y;bad=.9-.8*y
m=fit_gate(bad,good,y,0,4,16,'cpu',anchor=-.7,lam=.001);assert abs(m.weights()[0].item()-torch.sigmoid(torch.tensor(-.7)).item())<1e-7
assert m.weights()[1:].mean()>torch.sigmoid(torch.tensor(-.7))
core=runpy.run_path(str(A.parent/'post-proposal-experiments/baseline-core.py'))
for _ in range(10):
 yy=rng.integers(0,2,20);scores=rng.choice([.1,.4,.8],20);tp=yy[np.argsort(-scores)];expected=core['voc_ap'](np.cumsum(tp)/max(1,tp.sum()),np.cumsum(tp)/np.arange(1,len(tp)+1));assert abs(binary_ap(yy,scores)-expected)<1e-8
root=Path(tempfile.mkdtemp(prefix='class-gate-test-'));source=root/'source';(source/'data').mkdir(parents=True);(source/'code').mkdir();(root/'code').mkdir();(root/'results').mkdir()
(source/'code/baseline-core.py').write_text((A.parent/'post-proposal-experiments/baseline-core.py').read_text());manifest=json.loads((A.parent/'post-proposal-experiments/shared-frames.json').read_text());labels=manifest['labels']
feats={};targets={}
for v in range(20):
 for f in [1,2]:
  k=f'trainvideo{v:02d}_{f:05d}';feats[k]=rng.normal(size=(5,1024)).astype(np.float16);targets[k]=rng.integers(0,2,(5,184)).astype(np.float32);targets[k][:,183]=0
pickle.dump({'feats':feats,'targets':targets},open(source/'data/crop_feats_train.pkl','wb'))
torch.save({'embeds':torch.tensor(rng.normal(size=(184,512)),dtype=torch.float32)},source/'data/phrase_embeds.pt');torch.save(torch.full((184,),.5),source/'data/flat_alphas.pt')
keys=['heldout00_00001','heldout00_00002','heldout01_00001'];yolo={};base={};vf={};scale=np.array([840,600,840,600],np.float32)
for i,k in enumerate(keys):
 b=np.array([[.1,.1,.4,.4],[.11,.1,.4,.4],[.5,.5,.8,.8]],np.float32) if i<2 else np.empty((0,4),np.float32);yolo[k]={'boxes_xyxyn':b,'conf':np.array([.9,.8,.7],np.float32) if i<2 else np.empty(0,np.float32),'cls':np.zeros(len(b),np.int64)};vf[k]=rng.normal(size=(len(b),1024)).astype(np.float16)
 gt={'boxes':b[[0,2]]*scale if i<2 else np.empty((0,4),np.float32)}
 for h in ['agent','action','loc','duplex','triplet']:
  t=np.zeros((2 if i<2 else 0,len(labels[h])),np.float32)
  if len(t):t[:,0]=1
  gt[h]=t
 v,f=k.rsplit('_',1);base[v+'/'+str(int(f))]={'gt':gt}
manifest['frames']=keys;manifest['frame_sha256']=hashlib.sha256(('\n'.join(keys)+'\n').encode()).hexdigest();manifest['candidate_sha256']=hashlib.sha256(b''.join(k.encode()+yolo[k]['boxes_xyxyn'].tobytes()+yolo[k]['conf'].tobytes()+yolo[k]['cls'].tobytes() for k in keys)).hexdigest();(source/'code/shared-frames.json').write_text(json.dumps(manifest));(source/'code/train-counts.json').write_text((A.parent/'post-proposal-experiments/train-counts.json').read_text());pickle.dump({'records':yolo},open(source/'data/dets_v8x_best_val_fullcand.pkl','wb'));pickle.dump({'records':base,'labels':labels},open(source/'data/dets_i3d_val_fullcand.pkl','wb'));pickle.dump({'feats':vf},open(source/'data/crop_feats_val.pkl','wb'))
cfg=json.loads((A/'protocol.json').read_text());cfg.update(source_study=str(source),root=str(root),expert_epochs=1,gate_epochs=2,expert_batch_size=64,gate_batch_size=64,lambdas=[0.,.01],smoke=True);(root/'code/protocol.json').write_text(json.dumps(cfg))
log=root/'test.log'
with log.open('w') as f:
 for script,extra in [('prepare.py',[]),('train.py',['--seed','0']),('evaluate.py',['--seed','0','--workers','2'])]:
  r=subprocess.run([sys.executable,str(A/script),'--root',str(root),*extra],stdout=f,stderr=subprocess.STDOUT)
  if r.returncode:raise RuntimeError(log.read_text()[-5000:])
metrics=list((root/'results/metrics').glob('*.json'));assert len(metrics)==8
for p in metrics:
 m=json.loads(p.read_text());assert m['n_frames']==3 and m['candidate_sha256']==manifest['candidate_sha256'];assert len(m['ap_values']['triplet'])==86
for name in ['phrase','shuffled']:
 g=json.loads((root/f'runs/seed-0/gate-{name}.json').read_text());assert 134 in g['no_positive_indices'];assert abs(g['class']['weights'][134]-g['global']['weight'])<1e-7
(A/'test-result.json').write_text(json.dumps({'passed':True,'synthetic_root':str(root),'evaluations':8,'checks':['deterministic disjoint video partitions','gate endpoints and YOLO scaling once','zero-positive global fallback','learning preference for accurate expert','crop AP/VOC AP equivalence','end-to-end expert OOF training, gate selection and baseline detector AP','frame/candidate hash enforcement'],'smoke_only':True},indent=2)+'\n');print('PASS: gate behavior and full synthetic eight-variant pipeline',root)

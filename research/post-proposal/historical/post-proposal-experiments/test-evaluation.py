"""Compare the new common adapter with original stage evaluators on synthetic records."""
import hashlib,json,pickle,runpy,subprocess,sys,tempfile,types
from pathlib import Path
import numpy as np
import torch
from torch import nn
root=Path(__file__).parent.resolve();temp=Path(tempfile.mkdtemp(prefix='proposal-eval-test-'));data=temp/'data';data.mkdir()
manifest=json.loads((root/'shared-frames.json').read_text());labels=manifest['labels'];rng=np.random.default_rng(4);yolo={};base={};feats={};keys=[]
for i in range(4):
 stem=f'v{i}_00001';key=f'v{i}/1';keys.append(stem);n=0 if i==3 else 3;boxes=np.array([[.1,.1,.5,.5],[.12,.12,.52,.52],[.6,.6,.9,.9]],np.float32)[:n]
 yolo[stem]={'boxes_xyxyn':boxes,'conf':np.array([.9,.8,.7],np.float32)[:n],'cls':np.array([0,0,1])[:n]}
 gt={'boxes':(boxes[:1]*np.array([840,600,840,600])).astype(np.float32)}
 for h in ['agent','action','loc','duplex','triplet']:
  gt[h]=np.zeros((min(n,1),len(labels[h])),np.float32)
  if n:gt[h][0,0]=1
 base[key]={'boxes':(boxes*np.array([840,600,840,600])).astype(np.float32),'logits':rng.normal(size=(n,184)).astype(np.float32),'scores':np.array([.9,.8,.7],np.float32)[:n],'gt':gt};feats[key]=rng.normal(size=(n,256)).astype(np.float16)
for filename,payload in [('dets_v8x_best_val_fullcand.pkl',{'records':yolo}),('dets_i3d_val_fullcand.pkl',{'records':base,'labels':labels}),('roi_feats_i3d_val.pkl',{'feats':feats})]:pickle.dump(payload,open(data/filename,'wb'))
h=nn.Linear(256,184);torch.save({'state':h.state_dict()},data/'head_roialign_lam0_junkneg.pt');m=nn.Sequential(nn.Linear(305,512),nn.ReLU(),nn.Linear(512,135));torch.save({'state':m.state_dict(),'in_dim':305,'variant':'sigfeat'},data/'comp_mlp_sigfeat.pt')
manifest['frames']=keys;manifest['frame_sha256']=hashlib.sha256(('\n'.join(keys)+'\n').encode()).hexdigest();manifest['candidate_sha256']=hashlib.sha256(b''.join(s.encode()+np.asarray(yolo[s]['boxes_xyxyn']).tobytes()+np.asarray(yolo[s]['conf']).tobytes()+np.asarray(yolo[s]['cls']).tobytes() for s in keys)).hexdigest();(temp/'manifest.json').write_text(json.dumps(manifest))
subprocess.run([sys.executable,str(root/'evaluate-study.py'),'--data',str(data),'--runs',str(temp),'--out',str(temp/'results'),'--manifest',str(temp/'manifest.json'),'--only','stage0','stage1','stage2','stage3','--workers','1','--smoke'],check=True,stdout=subprocess.DEVNULL)
core=runpy.run_path(str(root/'baseline-core.py'));mod=types.ModuleType('modules.evaluation');mod.__dict__.update(core);parent=types.ModuleType('modules');parent.evaluation=mod;utils=types.ModuleType('modules.utils');utils.get_individual_labels=core['get_individual_labels'];sys.modules.update({'modules':parent,'modules.evaluation':mod,'modules.utils':utils})
E=Path('/data/repos/ROAD_Reason/experiments/exp11_yolo')
for n,filename in [(0,'eval_i3d_fullcand.py'),(1,'eval_hybrid_score_transfer.py'),(2,'eval_head.py'),(3,'eval_head.py')]:
 s=(E/filename).read_text().replace(str(E),str(data));ref=temp/f'ref{n}.json'
 if n==0:s=s.replace('results_i3d_fullcand_baseline.json',str(ref));sys.argv=[filename]
 elif n==1:sys.argv=[filename,'--out',str(ref)]
 else:
  sys.argv=[filename,'--head',str(data/'head_roialign_lam0_junkneg.pt'),'--out',str(ref),'--gate']
  if n==3:sys.argv+=['--comp-mlp',str(data/'comp_mlp_sigfeat.pt')]
 exec(compile(s,filename,'exec'),{'__name__':'__main__'})
 actual=json.loads((temp/'results'/f'stage{n}.json').read_text());expected=json.loads(ref.read_text());assert actual['summary']==expected['summary'],(n,actual['summary'],expected['summary']);assert actual['per_class']==expected['per_class'],n
print('PASS: stages 0–3 exactly match original source evaluators on synthetic overlapping, duplicate, and empty-frame records.')

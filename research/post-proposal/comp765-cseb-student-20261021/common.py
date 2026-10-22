"""Isolated COMP 765 causal distillation study, started 2026-10-21."""
import hashlib, importlib.util, json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
OLD=PARENT/'cseb-student'
sys.path.insert(0,str(OLD))
import pilot as p
from model import Student, SharedBackbone, objective
ROOT=Path('/home/brandon/.local/share/road-reason/comp765-cseb-student-20261021')
EXPANDED=Path('/home/brandon/.local/share/road-reason/cseb-student-expanded-20261019')
WIKI=Path('/data/repos/wiki')
INPUTS={'frames':WIKI/'artifacts/post-proposal-experiments/shared-frames.json',
 'counts':WIKI/'artifacts/proposal-tail/train-counts.json',
 'yolo':PARENT.parents[1]/'experiments/exp11_yolo/dets_v8x_best_val_fullcand.pkl',
 'gt':PARENT.parents[1]/'experiments/exp11_yolo/dets_i3d_val_fullcand.pkl',
 'teacher_result':WIKI/'artifacts/contextual-roi-1b-20260930/results-verified-20261005/attention-global-dcb-seed0-detector-results.json'}
HEADS=['agentness','agent','action','loc','duplex','triplet']
sha=p.sha
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def progress(phase,**kw):
 d=dict(time=time.time(),phase=phase,**kw);save(ROOT/'progress.json',d);print(json.dumps(d),flush=True)
def setup(seed=0):
 torch.set_num_threads(8);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.benchmark=False
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def tensor_hash(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def groups(labels):
 z={r['label']:r['z'] for r in read(INPUTS['counts'])['rows']};assert set(z)==set(labels)
 d={k:[i for i,l in enumerate(labels) if fn(z[l])] for k,fn in [('tail47',lambda x:x<0),('deep28',lambda x:x<-.5),('common39',lambda x:x>=0)]}
 assert [len(d[k]) for k in ['tail47','deep28','common39']]==[47,28,39]
 return d
def signature():return {'protocol_sha256':sha(HERE/'protocol.json'),'freeze_sha256':sha(ROOT/'freeze.json')}
def verify():
 for path,h in read(ROOT/'freeze.json')['files'].items():assert sha(path)==h,path
def image_path(v,f):return p.ROOT/'rgb-images'/v/f'{f:05d}.jpg'
def to_gpu(row):return {k:v.cuda().float() if torch.is_tensor(v) and k!='agents' else v.cuda() if k=='agents' else v for k,v in row.items()}

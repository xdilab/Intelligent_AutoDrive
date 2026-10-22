"""Shared paths and immutable-input helpers for the new COMP 765 study."""
import hashlib, importlib.util, json, os, runpy, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
REPO = Path('/data/repos/ROAD_Reason')
PARENT = REPO / 'research/post-proposal'
OLD = PARENT / 'historical/post-proposal-experiments'
E12 = REPO / 'experiments/exp12_phrase_head'
E11 = REPO / 'experiments/exp11_yolo'
WIKI = Path('/data/repos/wiki')
DEFAULT_ROOT = Path('/home/brandon/.local/share/road-reason/comp765-multiple-descriptions-20261021')
SNAPSHOT = Path('/home/brandon/.cache/huggingface/hub/models--OpenGVLab--InternVideo2_CLIP_S/snapshots/1f9fca1389fd883defc652634d95a21121c85a8c')
HEADS = ['agentness','agent','action','loc','duplex','triplet']
OFF = [0,1,11,33,49,98,184]
shared = runpy.run_path(str(OLD / 'train-repeats.py'))
PhraseHead, focal = shared['PhraseHead'], shared['focal']
INPUTS = {
    'train': E12/'crop_full/crop_feats_train.pkl',
    'val': E12/'crop_full/crop_feats_val.pkl',
    'yolo': E11/'dets_v8x_best_val_fullcand.pkl',
    'gt': E11/'dets_i3d_val_fullcand.pkl',
    'phrases': E12/'phrases.json', 'canonical_embeddings': E12/'phrase_embeds.pt',
    'alphas': REPO/'experiments/exp6_detection_steered/cache/flat_alphas.pt',
    'frames': WIKI/'artifacts/post-proposal-experiments/shared-frames.json',
    'counts': WIKI/'artifacts/proposal-tail/train-counts.json',
    'annotation': Path('/data/datasets/ROAD_plusplus/road_waymo_trainval_v1.1.json'),
    'text_weights': SNAPSHOT/'model.safetensors',
}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def digest_tensor(t): return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(p)

def checkpoint(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.partial');torch.save(obj,tmp);tmp.replace(p)

def progress(root,phase,**kw):
    d=dict(time=time.time(),phase=phase,**kw);save(root/'progress.json',d);print(json.dumps(d),flush=True)

def signature(root):
    return {'protocol':sha(HERE/'protocol.json'),'freeze':sha(root/'freeze.json')}

def verify_freeze(root):
    freeze=read(root/'freeze.json')
    for p,h in freeze['files'].items(): assert sha(p)==h, f'Frozen file changed: {p}'
    for k,v in read(root/'provenance.json')['inputs'].items():
        s=Path(v['path']).stat()
        assert s.st_size==v['bytes'] and s.st_mtime_ns==v['mtime_ns'],f'Input changed: {k}'

def groups(labels):
    data=read(INPUTS['counts']);z={r['label']:r['z'] for r in data['rows']}
    assert set(z)==set(labels)
    result={n:[i for i,l in enumerate(labels) if fn(z[l])] for n,fn in [('tail47',lambda x:x<0),('deep28',lambda x:x<-.5),('common39',lambda x:x>=0)]}
    assert [len(result[k]) for k in ['tail47','deep28','common39']]==[47,28,39]
    assert set(result['deep28'])<=set(result['tail47'])
    return result

def setup():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available(), 'GPU required; do not silently switch protocol to CPU'


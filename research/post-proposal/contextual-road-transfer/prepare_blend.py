"""Same45-video/global-grid selector as existing focal blend, using new DCB experts."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from model import ContextualRoIHead
from train_cached import Cache,atomic_json,atomic_torch,file_sha,ap_values

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--seed',type=int,required=True);a=ap.parse_args();root=a.root;GATE=root;cfg=json.loads((root/'protocol.json').read_text());torch.set_num_threads(4)
 for name,digest in cfg['gate_inputs_sha256'].items():assert file_sha(root/'inputs/gate'/name)==digest
 parents=[root/'runs'/f'attention-{c}-dcb-seed{a.seed}' for c in ['classification','contrastive-all184']];dest=root/'runs'/f'attention-global-dcb-seed{a.seed}';dest.mkdir(exist_ok=True)
 partition=np.load(GATE/'data/gate-partition.npy');ix=np.flatnonzero(partition==1);data=Cache(GATE/'data','gate');assert len(partition)==len(data)
 videos=set()
 for line in (GATE/'data/gate.jsonl').open():
  r=json.loads(line)
  if partition[r['row_start']]==1:videos.add(r['video']);assert np.all(partition[r['row_start']:r['row_end']]==1)
 assert len(videos)==45
 signature={'protocol_sha256':file_sha(root/'protocol.json'),'cache_marker_sha256':file_sha(root/'cache-ready.json'),'parents':[file_sha(p/'best.pt') for p in parents],'partition_sha256':file_sha(GATE/'data/gate-partition.npy'),'gate_preparation_sha256':file_sha(root/'inputs/gate/preparation.json'),'selection_videos':sorted(videos),'code_sha256':{n:file_sha(Path(__file__).parent/n) for n in ['model.py','prepare_blend.py','dcb.py']}}
 if (dest/'complete.json').exists():assert json.loads((dest/'complete.json').read_text())['signature']==signature;return
 bank=torch.load(root/'data/phrase_embeds.pt',map_location='cpu',weights_only=False)['embeds'];preds=[];states=[];epochs=[]
 for parent in parents:
  completed=json.loads((parent/'complete.json').read_text());assert completed['passed'];ck=torch.load(parent/'best.pt',map_location='cpu',weights_only=False);assert ck['signature']==completed['signature'];assert ck['signature']['protocol_sha256']==signature['protocol_sha256'];states.append(ck['model']);epochs.append(ck['epoch']);m=ContextualRoIHead(bank,fusion='attention').cuda().eval();m.load_state_dict(ck['model']);p=np.empty((len(ix),184),np.float32)
  with torch.no_grad():
   for start in range(0,len(ix),cfg['batch_size']):
    ids=ix[start:start+cfg['batch_size']];xs,_=data.batch(ids,'cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):out=m(*xs)
    p[start:start+len(ids)]=out['logits'].float().sigmoid().cpu().numpy()
    if start%(cfg['batch_size']*100)==0:atomic_json({'phase':'blend-selection-inference','position':start,'total':len(ix),'time':time.time()},dest/'progress.json')
  preds.append(p);del m;torch.cuda.empty_cache()
 y=np.asarray(data.arrays['targets'][ix]);sweep=[]
 for weight in cfg['global_grid']:
  probabilities=preds[0].copy();probabilities[:,49:]=(1-weight)*preds[0][:,49:]+weight*preds[1][:,49:]
  values=ap_values(y,probabilities);sweep.append({'weight':weight,'triplet':float(np.mean(values[98:]))});atomic_json({'phase':'blend-selection','weight':weight,'time':time.time()},dest/'progress.json')
 best=max(sweep,key=lambda v:(v['triplet'],-v['weight']))
 atomic_torch({'kind':'blend','models':states,'weight':best['weight'],'epoch':epochs,'signature':signature},dest/'best.pt')
 atomic_json({'passed':True,'signature':signature,'weight':best['weight'],'sweep':sweep,'time':time.time()},dest/'complete.json');print('BLEND',a.seed,best,flush=True)
if __name__=='__main__':main()

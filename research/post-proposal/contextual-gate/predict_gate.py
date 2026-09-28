"""Frozen experts on the previously reserved gate videos, resumable by batch."""
import argparse,json,time
import numpy as np
import torch
from common import ROOT,BASE,Cache,ContextualRoIHead,parents,signature,atomic_json
from router import actor_features

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);a=ap.parse_args();torch.set_num_threads(4);dest=ROOT/'runs'/f'seed{a.seed}';dest.mkdir(parents=True,exist_ok=True);sig=signature(a.seed)
 done=dest/'predictions-ready.json'
 if done.exists():assert json.loads(done.read_text())['signature']==sig;return
 data=Cache(ROOT/'data','gate');n=len(data);bank=torch.load(BASE/'data/phrase_embeds.pt',weights_only=False)['embeds'];models=[]
 for path in parents(a.seed):
  ck=torch.load(path/'best.pt',map_location='cpu',weights_only=False);complete=json.loads((path/'complete.json').read_text());assert ck['signature']==complete['signature'];m=ContextualRoIHead(bank,fusion='attention').cuda().eval();m.load_state_dict(ck['model']);m.requires_grad_(False);models.append(m)
 marker=dest/'prediction-progress.json';state=json.loads(marker.read_text()) if marker.exists() else {'signature':sig,'position':0};assert state['signature']==sig;assert 0<=state['position']<=n;arrays={}
 for name,shape in [('a',(n,184)),('b',(n,184)),('actor',(n,16))]:
  p=dest/f'{name}.npy'
  if state['position']:assert p.exists()
  arrays[name]=np.lib.format.open_memmap(p,mode='r+' if p.exists() else 'w+',dtype='float32',shape=shape);assert arrays[name].shape==shape and arrays[name].dtype==np.float32
 with torch.no_grad():
  for count,j in enumerate(range(state['position'],n,256),1):
   end=min(j+256,n);x,_=data.batch(np.arange(j,end),'cuda')
   with torch.autocast('cuda',dtype=torch.bfloat16):
    for name,m in zip(['a','b'],models):arrays[name][j:end]=m(*x)['logits'].float().sigmoid().cpu().numpy()
   arrays['actor'][j:end]=actor_features(x[0],x[1],x[3]).cpu().numpy()
   if count%100==0 or end==n:
    for array in arrays.values():array.flush()
    atomic_json({'signature':sig,'position':end},marker);atomic_json({'phase':'expert-predictions','position':end,'total':n,'time':time.time()},dest/'progress.json');print('PREDICT',a.seed,end,n,flush=True)
 atomic_json({'passed':True,'signature':sig,'rows':n},done)
if __name__=='__main__':main()

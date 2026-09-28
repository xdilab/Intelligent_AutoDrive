"""Class-balanced pairwise ranking on45 fit videos; selection only on other45."""
import argparse,json,time,signal,sys
import numpy as np
import torch
from common import ROOT,BASE,signature,atomic_json,atomic_torch
from router import Router,composition_map,ranking_loss
STOP=False
def stop(*_):
 global STOP
 STOP=True

def aps(y,p):
 out=[]
 for c in range(135):
  truth=y[:,c][np.argsort(-p[:,c],kind='stable')]>0;n=truth.sum()
  if not n:out.append(0.);continue
  prec=truth.cumsum()/np.arange(1,len(truth)+1);out.append(float(np.maximum.accumulate(prec[::-1])[::-1][truth].sum()/n*100))
 return out

@torch.no_grad()
def predict(m,a,b,actor):
 p=np.empty((len(a),135),np.float32);m.eval()
 for j in range(0,len(a),128):
  end=min(j+128,len(a));n=end-j;classes=torch.arange(135,device='cuda').repeat(n)
  tensors=[torch.from_numpy(np.array(x[j:end],copy=True)).cuda().repeat_interleave(135,0) for x in [a,b,actor]]
  p[j:end]=m(*tensors,classes)[0].reshape(n,135).cpu().numpy()
 return p

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);a=ap.parse_args();torch.set_num_threads(4);dest=ROOT/'runs'/f'seed{a.seed}';sig=signature(a.seed);cfg=json.loads((ROOT/'protocol.json').read_text());ready=json.loads((dest/'predictions-ready.json').read_text());assert ready['signature']==sig
 for sign in [signal.SIGTERM,signal.SIGINT,signal.SIGUSR1]:signal.signal(sign,stop)
 arrays=[np.load(dest/f'{k}.npy',mmap_mode='r') for k in ['a','b','actor']];y=np.load(ROOT/'data/gate-targets.npy',mmap_mode='r')[:,49:];part=np.load(ROOT/'data/gate-partition.npy',mmap_mode='r');fit=np.flatnonzero(part==0);sel=np.flatnonzero(part==1);ys=np.array(y[sel]);sarrays=[np.array(x[sel]) for x in arrays]
 labels=json.loads((BASE/'data/shared-frames.json').read_text())['labels'];mapping=composition_map(labels)
 support=np.zeros(135,np.int64);byvideo={}
 for line in (ROOT/'data/gate.jsonl').open():
  row=json.loads(line)
  if part[row['row_start']]==0:byvideo.setdefault(row['video'],np.zeros(135,bool));byvideo[row['video']]|=np.asarray(y[row['row_start']:row['row_end']]).any(0)
 for hits in byvideo.values():support+=hits
 assert len(byvideo)==45
 globalpath=dest/'global.json'
 if globalpath.exists():globalresult=json.loads(globalpath.read_text());assert globalresult['signature']==sig
 else:
  sweep=[]
  for weight in cfg['global_grid']:
   vals=aps(ys,(1-weight)*sarrays[0][:,49:]+weight*sarrays[1][:,49:]);sweep.append({'weight':weight,'triplet':float(np.mean(vals[49:])),'per_class':vals});atomic_json({'phase':'global-selection','weight':weight,'time':time.time()},dest/'progress.json')
  best=max(sweep,key=lambda v:(v['triplet'],-v['weight']));globalresult={'signature':sig,'weight':best['weight'],'sweep':sweep,'support_videos':support.tolist()};atomic_json(globalresult,globalpath)
 anchor=globalresult['weight'];pos=[fit[np.asarray(y[fit,c])>0] for c in range(135)];neg=[fit[np.asarray(y[fit,c])==0] for c in range(135)];eligible=np.array([c for c in range(135) if len(pos[c]) and len(neg[c])]);assert len(eligible)>0
 for kind in cfg['learned_variants']:
  out=dest/kind;out.mkdir(exist_ok=True);done=out/'complete.json'
  if done.exists():assert json.loads(done.read_text())['signature']==sig;continue
  torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed);m=Router(mapping,torch.tensor(support),anchor,kind).cuda();opt=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);ep=step=0;best=-float('inf');history=[];resume=out/'resume.pt'
  if resume.exists():
   ck=torch.load(resume,map_location='cuda',weights_only=False);assert ck['signature']==sig;m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);ep=ck['epoch'];step=ck['step'];best=ck['best'];history=ck['history'];rng.bit_generator.state=ck['rng']
  def save(epoch,step):atomic_torch({'signature':sig,'model':m.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'step':step,'best':best,'history':history,'rng':rng.bit_generator.state},resume)
  for epoch in range(ep,cfg['epochs']):
   m.train()
   for iteration in range(step,cfg['steps_per_epoch']):
    chosen=rng.choice(eligible,cfg['classes_per_step'],replace=len(eligible)<cfg['classes_per_step']);count=cfg['pairs_per_class'];ix=np.concatenate([np.r_[rng.choice(pos[c],count),rng.choice(neg[c],count)] for c in chosen]);classes=torch.tensor(np.repeat(chosen,2*count),device='cuda');inputs=[torch.from_numpy(np.array(x[ix])).cuda() for x in arrays]
    prob,g=m(*inputs,classes);loss=ranking_loss(prob,len(chosen),count)+cfg['shrink_penalty']*(g-float(anchor)).square().mean();opt.zero_grad(set_to_none=True);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step()
    if (iteration+1)%100==0 or STOP:
     save(epoch,iteration+1);atomic_json({'phase':'gate-fit','kind':kind,'epoch':epoch+1,'step':iteration+1,'loss':float(loss.detach()),'time':time.time()},dest/'progress.json');print('FIT',a.seed,kind,epoch+1,iteration+1,float(loss.detach()),flush=True)
    if STOP:sys.exit(75)
   save(epoch,cfg['steps_per_epoch']);vals=aps(ys,predict(m,*sarrays));value=float(np.mean(vals[49:]));history.append({'epoch':epoch+1,'triplet':value,'per_class':vals})
   if value>best:
    best=value;atomic_torch({'signature':sig,'kind':kind,'model':m.state_dict(),'anchor':anchor,'support':support.tolist(),'selected_epoch':epoch+1,'selection_triplet':value},out/'best.pt')
   step=0;save(epoch+1,0);atomic_json({'phase':'gate-selection','kind':kind,'epoch':epoch+1,'triplet':value,'time':time.time()},dest/'progress.json');print('SELECT',a.seed,kind,epoch+1,value,flush=True)
  atomic_json({'passed':True,'signature':sig,'history':history,'parameters':sum(p.numel() for p in m.parameters()),'best_selection_triplet':best},done)
 atomic_json({'passed':True,'signature':sig,'time':time.time()},dest/'training-complete.json')
if __name__=='__main__':main()

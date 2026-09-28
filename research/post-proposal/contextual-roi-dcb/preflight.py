"""Disposable real-cache replay and benchmark; no study checkpoint used."""
from pathlib import Path
import importlib.util,json,time,shutil
import numpy as np
import torch
from model import ContextualRoIHead
from dcb import DCBState,objective
from train_cached import Cache,file_sha,atomic_json
ROOT=Path('/data/repos/wiki/artifacts/contextual-roi-dcb-20260921');CODE=Path(__file__).parent

def main():
 torch.set_num_threads(4);cfg=json.loads((ROOT/'protocol.json').read_text());assert file_sha(ROOT/'cache-ready.json')==cfg['cache_marker_sha256'];assert shutil.disk_usage(ROOT/'runs').free>5*1024**3
 available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024;assert available>64*1024**3
 oldpath=CODE.parent/'contextual-roi/model.py';spec=importlib.util.spec_from_file_location('parent_model',oldpath);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);bank=torch.load(ROOT/'data/phrase_embeds.pt',weights_only=False)['embeds'].float();alpha=torch.load(ROOT/'data/flat_alphas.pt',weights_only=True).float().cuda();assert bank.shape==(184,512)
 official=json.loads((ROOT/'data/shared-frames.json').read_text());checks=[]
 gate=ROOT.parent/'contextual-gate-20260919'
 for name,digest in cfg['gate_inputs_sha256'].items():assert file_sha(gate/name)==digest
 assert cfg['global_grid']==json.loads((gate/'protocol.json').read_text())['global_grid']
 for seed in cfg['seeds']:
  for family,run in [('contextual-roi-20260917',f'attention-classification-seed{seed}'),('contextual-roi-all184-20260918',f'attention-contrastive-all184-seed{seed}')]:
   d=ROOT.parent/family/'runs'/run;ck=json.loads((d/'complete.json').read_text());res=json.loads((d/'detector-results.json').read_text());assert ck['passed'] and ck['signature']['cache_marker_sha256']==cfg['cache_marker_sha256'];assert res['n_frames']==36717 and res['candidate_sha256']==official['candidate_sha256'] and res['frame_sha256']==official['frame_sha256']
  torch.manual_seed(seed);ref=old.ContextualRoIHead(bank,fusion='attention');rng=torch.get_rng_state();torch.manual_seed(seed);st=DCBState();m=ContextualRoIHead(bank,fusion='attention');assert torch.equal(rng,torch.get_rng_state());assert all(torch.equal(v,m.state_dict()[k]) for k,v in ref.state_dict().items());checks.append({'seed':seed,'initialization_and_rng_match':True})
 tr=Cache(ROOT/'data','train');order=np.random.default_rng(0).permutation(len(tr));batch=cfg['batch_size']
 # Same200 original/fork focal updates, true parent implementation.
 torch.manual_seed(0);m=old.ContextualRoIHead(bank,fusion='attention').cuda();torch.manual_seed(0);new=ContextualRoIHead(bank,fusion='attention').cuda();opt=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);opt2=torch.optim.AdamW(new.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);st=DCBState().cuda()
 for step in range(200):
  xs,y=tr.batch(order[step*batch:(step+1)*batch],'cuda')
  for model,optimizer,is_new in [(m,opt,False),(new,opt2,True)]:
   optimizer.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):o=model(*xs);loss=(objective(o,y,alpha,0,st,'focal') if is_new else old.objective(o,y,alpha,0))[0]
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
 assert all(torch.equal(v,new.state_dict()[k]) for k,v in m.state_dict().items());del m,new,opt,opt2;torch.cuda.empty_cache()
 timings={}
 for weight in [0,.001]:
  torch.manual_seed(0);m=ContextualRoIHead(bank,fusion='attention').cuda();state=DCBState().cuda();opt=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);norms=[];losses=[]
  for step in range(210):
   if step==10:torch.cuda.synchronize();start=time.monotonic()
   xs,y=tr.batch(order[step*batch:(step+1)*batch],'cuda');opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):out=m(*xs);loss,cls,aux=objective(out,y,alpha,weight,state)
   assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step();norms.append(float(norm));losses.append(float(cls.detach()))
  torch.cuda.synchronize();timings[str(weight)]={'rows_per_second':200*batch/(time.monotonic()-start),'gradient_norm_mean':float(np.mean(norms)),'gradient_norm_max':max(norms),'clipped_fraction':float(np.mean(np.array(norms)>1)),'classification_first':losses[0],'classification_last':losses[-1]};assert state.N.sum()>0 and torch.isfinite(state.S).all();del m,opt,state;torch.cuda.empty_cache()
 atomic_json({'passed':True,'time':time.time(),'protocol_sha256':file_sha(ROOT/'protocol.json'),'checks':checks,'focal_replay_steps':200,'focal_replay_bit_identical':True,'benchmark':timings,'available_RAM_GiB':available/1024**3},ROOT/'preflight.json');print(json.dumps(timings),flush=True)
if __name__=='__main__':main()

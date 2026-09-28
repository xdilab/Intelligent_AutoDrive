"""Actual-dimension baseline matching, prototype geometry and disposable GPU timing."""
from pathlib import Path
import json,hashlib,importlib.util,time,shutil
import numpy as np
import torch
from model import ContextualRoIHead,objective
from train_cached import Cache,file_sha,atomic_json
ROOT=Path('/data/repos/wiki/artifacts/contextual-roi-langctl-20260919');CODE=Path(__file__).parent;OLD=CODE.parent/'contextual-roi/model.py'
def geometry(bank):
 x=bank.detach().float();norm=x.norm(dim=-1);x=torch.nn.functional.normalize(x,dim=-1);v=(x@x.T)[~torch.eye(184,dtype=torch.bool)]
 return {'norm_mean':norm.mean().item(),'norm_min':norm.min().item(),'norm_max':norm.max().item(),'offdiagonal_cosine_mean':v.mean().item(),'offdiagonal_cosine_sd':v.std().item()}
def main():
 torch.set_num_threads(4);cfg=json.loads((ROOT/'protocol.json').read_text());assert file_sha(ROOT/'cache-ready.json')==cfg['cache_marker_sha256'];available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024;assert available>64*1024**3;assert shutil.disk_usage(ROOT).free>6*1024**3
 bank=torch.load(ROOT/'data/phrase_embeds.pt',map_location='cpu',weights_only=False)['embeds'].float();assert bank.shape==(184,512);spec=importlib.util.spec_from_file_location('historical',OLD);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);records=[];official=json.loads((ROOT/'data/shared-frames.json').read_text())
 for seed in cfg['seeds']:
  base=Path(cfg['baseline_runs'][seed]);done=json.loads((base/'complete.json').read_text());assert done['passed'] and done['signature']['code_sha256']['model.py']==file_sha(OLD);assert done['signature']['cache_marker_sha256']==cfg['cache_marker_sha256'];res=json.loads((base/'detector-results.json').read_text());assert res['n_frames']==36717;assert (res['frame_sha256'],res['candidate_sha256'])==(official['frame_sha256'],official['candidate_sha256'])
  torch.manual_seed(seed);reference=old.ContextualRoIHead(bank,fusion='attention');rng=torch.get_rng_state();h=hashlib.sha256()
  for name,value in sorted(reference.named_parameters()):h.update(name.encode());h.update(value.detach().numpy().tobytes())
  for mode in ['bypass','randbank']:
   torch.manual_seed(seed);m=ContextualRoIHead(bank,fusion='attention',mode=mode,seed=seed);assert torch.equal(rng,torch.get_rng_state())
   for name,p in reference.named_parameters():assert torch.equal(p,dict(m.named_parameters())[name]),(seed,mode,name)
   records.append({'seed':seed,'mode':mode,'parameter_sha256':h.hexdigest(),'baseline_parameters_bit_identical':True,'global_rng_identical':True,'bank_seed':m.bank_seed,'bank_sha256':hashlib.sha256(m.phrase_bank.numpy().tobytes()).hexdigest(),'geometry':geometry(m.phrase_bank),'nominal_parameters':sum(p.numel() for p in m.parameters()),'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad)})
   del m
  del reference
 tr=Cache(ROOT/'data','train');order=np.random.default_rng(0).permutation(len(tr));alpha=torch.load(ROOT/'data/flat_alphas.pt',map_location='cpu',weights_only=True).float().cuda();timings={}
 for mode in ['bypass','randbank']:
  torch.manual_seed(0);m=ContextualRoIHead(bank,fusion='attention',mode=mode,seed=0).cuda();opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);frozen={k:p.detach().clone() for k,p in m.named_parameters() if not p.requires_grad};batch=cfg['batch_size']
  for step in range(210):
   if step==10:torch.cuda.synchronize();start=time.monotonic()
   x,y=tr.batch(order[step*batch:(step+1)*batch],'cuda');opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):out=m(*x);loss,cls,aux=objective(out,y,alpha,0)
   assert torch.equal(loss,cls) and torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step()
  torch.cuda.synchronize();timings[mode]={'training_rows_per_second':200*batch/(time.monotonic()-start),'measured_steps':200}
  for k,v in frozen.items():assert torch.equal(v,dict(m.named_parameters())[k])
  assert all(torch.isfinite(p).all() for p in m.parameters());del m,opt;torch.cuda.empty_cache()
 result={'passed':True,'time':time.time(),'model_sha256':file_sha(CODE/'model.py'),'protocol_sha256':file_sha(ROOT/'protocol.json'),'checks':records,'real_bank_geometry':geometry(torch.nn.functional.normalize(bank,dim=-1)),'benchmark':timings,'available_RAM_GiB':available/1024**3,'RAM_budget_note':'Two4.23GiB score matrices plus worker temporaries; required64GiB available','benchmark_disposable':True}
 atomic_json(result,ROOT/'preflight.json');print(json.dumps(result,indent=2))
if __name__=='__main__':main()

from pathlib import Path
import json,time,hashlib
import numpy as np
import torch
from model import make_model,objective
from train_cached import Cache,atomic_json,file_sha
ROOT=Path('/data/repos/wiki/artifacts/stage7-stage5-20260918')
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 cfg=json.loads((ROOT/'protocol.json').read_text());dev=Cache(ROOT/'data','dev');tr=Cache(ROOT/'data','train');val=Cache(ROOT/'data','val')
 expert={json.loads(l)['video'] for l in (ROOT/'data/train.jsonl').open()};development={json.loads(l)['video'] for l in (ROOT/'data/dev.jsonl').open()};assert len(expert)==420 and len(development)==90 and not expert&development
 report={'seeds':{},'cache_rows':{'train':len(tr),'dev':len(dev),'val':len(val)},'started':time.time()}
 for seed in cfg['seeds']:
  checks=json.loads((ROOT/'inputs'/f'train-seed{seed}.json').read_text())['checkpoint_sha256']
  for name in ['head-flat.pt','stage5.pt']:
   p=ROOT/'inputs'/f'seed-{seed}'/name;assert file_sha(p)==checks[name];c=torch.load(p,map_location='cpu',weights_only=False);assert set(c['expert_videos'])==expert
  m=make_model(ROOT,seed).cuda().eval();parity=[]
  for split,data in [('train',tr),('dev',dev),('val',val)]:
   xs,_=data.batch(np.unique(np.linspace(0,len(data)-1,256,dtype=int)),'cuda')
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):actual=m(*xs)['logits']
   with torch.no_grad(),torch.autocast('cuda',enabled=False):
    flat=m.flat(xs[0].float());expected=torch.cat([flat[:,:49],m.comp(torch.cat([flat[:,:49].sigmoid(),xs[0].float()],1))],1)
   err=(actual-expected).abs().max().item();assert err==0;parity.append({'split':split,'max_abs_logit_error':err})
  report['seeds'][seed]={'initial_parity':parity};del m
 m=make_model(ROOT,0).cuda().train();alpha=torch.load(ROOT/'data/flat_alphas.pt',map_location='cuda',weights_only=True).float();params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);frozen={n:p.clone() for n,p in m.named_parameters() if not p.requires_grad}
 order=np.random.default_rng(20260918).permutation(len(tr));start=None
 for i in range(60):
  if i==10:torch.cuda.synchronize();start=time.monotonic()
  xs,y=tr.batch(order[i*256:(i+1)*256],'cuda');opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):out=m(*xs);loss,cls,aux=objective(out,y,alpha,cfg['contrastive_weight'])
  loss.backward();torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True);opt.step();assert torch.isfinite(loss)
 torch.cuda.synchronize();seconds=time.monotonic()-start
 for n,p in m.named_parameters():
  if n in frozen:assert torch.equal(p,frozen[n])
 m.eval();torch.cuda.synchronize();start=time.monotonic()
 with torch.no_grad():
  for i in range(50):
   xs,_=dev.batch(np.arange(i*256,(i+1)*256),'cuda')
   with torch.autocast('cuda',dtype=torch.bfloat16):out=m(*xs)
 torch.cuda.synchronize();predict_seconds=time.monotonic()-start
 report.update(passed=True,benchmark_training_steps=50,training_seconds=seconds,training_rows_per_second=12800/seconds,predict_rows_per_second=12800/predict_seconds,trainable_parameters=sum(p.numel() for p in params),frozen_heads_unchanged=True,peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9,completed=time.time())
 atomic_json(report,ROOT/'preflight.json');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

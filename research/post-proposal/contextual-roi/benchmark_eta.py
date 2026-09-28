import sys,time,json,torch,numpy as np
from pathlib import Path
sys.path.insert(0,'/data/repos/ROAD_Reason/research/post-proposal/contextual-roi')
from model import ContextualRoIHead,objective
r=Path('/data/repos/wiki/artifacts/contextual-roi-20260917');torch.set_num_threads(4)
result={}
for fusion in ['mlp','attention']:
 m=ContextualRoIHead(torch.randn(184,512),fusion=fusion).cuda();o=torch.optim.AdamW(m.parameters(),lr=.0001);alpha=torch.full((184,),.5,device='cuda')
 cpu=[torch.randn(256,1024),torch.randn(256,1024),torch.randn(256,16,1024),torch.tensor([[.1,.1,.8,.8]]*256)];y=torch.zeros(256,184,device='cuda');y[:,98]=1
 times=[]
 for i in range(35):
  torch.cuda.synchronize();t=time.perf_counter();x=[a.cuda() for a in cpu];o.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):loss,_,_=objective(m(*x),y,alpha,.001)
  loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);o.step();torch.cuda.synchronize()
  if i>=5:times.append(time.perf_counter()-t)
 with torch.no_grad():
  torch.cuda.synchronize();t=time.perf_counter()
  for i in range(30):
   x=[a.cuda() for a in cpu]
   with torch.autocast('cuda',dtype=torch.bfloat16):out=m(*x)
   p=out['logits'].float().sigmoid().cpu().numpy()
  torch.cuda.synchronize();infer=(time.perf_counter()-t)/30
 result[fusion]={'train_batch_seconds':float(np.median(times)),'infer_batch_seconds':infer,'three_epoch_compute_hours':float(np.median(times))*30090/3600,'val_forward_hours':infer*24079/3600}
 del m,o;torch.cuda.empty_cache()
result['time']=time.time();result['scope']='Synthetic tensors with real shapes, copies and head/loss/optimizer; concurrent scene extraction. Excludes disk IO, checkpoints, AP and startup. Compute component only.'
(r/'eta-benchmark.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

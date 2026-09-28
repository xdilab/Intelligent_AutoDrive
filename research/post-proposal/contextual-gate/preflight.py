"""Check real held-out data, mappings, frozen experts, router gradients and resume."""
import json,time,tempfile
import numpy as np
import torch
from common import ROOT,BASE,parents,atomic_json,atomic_torch,ContextualRoIHead,Cache,file_sha
from router import Router,composition_map,actor_features,ranking_loss

def main():
 torch.set_num_threads(4);prep=json.loads((ROOT/'preparation.json').read_text());ready=json.loads((ROOT/'cache-ready.json').read_text());assert ready['passed'] and ready['preparation_sha256']==file_sha(ROOT/'preparation.json')
 for name,digest in ready['files'].items():assert file_sha(ROOT/'data'/name)==digest
 labels=json.loads((BASE/'data/shared-frames.json').read_text())['labels'];mapping=composition_map(labels);assert mapping.shape==(135,3) and (mapping[:49,2]==0).all() and (mapping[49:,2]>=33).all()
 data=Cache(ROOT/'data','gate');bank=torch.load(BASE/'data/phrase_embeds.pt',map_location='cpu',weights_only=False)['embeds'];x,y=data.batch(np.arange(32),'cuda');outputs=[]
 for path in parents(0):
  ck=torch.load(path/'best.pt',map_location='cpu',weights_only=False);assert ck['signature']==json.loads((path/'complete.json').read_text())['signature'];m=ContextualRoIHead(bank,fusion='attention').cuda().eval();m.load_state_dict(ck['model']);m.requires_grad_(False)
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):outputs.append(m(*x)['logits'].float().sigmoid())
  assert not any(p.requires_grad for p in m.parameters());del m
 act=actor_features(x[0],x[1],x[3]);params={}
 for kind in ['class','generic','factorized']:
  m=Router(mapping,torch.ones(135)*5,.5,kind).cuda();params[kind]=sum(p.numel() for p in m.parameters());opt=torch.optim.AdamW(m.parameters(),lr=1e-3);classes=torch.arange(32,device='cuda');p,g=m(*outputs,act,classes);assert p.shape==(32,) and torch.isfinite(p).all();torch.testing.assert_close(g,torch.full_like(g,.5));loss=ranking_loss(p,2,8);loss.backward();assert any(v.grad is not None and v.grad.abs().sum()>0 for v in m.parameters());opt.step()
  with tempfile.TemporaryDirectory() as td:
   from pathlib import Path
   path=Path(td)/'test.pt';atomic_torch({'model':m.state_dict(),'optimizer':opt.state_dict()},path);ck=torch.load(path,weights_only=False);new=Router(mapping,torch.ones(135)*5,.5,kind).cuda();new.load_state_dict(ck['model']);torch.testing.assert_close(m(*outputs,act,classes)[0],new(*outputs,act,classes)[0],atol=0,rtol=0)
   newopt=torch.optim.AdamW(new.parameters(),lr=1e-3);newopt.load_state_dict(ck['optimizer'])
   for model,optimizer in [(m,opt),(new,newopt)]:
    optimizer.zero_grad();prob,_=model(*outputs,act,classes);ranking_loss(prob,2,8).backward();optimizer.step()
   for key,value in m.state_dict().items():torch.testing.assert_close(value,new.state_dict()[key],atol=0,rtol=0)
 assert abs(params['generic']-params['factorized'])/params['factorized']<.05
 torch.cuda.synchronize();start=time.time()
 for _ in range(100):
  opt.zero_grad();p,g=m(*outputs,act,classes);loss=ranking_loss(p,2,8);loss.backward();opt.step()
 torch.cuda.synchronize();atomic_json({'passed':True,'time':time.time(),'rows':len(data),'parameters':params,'training_smoke_100_steps_seconds':time.time()-start,'fit_videos':45,'selection_videos':45,'mapping':'derived from verified label strings; no childs arrays','frozen_experts':True},ROOT/'preflight.json')
 print('PREFLIGHT_PASS',params,flush=True)
if __name__=='__main__':main()

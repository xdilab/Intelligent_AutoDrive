"""Paired initialization, concatenative route gradients, real-cache training and resume checks."""
from pathlib import Path
import importlib.util,json,time,shutil,io
import numpy as np
import torch
from model import ContextualRoIHead
from dcb import DCBState,objective
from train_cached import Cache,file_sha,atomic_json
ROOT=Path('/data/repos/wiki/artifacts/contextual-roi-concat-20260924');CODE=Path(__file__).parent

def main():
 torch.set_num_threads(4);cfg=json.loads((ROOT/'protocol.json').read_text());assert file_sha(ROOT/'cache-ready.json')==cfg['cache_marker_sha256'];assert shutil.disk_usage(ROOT/'runs').free>5*1024**3
 spec=importlib.util.spec_from_file_location('reference',CODE.parent/'contextual-roi-dcb/model.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 bank=torch.load(ROOT/'data/phrase_embeds.pt',weights_only=False)['embeds'].float();alpha=torch.load(ROOT/'data/flat_alphas.pt',weights_only=True).float().cuda()
 tr=Cache(ROOT/'data','train');dev=Cache(ROOT/'data','dev');val=Cache(ROOT/'data','val');xs,y=tr.batch(np.arange(cfg['batch_size']),'cuda');checks=[]
 official=json.loads((ROOT/'data/shared-frames.json').read_text())
 for seed in cfg['seeds']:
  for kind in ['classification','contrastive-all184','global']:
   dest=Path(cfg['baseline_root'])/'runs'/f'attention-{kind}-dcb-seed{seed}'
   r=json.loads((dest/'detector-results.json').read_text());assert r['n_frames']==36717 and r['candidate_sha256']==official['candidate_sha256'] and r['frame_sha256']==official['frame_sha256']
  torch.manual_seed(seed);ref=old.ContextualRoIHead(bank,fusion='attention');rng=torch.get_rng_state();torch.manual_seed(seed);m=ContextualRoIHead(bank,fusion='attention');assert torch.equal(rng,torch.get_rng_state());assert all(torch.equal(v,m.state_dict()[k]) for k,v in ref.state_dict().items());added=sum(p.numel() for p in m.parameters())-sum(p.numel() for p in ref.parameters());assert added==1310720
  ref=ref.cuda().eval();m=m.cuda().eval()
  with torch.no_grad():
   for dtype in [None,torch.bfloat16]:
    with torch.autocast('cuda',dtype=dtype or torch.bfloat16,enabled=dtype is not None):a=ref(*xs);b=m(*xs)
    for k in a:torch.testing.assert_close(a[k],b[k],atol=1e-5 if dtype is None else .02,rtol=1e-5 if dtype is None else .01)
  checks.append({'seed':seed,'common_parameters_bit_identical':True,'RNG_identical':True,'initial_outputs_equivalent_fp32_bf16':True,'added_parameters':added})
  del ref,m;torch.cuda.empty_cache()
 timings={}
 for weight in [0.,.001]:
  torch.manual_seed(0);m=ContextualRoIHead(bank,fusion='attention').cuda();state=DCBState().cuda();opt=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);order=np.random.default_rng(0).permutation(len(tr));batch=cfg['batch_size']
  for step in range(110):
   if step==10:torch.cuda.synchronize();start=time.monotonic()
   xs,y=tr.batch(order[step*batch:(step+1)*batch],'cuda');opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):out=m(*xs);loss,cls,aux=objective(out,y,alpha,weight,state)
   assert torch.isfinite(loss);loss.backward()
   if step==0:
    grad=m.full_context_projection.weight.grad;assert all(grad[:,i*512:(i+1)*512].abs().sum()>0 for i in range(5))
    # Auxiliary loss independently reaches both learned towers.
    if weight:
     opt.zero_grad(set_to_none=True);out=m(*xs);_,_,aux=objective(out,y,alpha,weight,state,update=False);aux.backward();assert m.full_context_projection.weight.grad.abs().sum()>0 and m.text_adapter.net[-1].weight.grad.abs().sum()>0
     opt.zero_grad(set_to_none=True);out=m(*xs);loss,_,_=objective(out,y,alpha,weight,state,update=False);loss.backward()
   torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step()
  torch.cuda.synchronize();elapsed=time.monotonic()-start
  # Ensure model/optimizer/history survive the actual checkpoint serialization.
  buffer=io.BytesIO();torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'dcb_state':state.state_dict()},buffer);buffer.seek(0);ck=torch.load(buffer,weights_only=False);copy=ContextualRoIHead(bank,fusion='attention').cuda();copy.load_state_dict(ck['model']);assert all(torch.equal(v,copy.state_dict()[k]) for k,v in m.state_dict().items())
  timings[str(weight)]={'train_rows_per_second':100*batch/elapsed,'steps':100,'finite_loss':float(loss.detach()),'all_five_concat_blocks_receive_gradients':True,'checkpoint_roundtrip':True}
  del m,opt,state,copy;torch.cuda.empty_cache()
 gate=ROOT.parent/'contextual-gate-20260919'
 for name,digest in cfg['gate_inputs_sha256'].items():assert file_sha(gate/name)==digest
 atomic_json({'passed':True,'time':time.time(),'protocol_sha256':file_sha(ROOT/'protocol.json'),'checks':checks,'benchmark':timings,'rows':{'train':len(tr),'dev':len(dev),'val':len(val)},'no_real_study_checkpoint_modified':True},ROOT/'preflight.json');print(json.dumps(timings),flush=True)
if __name__=='__main__':main()

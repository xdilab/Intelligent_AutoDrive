"""Production-dimension initialization checks and disposable real-cache benchmark."""
import hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
import torch
from model import ContextualRoIHead,objective
from train_cached import Cache,atomic_json,file_sha
ROOT=Path('/data/repos/wiki/artifacts/contextual-roi-comp-20260919')
BASE=Path('/data/repos/ROAD_Reason/research/post-proposal')
def load(name):
 spec=importlib.util.spec_from_file_location(name,BASE/name/'model.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main():
 torch.set_num_threads(4);cfg=json.loads((ROOT/'protocol.json').read_text());assert file_sha(ROOT/'cache-ready.json')==cfg['cache_marker_sha256']
 bank=torch.load(ROOT/'data/phrase_embeds.pt',map_location='cpu',weights_only=False)['embeds'].float();assert bank.shape==(184,512)
 records=[];hashes=set()
 for seed in cfg['seeds']:
  torch.manual_seed(seed);new=ContextualRoIHead(bank,fusion='attention');state=new.state_dict()
  torch.manual_seed(seed);other=ContextualRoIHead(bank,fusion='attention')
  for k,v in state.items():assert torch.equal(v,other.state_dict()[k]),k
  del other
  for kind,folder in [('classification','contextual-roi'),('contrastive','contextual-roi-all184')]:
   run=Path(cfg['baseline_runs'][kind][seed]);done=json.loads((run/'complete.json').read_text());assert done['passed']
   assert done['signature']['code_sha256']['model.py']==file_sha(BASE/folder/'model.py')
   assert done['signature']['cache_marker_sha256']==cfg['cache_marker_sha256']
   result=json.loads((run/'detector-results.json').read_text());assert result['n_frames']==36717;hashes.add((result['frame_sha256'],result['candidate_sha256']))
   module=load(folder);torch.manual_seed(seed);old=module.ContextualRoIHead(bank,fusion='attention')
   shared={k:v for k,v in old.state_dict().items() if not k.startswith('classifier.')}
   assert set(shared)=={k for k in state if not k.startswith(('flat.','comp.'))}
   for k,v in shared.items():assert torch.equal(v,state[k]),(seed,kind,k)
   h=hashlib.sha256()
   for k,v in sorted(shared.items()):h.update(k.encode());h.update(v.numpy().tobytes())
   records.append({'seed':seed,'baseline':str(run),'backbone_sha256':h.hexdigest(),'bit_identical':True})
   assert sum(p.numel() for p in old.classifier.parameters())==cfg['old_readout_parameters']
   del old
  assert sum(p.numel() for mod in [new.flat,new.comp] for p in mod.parameters())==cfg['new_readout_parameters']
  del new
 assert len(hashes)==1
 shared=json.loads((ROOT/'data/shared-frames.json').read_text());assert next(iter(hashes))==(shared['frame_sha256'],shared['candidate_sha256'])
 device=torch.device('cuda');torch.manual_seed(0);m=ContextualRoIHead(bank,fusion='attention').to(device);opt=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
 alpha=torch.load(ROOT/'data/flat_alphas.pt',map_location='cpu',weights_only=True).float().to(device);tr=Cache(ROOT/'data','train');order=np.random.default_rng(0).permutation(len(tr));batch=cfg['batch_size']
 for step in range(70):
  if step==10:torch.cuda.synchronize();start=time.monotonic()
  x,y=tr.batch(order[step*batch:(step+1)*batch],device);opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):out=m(*x);loss,_,_=objective(out,y,alpha,.001)
  assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step()
 torch.cuda.synchronize();train_rate=60*batch/(time.monotonic()-start)
 assert all(torch.isfinite(p).all() for p in m.parameters())
 m.eval();torch.cuda.synchronize();start=time.monotonic()
 with torch.no_grad():
  for step in range(60):
   x,_=tr.batch(order[step*batch:(step+1)*batch],device)
   with torch.autocast('cuda',dtype=torch.bfloat16):out=m(*x)
   assert out['logits'].shape==(batch,184)
 torch.cuda.synchronize();infer_rate=60*batch/(time.monotonic()-start)
 result={'passed':True,'time':time.time(),'model_sha256':file_sha(Path(__file__).parent/'model.py'),'backbone_checks':records,'frame_candidate_hashes':list(hashes)[0],'train_rows_per_second_single_gpu':train_rate,'inference_rows_per_second_single_gpu':infer_rate,'benchmark_steps':60,'benchmark_disposable':True,'note':'Single-GPU short benchmark excludes full development AP, detector AP and two-lane contention; ETA must include historical end-to-end overhead.'}
 atomic_json(result,ROOT/'preflight.json');print(json.dumps(result,indent=2))
if __name__=='__main__':main()

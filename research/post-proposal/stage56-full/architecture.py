import argparse,json,time,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import base
from heads import load_head,load_mlp

class FullStage(base.Model):
 def __init__(self,cfg,condition,stage):
  super().__init__(cfg,condition);self.stage=stage
  root=Path(cfg['expert_study']);run=root/'runs/seed-0';checks=json.loads((root/'results/train-seed0.json').read_text())['checkpoint_sha256']
  assert base.sha(run/f'{stage}.pt')==checks[f'{stage}.pt']
  self.comp=load_mlp(run/f'{stage}.pt')
  if stage=='stage6':self.phrase=load_head(run/'head-phrase.pt')
  self.float().cuda().eval();self.parity_error=None
 def assemble(self,f):
  flat=self.head(f);parts=[flat[:,:49].sigmoid()]
  if self.stage=='stage6':parts.append(self.phrase(f).sigmoid()[:,49:])
  parts.append(f);z=self.comp(torch.cat(parts,1));return torch.cat([flat[:,:49],z],1)
 def forward(self,x,key):
  _,sim,f=super().forward(x,key);return self.assemble(f),sim,f
 def parity(self):
  # Compare complete inference against independently loaded original stage modules in FP32.
  run=Path(self.cfg['expert_study'])/'runs/seed-0';flat=load_head(run/'head-flat.pt').cuda();comp=load_mlp(run/f'{self.stage}.pt').cuda();g=torch.Generator(device='cuda').manual_seed(42);x=torch.randn(9,1024,device='cuda',generator=g)
  with torch.no_grad():
   raw=flat(x);parts=[raw.sigmoid()[:,:49]]
   if self.stage=='stage6':parts.append(load_head(run/'head-phrase.pt').cuda()(x).sigmoid()[:,49:])
   parts.append(x);expected=torch.cat([raw[:,:49],comp(torch.cat(parts,1))],1);actual=self.assemble(x);err=float((expected-actual).abs().max());assert torch.allclose(expected,actual,atol=1e-6,rtol=1e-6)
  return err

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['stage5','stage6'],required=True);p.add_argument('--condition',choices=['frozen','classification','contrastive'],required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();root=a.root;cfg=json.loads((root/'code/protocol.json').read_text());data=json.loads((root/'manifest.json').read_text());assert data['protocol']==cfg
 torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed']);torch.set_num_threads(8)
 for name,path in [('phrase_embeds',Path(cfg['original_study'])/'data/phrase_embeds.pt'),('alphas',Path(cfg['original_study'])/'data/flat_alphas.pt')]:assert base.sha(path)==data['source_sha256'][name]
 m=FullStage(cfg,a.condition,a.stage);parity=m.parity();vision=[p for p in m.encoder.parameters() if p.requires_grad];other=[p for n,p in m.named_parameters() if p.requires_grad and not n.startswith('encoder.')];params=vision+other;opt=torch.optim.AdamW([{'params':vision,'lr':cfg['vision_lr']},{'params':other,'lr':cfg['head_lr']}],weight_decay=cfg['weight_decay']);before=base.frozen_hash(m);vb=[p.detach().clone() for p in vision];report={'stage':a.stage,'condition':a.condition,'protocol':cfg,'manifest_sha256':base.sha(root/'manifest.json'),'inference_parity_max_abs_error':parity,'frozen_sha256_before':before,'trainable_names':[n for n,p in m.named_parameters() if p.requires_grad],'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};name=f'{a.stage}-{a.condition}';dest=root/'runs'/name;dest.mkdir(parents=True,exist_ok=True)
 if a.smoke:
  row=next(r for r in data['train'] if np.asarray(r['targets'])[:,98:].sum()>0);x,y=base.crops(row,cfg)
  with torch.autocast('cuda',dtype=torch.bfloat16):z,s,f=m(x,row['key_t']);lc=base.focal(z.float(),y,m.alpha);la=base.multi_positive_loss(s,y[:,98:]);loss=lc+cfg['contrastive_lambda']*la
  report['classification_visual_gradient_norm']=base.grad_norm(lc,vision);report['contrastive_visual_gradient_norm']=base.grad_norm(la,vision);report['composition_mlp_gradient_norm']=base.grad_norm(lc,list(m.comp.parameters()));assert min(report[k] for k in ['classification_visual_gradient_norm','contrastive_visual_gradient_norm','composition_mlp_gradient_norm'])>0
  if a.stage=='stage6':report['phrase_branch_gradient_norm']=base.grad_norm(lc,list(m.phrase.parameters()));assert report['phrase_branch_gradient_norm']>0
  loss.backward();assert all(p.grad is None for p in m.parameters() if not p.requires_grad);torch.nn.utils.clip_grad_norm_(params,cfg['gradient_clip'],error_if_nonfinite=True);opt.step()
 else:
  report['baseline']=base.evaluate(m,data['dev'],cfg);best=report['baseline']['triplet_crop_AP'];report['selected']='baseline';report['epochs']=[];start=time.time()
  for ep in range(cfg['epochs']):
   total=0;n=0
   for j in np.random.default_rng(cfg['seed']+ep).permutation(len(data['train'])):
    row=data['train'][j];x,y=base.crops(row,cfg);opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):z,s,f=m(x,row['key_t']);lc=base.focal(z.float(),y,m.alpha);la=base.multi_positive_loss(s,y[:,98:]);loss=lc+(cfg['contrastive_lambda']*la if a.condition=='contrastive' else 0)
    assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(params,cfg['gradient_clip'],error_if_nonfinite=True);opt.step();total+=float(loss)*len(y);n+=len(y)
   metrics=base.evaluate(m,data['dev'],cfg);metrics.update(epoch=ep+1,training_objective=total/n);report['epochs'].append(metrics);ck=dest/f'epoch-{ep+1}.pt';torch.save({'state_dict':{n:p.detach().cpu() for n,p in m.named_parameters() if p.requires_grad},'stage':a.stage,'condition':a.condition,'protocol':cfg,'epoch':ep+1},ck)
   if metrics['triplet_crop_AP']>best:best=metrics['triplet_crop_AP'];report['selected']=str(ck);report['selected_sha256']=base.sha(ck)
   (root/f'results/{name}.partial.json').write_text(json.dumps(report,indent=2));print('EPOCH',ep+1,metrics['triplet_crop_AP'],flush=True)
  report['training_wall_seconds']=time.time()-start
 report['vision_parameter_update_l2']=float(sum((p.detach()-b).square().sum() for p,b in zip(vision,vb)).sqrt()) if vision else 0
 if a.condition!='frozen':assert report['vision_parameter_update_l2']>0
 report['frozen_sha256_after']=base.frozen_hash(m);assert before==report['frozen_sha256_after'];report['peak_gpu_gb']=torch.cuda.max_memory_allocated()/1e9;report['completed_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());report['passed']=True
 (root/f'results/{"smoke-" if a.smoke else ""}{name}.json').write_text(json.dumps(report,indent=2));print('COMPLETE',name,flush=True)
if __name__=='__main__':main()

import argparse,hashlib,json,time,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from torchvision.ops import roi_align

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def multi_positive_loss(logits,targets):
 valid=targets.sum(1)>0
 if not valid.any():return logits.sum()*0
 y=targets[valid];return -((y*F.log_softmax(logits[valid].float(),dim=1)).sum(1)/y.sum(1)).mean()
def focal(logits,y,alpha):
 p=logits.sigmoid();pt=y*p+(1-y)*(1-p)
 return ((y*alpha+(1-y)*(1-alpha))*(1-pt).pow(2)*F.binary_cross_entropy_with_logits(logits,y,reduction='none')).mean()
def crop_ap(y,p):
 vals=[]
 for c in range(y.shape[1]):
  yy=y[np.argsort(-p[:,c]),c]>0;n=yy.sum()
  if not n:vals.append(0.);continue
  prec=np.cumsum(yy)/np.arange(1,len(yy)+1);vals.append(float(np.maximum.accumulate(prec[::-1])[::-1][yy].sum()/n*100))
 return vals

def crops(row,cfg):
 imgs=[]
 for fid,digest in zip(row['fids'],row['frame_sha256']):
  p=Path(cfg['frames'])/row['video']/f'{fid:05d}.jpg';assert sha(p)==digest
  with Image.open(p) as im:imgs.append(np.asarray(im.convert('RGB')).copy())
 ft=torch.from_numpy(np.stack(imgs)).cuda().permute(0,3,1,2).float()/255;H,W=imgs[0].shape[:2]
 mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None];ft=(ft-mean)/std;b=np.array(row['boxes']);n=len(b);cx=(b[:,0]+b[:,2])/2*W;cy=(b[:,1]+b[:,3])/2*H;bw=np.maximum((b[:,2]-b[:,0])*W,8)*2;bh=np.maximum((b[:,3]-b[:,1])*H,8)*2
 boxes=torch.tensor(np.stack([np.clip(cx-bw/2,0,W-1),np.clip(cy-bh/2,0,H-1),np.clip(cx+bw/2,1,W),np.clip(cy+bh/2,1,H)],1),device='cuda',dtype=torch.float32);rois=torch.cat([torch.arange(8,device='cuda').repeat_interleave(n)[:,None],boxes.repeat(8,1)],1)
 x=roi_align(ft,rois,output_size=(224,224),spatial_scale=1.,aligned=True).view(8,n,3,224,224).permute(1,0,2,3,4).contiguous();y=torch.tensor(row['targets'],device='cuda',dtype=torch.float32);assert x.shape==(n,8,3,224,224) and torch.isfinite(x).all();return x,y

class Model(nn.Module):
 def __init__(self,cfg,condition):
  super().__init__();from transformers import AutoConfig
  from transformers.dynamic_module_utils import get_class_from_dynamic_module
  from huggingface_hub import hf_hub_download
  from safetensors.torch import load_file
  ac=AutoConfig.from_pretrained(cfg['model'],revision=cfg['revision'],trust_remote_code=True,local_files_only=True);cls=get_class_from_dynamic_module('modeling_internvideo2encoder.InternVideo2_CLIP_small',cfg['model'],revision=cfg['revision'],local_files_only=True);self.encoder=cls(ac);self.encoder.load_state_dict(load_file(hf_hub_download(cfg['model'],'model.safetensors',revision=cfg['revision'],local_files_only=True)),strict=True)
  for p in self.encoder.parameters():p.requires_grad=False
  block=self.encoder.vision_encoder.blocks[-1];block.with_cp=False
  if condition!='frozen':
   for p in block.parameters():p.requires_grad=True
  run=Path(cfg['expert_study'])/'runs/seed-0';tr=json.loads((Path(cfg['expert_study'])/'results/train-seed0.json').read_text())
  for name in ['head-flat.pt','head-phrase.pt']:assert sha(run/name)==tr['checkpoint_sha256'][name]
  flat=torch.load(run/'head-flat.pt',map_location='cpu',weights_only=False)['state'];phrase=torch.load(run/'head-phrase.pt',map_location='cpu',weights_only=False)['state'];self.head=nn.Linear(1024,184);self.head.load_state_dict(flat);self.proj=nn.Linear(1024,512);self.proj.load_state_dict({'weight':phrase['proj.weight'],'bias':phrase['proj.bias']})
  for p in self.proj.parameters():p.requires_grad=condition=='contrastive'
  original=torch.load(Path(cfg['original_study'])/'data/phrase_embeds.pt',map_location='cpu',weights_only=False);P=F.normalize(original['embeds'].float(),dim=1);assert torch.allclose(P,phrase['P'],atol=1e-6);self.register_buffer('P',P[98:]);assert self.P.shape==(86,512)
  self.register_buffer('alpha',torch.load(Path(cfg['original_study'])/'data/flat_alphas.pt',map_location='cpu',weights_only=True).float());self.tokens=None;block.register_forward_hook(self.capture);self.cfg=cfg;self.float().cuda().eval()
 def capture(self,module,inputs,output):self.tokens=output if isinstance(output,torch.Tensor) else output[0]
 def forward(self,x,key):
  self.tokens=None;self.encoder.encode_vision(x);tok=self.tokens;assert tok is not None and tok.shape[-1]==1024 and tok.shape[1]>=2048
  feature=tok[:,-2048:,:].reshape(len(x),8,256,1024)[:,key].float().mean(1);self.tokens=None
  logits=self.head(feature);sim=F.normalize(self.proj(feature).float(),dim=-1)@self.P.T/self.cfg['temperature'];return logits,sim,feature

def frozen_hash(m):
 h=hashlib.sha256()
 for n,p in m.named_parameters():
  if not p.requires_grad:h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def grad_norm(loss,params):
 gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True);return float(sum(g.float().square().sum() for g in gs if g is not None).sqrt())
@torch.no_grad()
def evaluate(m,rows,cfg):
 ys=[];ps=[];ls=0;N=0;cs=0;validn=0
 for row in rows:
  x,y=crops(row,cfg)
  with torch.autocast('cuda',dtype=torch.bfloat16):z,s,f=m(x,row['key_t']);l=focal(z.float(),y,m.alpha);c=multi_positive_loss(s,y[:,98:])
  ys.append(y.cpu().numpy());ps.append(z.float().sigmoid().cpu().numpy());ls+=float(l)*len(y);N+=len(y);nv=int((y[:,98:].sum(1)>0).sum());cs+=float(c)*nv;validn+=nv
 y=np.concatenate(ys);p=np.concatenate(ps);ap=crop_ap(y[:,98:],p[:,98:]);return {'focal_loss':ls/N,'contrastive_loss':cs/max(1,validn),'triplet_crop_AP':float(np.mean(ap)),'triplet_per_class_AP':ap,'rows':N,'valid_contrastive_rows':validn}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--condition',choices=['frozen','classification','contrastive'],default='contrastive');parser.add_argument('--smoke',action='store_true');args=parser.parse_args();root=args.root;cfg=json.loads((root/'code/protocol.json').read_text());data=json.loads((root/'manifest.json').read_text());assert data['protocol']==cfg;torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed']);torch.set_num_threads(8);assert torch.cuda.is_available()
 for name,path in [('phrase_embeds',Path(cfg['original_study'])/'data/phrase_embeds.pt'),('alphas',Path(cfg['original_study'])/'data/flat_alphas.pt')]:assert sha(path)==data['source_sha256'][name]
 m=Model(cfg,args.condition);vision=[p for p in m.encoder.parameters() if p.requires_grad];head=list(m.head.parameters());aux=[p for p in m.proj.parameters() if p.requires_grad];params=vision+head+aux;opt=torch.optim.AdamW([{'params':vision,'lr':cfg['vision_lr']},{'params':head+aux,'lr':cfg['head_lr']}],weight_decay=cfg['weight_decay']);frozen_before=frozen_hash(m);vbefore=[p.detach().clone() for p in vision];report={'condition':args.condition,'protocol':cfg,'manifest_sha256':sha(root/'manifest.json'),'trainable_parameters':sum(p.numel() for p in params),'trainable_names':[n for n,p in m.named_parameters() if p.requires_grad],'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'frozen_sha256_before':frozen_before}
 if args.smoke:
  row=next(r for r in data['train'] if np.array(r['targets'])[:,98:].sum()>0);x,y=crops(row,cfg)
  with torch.autocast('cuda',dtype=torch.bfloat16):z,s,f=m(x,row['key_t']);lc=focal(z.float(),y,m.alpha);la=multi_positive_loss(s,y[:,98:]);loss=lc+cfg['contrastive_lambda']*la
  assert f.requires_grad;report['classification_visual_gradient_norm']=grad_norm(lc,vision);report['contrastive_visual_gradient_norm']=grad_norm(la,vision);assert min(report['classification_visual_gradient_norm'],report['contrastive_visual_gradient_norm'])>0
  loss.backward();assert all(p.grad is None for p in m.encoder.text_encoder.parameters());assert all(p.grad is None for p in m.parameters() if not p.requires_grad);torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();report['vision_parameter_update_l2']=float(sum((p.detach()-b).square().sum() for p,b in zip(vision,vbefore)).sqrt());assert report['vision_parameter_update_l2']>0;report['feature_shape']=list(f.shape);report['batch_crops']=len(x)
 else:
  tr=data['train'];dv=data['dev'];report['baseline']=evaluate(m,dv,cfg);best=report['baseline']['triplet_crop_AP'];report['selected']='baseline';report['epochs']=[];dest=root/'runs'/args.condition;dest.mkdir(parents=True,exist_ok=True);print('BASELINE',json.dumps(report['baseline']),flush=True);start=time.time();steps=0
  for ep in range(cfg['epochs']):
   order=np.random.default_rng(cfg['seed']+ep).permutation(len(tr));total=0;n=0
   for j in order:
    x,y=crops(tr[j],cfg);opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):z,s,f=m(x,tr[j]['key_t']);lc=focal(z.float(),y,m.alpha);la=multi_positive_loss(s,y[:,98:]);loss=lc+(cfg['contrastive_lambda']*la if args.condition=='contrastive' else 0)
    assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(params,cfg['gradient_clip'],error_if_nonfinite=True);opt.step();total+=float(loss)*len(y);n+=len(y);steps+=1
    if steps%32==0:print(json.dumps({'epoch':ep+1,'steps':steps,'loss':float(loss),'seconds':time.time()-start}),flush=True)
   metrics=evaluate(m,dv,cfg);metrics.update(epoch=ep+1,training_objective=total/n);report['epochs'].append(metrics);ck=dest/f'epoch-{ep+1}.pt';torch.save({'state_dict':{n:p.detach().cpu() for n,p in m.named_parameters() if p.requires_grad},'condition':args.condition,'protocol':cfg,'manifest_sha256':report['manifest_sha256'],'epoch':ep+1,'metrics':metrics},ck)
   if metrics['triplet_crop_AP']>best:best=metrics['triplet_crop_AP'];report['selected']=str(ck);report['selected_sha256']=sha(ck)
   (root/f'results/{args.condition}.partial.json').write_text(json.dumps(report,indent=2));print('EPOCH',json.dumps(metrics),flush=True)
  report['training_steps']=steps;report['training_wall_seconds']=time.time()-start;report['vision_parameter_update_l2']=float(sum((p.detach()-b).square().sum() for p,b in zip(vision,vbefore)).sqrt()) if vision else 0.
 report['frozen_sha256_after']=frozen_hash(m);assert report['frozen_sha256_before']==report['frozen_sha256_after'];report['peak_gpu_gb']=torch.cuda.max_memory_allocated()/1e9;report['completed_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());report['passed']=True;name='smoke' if args.smoke else args.condition;(root/f'results/{name}.json').write_text(json.dumps(report,indent=2));print('COMPLETE',name,'peak_gpu_gb',report['peak_gpu_gb'],flush=True)
if __name__=='__main__':main()

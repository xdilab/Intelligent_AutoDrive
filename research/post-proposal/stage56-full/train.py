import argparse,json,time,random,signal
from pathlib import Path
import numpy as np
import torch
from architecture import FullStage
import base

class Rows:
 def __init__(self,path):
  self.path=path;self.offset=[]
  with path.open('rb') as f:
   while True:
    pos=f.tell();line=f.readline()
    if not line:break
    self.offset.append(pos)
 def __len__(self):return len(self.offset)
 def __getitem__(self,i):
  with self.path.open('rb') as f:f.seek(self.offset[i]);d=json.loads(f.readline())
  y=np.zeros((len(d['boxes']),184),np.float32)
  for j,ids in enumerate(d.pop('positive_indices')):y[j,ids]=1
  d['targets']=y;return d

def parts(row,size):
 for start in range(0,len(row['boxes']),size):
  d=dict(row);d['boxes']=row['boxes'][start:start+size];d['targets']=row['targets'][start:start+size];yield d
@torch.no_grad()
def evaluate(m,rows,cfg):
 yy=[];pp=[]
 for i in range(len(rows)):
  for r in parts(rows[i],64):
   x,y=base.crops(r,cfg)
   with torch.autocast('cuda',dtype=torch.bfloat16):z,_,_=m(x,r['key_t'])
   yy.append(y[:,98:].cpu().numpy());pp.append(z[:,98:].float().sigmoid().cpu().numpy())
  if i%2000==0:print('EVAL',i,len(rows),flush=True)
 y=np.concatenate(yy);p=np.concatenate(pp);ap=base.crop_ap(y,p);return {'triplet_crop_AP':float(np.mean(ap)),'triplet_per_class_AP':ap,'rows':len(y)}

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['stage5','stage6'],required=True);p.add_argument('--condition',choices=['frozen','classification','contrastive'],required=True);a=p.parse_args();root=Path('/work/bbyrd1/stage56-full-20260914');cfg=json.loads((root/'code/protocol.json').read_text());prep=json.loads((root/'results/preparation.json').read_text());assert prep['protocol']==cfg
 for s in ['train','dev']:assert base.sha(root/f'data/{s}.jsonl')==prep['data_sha256'][s]
 torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed']);torch.set_num_threads(8)
 m=FullStage(cfg,a.condition,a.stage);assert m.parity()<=1e-6
 vision=[p for p in m.encoder.parameters() if p.requires_grad];other=[p for n,p in m.named_parameters() if p.requires_grad and not n.startswith('encoder.')];params=vision+other;opt=torch.optim.AdamW([{'params':vision,'lr':cfg['vision_lr']},{'params':other,'lr':cfg['head_lr']}],weight_decay=cfg['weight_decay']);frozen=base.frozen_hash(m)
 name=f'{a.stage}-{a.condition}';dest=root/'runs'/name;dest.mkdir(parents=True,exist_ok=True);resume=dest/'resume.pt';tr=Rows(root/'data/train.jsonl');dv=Rows(root/'data/dev.jsonl');ep=0;pos=0;steps=0
 report={'stage':a.stage,'condition':a.condition,'protocol':cfg,'data_sha256':prep['data_sha256'],'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'epochs':[],'selected':'baseline','trainable_names':[n for n,p in m.named_parameters() if p.requires_grad],'frozen_sha256_before':frozen}
 def save(path,next_epoch,next_pos):
  state={n:p.detach().cpu() for n,p in m.named_parameters() if p.requires_grad};temp=path.with_suffix('.tmp');torch.save({'state':state,'optimizer':opt.state_dict(),'epoch':next_epoch,'position':next_pos,'steps':steps,'report':report,'protocol':cfg,'data_sha256':prep['data_sha256']},temp);temp.replace(path)
 if resume.exists():
  ck=torch.load(resume,map_location='cpu',weights_only=False);assert ck['protocol']==cfg and ck['data_sha256']==prep['data_sha256'];named=dict(m.named_parameters())
  with torch.no_grad():
   for n,v in ck['state'].items():named[n].copy_(v)
  opt.load_state_dict(ck['optimizer']);ep=ck['epoch'];pos=ck['position'];steps=ck['steps'];report=ck['report']
 else:
  report['baseline']=evaluate(m,dv,cfg);save(resume,0,0)
 for epoch in range(ep,cfg['epochs']):
  order=np.random.default_rng(cfg['seed']+epoch).permutation(len(tr));cursor=pos if epoch==ep else 0;loss_sum=0;row_count=0;start=time.time()
  while cursor<len(order):
   bucket=[];count=0
   while cursor<len(order) and count<cfg['effective_batch']:
    row=tr[int(order[cursor])];bucket.append(row);count+=len(row['boxes']);cursor+=1
   if not count:continue
   opt.zero_grad(set_to_none=True);value=0
   for row in bucket:
    for r in parts(row,cfg['microbatch']):
     x,y=base.crops(r,cfg)
     with torch.autocast('cuda',dtype=torch.bfloat16):z,s,f=m(x,r['key_t']);lc=base.focal(z.float(),y,m.alpha);la=base.multi_positive_loss(s,y[:,98:]);loss=(lc+(cfg['contrastive_lambda']*la if a.condition=='contrastive' else 0))*len(y)/count
     assert torch.isfinite(loss);loss.backward();value+=float(loss)
   torch.nn.utils.clip_grad_norm_(params,cfg['gradient_clip'],error_if_nonfinite=True);opt.step();steps+=1;loss_sum+=value*count;row_count+=count
   if steps%25==0:
    save(resume,epoch,cursor);(root/f'results/{name}.progress.json').write_text(json.dumps({'stage':a.stage,'condition':a.condition,'epoch':epoch+1,'frames_done':cursor,'frames_total':len(order),'steps':steps,'training_objective':value,'segment_rows_per_second':row_count/max(time.time()-start,1),'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}));print('TRAIN',epoch+1,cursor,len(order),steps,value,flush=True)
  metrics=evaluate(m,dv,cfg);metrics.update(epoch=epoch+1,training_objective_segment=loss_sum/max(row_count,1));report['epochs'].append(metrics)
  previous=max([report['baseline']['triplet_crop_AP']]+[e['triplet_crop_AP'] for e in report['epochs'][:-1]])
  if metrics['triplet_crop_AP']>previous:report['selected']=f'epoch-{epoch+1}';save(dest/'best.pt',epoch+1,0)
  save(dest/f'epoch-{epoch+1}.pt',epoch+1,0);save(resume,epoch+1,0);(root/f'results/{name}.partial.json').write_text(json.dumps(report,indent=2));pos=0
 report['frozen_sha256_after']=base.frozen_hash(m);assert frozen==report['frozen_sha256_after'];report['completed_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());report['passed']=True;report['peak_gpu_gb']=torch.cuda.max_memory_allocated()/1e9;(root/f'results/{name}.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()

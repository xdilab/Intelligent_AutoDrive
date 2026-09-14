"""Bounded BDD-X two-tower pilot. No ROAD-Waymo validation is used for selection."""
import argparse, json, random, time, hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REV='1f9fca1389fd883defc652634d95a21121c85a8c'
MODEL='OpenGVLab/InternVideo2_CLIP_S'

def load_model():
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    cfg=AutoConfig.from_pretrained(MODEL,revision=REV,trust_remote_code=True,local_files_only=True)
    cls=get_class_from_dynamic_module('modeling_internvideo2encoder.InternVideo2_CLIP_small',MODEL,revision=REV,local_files_only=True)
    model=cls(cfg)
    model.load_state_dict(load_file(hf_hub_download(MODEL,'model.safetensors',revision=REV,local_files_only=True)),strict=True)
    # FP32 optimizer parameters; bf16 autocast for encoder operations.
    model=model.float().cuda()
    for p in model.parameters(): p.requires_grad=False
    modules=list(model.vision_encoder.blocks[-2:])+[model.vision_encoder.clip_projector,model.vision_encoder.fc_norm,model.vision_align]+list(model.text_encoder.transformer[-2:])
    for block in model.vision_encoder.blocks[-2:]: block.with_cp=False
    for m in modules:
        for p in m.parameters():p.requires_grad=True
    model.text_encoder.projection_layer.requires_grad=True
    model.temp.requires_grad=True
    return model

def positive_mask(texts,device):
    return torch.tensor([[a==b for b in texts] for a in texts],device=device,dtype=torch.bool)

def loss_fn(v,t,texts,temp):
    z=F.normalize(v.float(),dim=-1)@F.normalize(t.float(),dim=-1).T/temp.clamp(.01,.5)
    pos=positive_mask(texts,z.device)
    def loss(a,p):return (torch.logsumexp(a,1)-torch.logsumexp(a.masked_fill(~p,-torch.inf),1)).mean()
    return (loss(z,pos)+loss(z.T,pos.T))/2

class Clips(Dataset):
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        import av
        from PIL import Image
        r=self.rows[i]
        with av.open(r['video']) as c:
            frames=[f.to_image() for f in c.decode(video=0)]
        if len(frames)<8:raise ValueError(f"Fewer than eight real frames: {r['video']}")
        ids=np.linspace(0,len(frames)-1,8).round().astype(int)
        x=np.stack([np.asarray(frames[j].resize((224,224),Image.Resampling.BICUBIC)) for j in ids])
        x=torch.from_numpy(x.copy()).permute(0,3,1,2).float()/255
        x=(x-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
        return x,r['text']

@torch.no_grad()
def evaluate(m,loader):
    m.eval();vs=[];ts=[];texts=[]
    for x,txt in loader:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            vs.append(F.normalize(m.encode_vision(x.cuda()).float(),dim=-1));ts.append(F.normalize(m.encode_text(m.tokenizer(list(txt)).cuda()).float(),dim=-1))
        texts+=list(txt)
    v=torch.cat(vs);t=torch.cat(ts);s=v@t.T;pos=positive_mask(texts,s.device)
    return {'loss':float(loss_fn(v,t,texts,m.temp)), 'v2t_r1':float(pos.gather(1,s.argmax(1)[:,None]).float().mean()),'t2v_r1':float(pos.T.gather(1,s.T.argmax(1)[:,None]).float().mean()),'n':len(texts)}

def main():
    a=argparse.ArgumentParser();a.add_argument('--manifest',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--epochs',type=int,default=3);a.add_argument('--batch-size',type=int,default=16);a.add_argument('--smoke',action='store_true');args=a.parse_args()
    torch.manual_seed(0);np.random.seed(0);random.seed(0);torch.set_num_threads(8)
    data=json.loads(args.manifest.read_text());tr=data['train'];dv=data['dev'];assert not ({r['source_video'] for r in tr}&{r['source_video'] for r in dv})
    if args.smoke:tr=tr[:4];dv=dv[:4];args.batch_size=4;args.epochs=1
    assert len(tr)>=args.batch_size and len(dv)>=2
    args.out.mkdir(parents=True,exist_ok=True)
    loader=DataLoader(Clips(tr),batch_size=args.batch_size,shuffle=True,generator=torch.Generator().manual_seed(0),num_workers=2,drop_last=True)
    dev=DataLoader(Clips(dv),batch_size=args.batch_size,num_workers=2)
    m=load_model();params=[p for p in m.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=1e-5,weight_decay=.01)
    report={'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),'learning_rate':1e-5,'weight_decay':.01,'clip_grad_norm':1.,'revision':REV,'seed':0,'trainable_parameters':sum(p.numel() for p in params),'trainable_names':[n for n,p in m.named_parameters() if p.requires_grad],'train_n':len(tr),'dev_n':len(dv),'batch_size':args.batch_size,'epochs':args.epochs,'baseline':evaluate(m,dev),'epochs_results':[]}
    print(json.dumps({'baseline':report['baseline'],'trainable_parameters':report['trainable_parameters']}),flush=True)
    best=report['baseline']['loss'];report['selected']='released_baseline'
    step=0;t0=time.time()
    for ep in range(args.epochs):
        # Keep frozen-layer dropout off; gradients still flow through selected layers.
        m.eval();losses=[]
        for x,txt in loader:
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                v=m.encode_vision(x.cuda());t=m.encode_text(m.tokenizer(list(txt)).cuda());loss=loss_fn(v,t,list(txt),m.temp)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward()
            if step==0:
                grads={key:sum(float(p.grad.float().square().sum()) for n,p in m.named_parameters() if n.startswith(prefix) and p.grad is not None)**.5 for key,prefix in [('video','vision_encoder.blocks.'),('text','text_encoder.transformer.')]}
                assert all(np.isfinite(g) and g>0 for g in grads.values()),grads
                report['first_step_tower_grad_norms']=grads;print(json.dumps({'tower_gradients':grads}),flush=True)
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
            with torch.no_grad():m.temp.clamp_(.01,.5)
            losses.append(float(loss));step+=1
            if step%10==0:print(json.dumps({'epoch':ep+1,'step':step,'loss':float(loss),'seconds':time.time()-t0}),flush=True)
        metrics=evaluate(m,dev);metrics.update(epoch=ep+1,training_loss=sum(losses)/len(losses));report['epochs_results'].append(metrics)
        state={n:p.detach().cpu() for n,p in m.named_parameters() if p.requires_grad}
        torch.save({'state_dict':state,'revision':REV,'epoch':ep+1,'dev':metrics},args.out/f'epoch-{ep+1}.pt')
        if metrics['loss']<best:best=metrics['loss'];report['selected']=f'epoch-{ep+1}.pt'
        (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(metrics),flush=True)
    print('COMPLETE',flush=True)
if __name__=='__main__':main()

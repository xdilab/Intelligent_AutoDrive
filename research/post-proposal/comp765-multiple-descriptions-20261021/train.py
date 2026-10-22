"""Fresh matched heads; original PhraseHead/focal reused, exact resumable batches."""
import argparse, math, pickle, time
import numpy as np
import torch
from common import *

def load_training(root):
    t=time.monotonic();d=pickle.load(INPUTS['train'].open('rb'))
    keys=[k for k in sorted(d['feats']) if len(d['feats'][k])]
    x=np.concatenate([d['feats'][k] for k in keys]).astype(np.float32)
    y=np.concatenate([d['targets'][k] for k in keys]).astype(np.float32)
    np.nan_to_num(x,copy=False,nan=0,posinf=0,neginf=0)
    assert len(x)==read(root/'preflight-assets.json')['training']['rows']
    assert np.isfinite(y).all();del d
    progress(root,'training-loaded',rows=len(x),seconds=time.monotonic()-t)
    return torch.from_numpy(x),torch.from_numpy(y)

def initial(p,seed):
    torch.manual_seed(seed)
    model=PhraseHead(p).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    return model,optimizer

def initial_hash(model):
    return hashlib.sha256(b''.join(k.encode()+v.detach().cpu().numpy().tobytes() for k,v in model.named_parameters())).hexdigest()

def fit(root,condition,seed,x,y,stop_after=None,dest_override=None):
    cfg=read(HERE/'protocol.json')['training'];p=torch.load(root/f'prototypes-{condition}.pt',weights_only=False)['embeds']
    model,opt=initial(p,seed);init=initial_hash(model);order=torch.Generator().manual_seed(seed)
    dest=dest_override or root/'runs'/f'{condition}-seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    sig=signature(root) if (root/'freeze.json').exists() else {'smoke':True}
    sig=dict(sig,condition=condition,seed=seed,prototype_sha256=sha(root/f'prototypes-{condition}.pt'),rows=len(x))
    epoch=offset=steps=0;epoch_loss=0.;epoch_batches=0;order_start=order.get_state();history=[]
    alpha=torch.load(INPUTS['alphas'],weights_only=True).float().cuda()
    last=dest/'last.pt'
    if last.exists():
        ck=torch.load(last,map_location='cpu',weights_only=False);assert ck['signature']==sig
        model.load_state_dict(ck['state']);opt.load_state_dict(ck['optimizer'])
        epoch=ck['epoch'];offset=ck['offset'];steps=ck['steps'];epoch_loss=ck['epoch_loss'];epoch_batches=ck['epoch_batches'];history=ck['history']
        order_start=ck['order_start'];order.set_state(order_start);torch.set_rng_state(ck['rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
        assert ck['initial_hash']==init
    param_count=sum(v.numel() for v in model.parameters())
    save(dest/'initialization.json',dict(seed=seed,condition=condition,trainable_sha256=init,parameters=param_count))
    def persist():
        checkpoint(last,dict(signature=sig,state={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),epoch=epoch,offset=offset,steps=steps,epoch_loss=epoch_loss,epoch_batches=epoch_batches,history=history,order_start=order_start,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),initial_hash=init,head='phrase',feat_dim=1024))
    start=time.monotonic();session_steps=0
    for ep in range(epoch,cfg['epochs']):
        epoch=ep;order.set_state(order_start);perm=torch.randperm(len(x),generator=order)
        permutation_hash=digest_tensor(perm)
        for i in range(offset,len(x),cfg['batch_size']):
            idx=perm[i:i+cfg['batch_size']];loss=focal(model(x[idx].cuda()),y[idx].cuda(),alpha)
            assert torch.isfinite(loss),'Nonfinite training loss'
            opt.zero_grad();loss.backward();opt.step();epoch_loss+=float(loss.detach());epoch_batches+=1;steps+=1;session_steps+=1;offset=i+len(idx)
            if steps%100==0 or (stop_after and session_steps>=stop_after):
                persist();progress(root,'train',run=dest.name,epoch=ep+1,offset=offset,total=len(x),steps=steps,loss=float(loss.detach()))
            if stop_after and session_steps>=stop_after:return model,opt
        history.append(dict(epoch=ep+1,loss=epoch_loss/epoch_batches,permutation_sha256=permutation_hash))
        epoch=ep+1;offset=0;epoch_loss=0.;epoch_batches=0;order_start=order.get_state();persist()
        save(dest/'epochs.json',history);progress(root,'train-epoch',run=dest.name,epoch=epoch,loss=history[-1]['loss'],elapsed=time.monotonic()-start)
    checkpoint(dest/'final.pt',dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},signature=sig,epoch=epoch,head='phrase',feat_dim=1024))
    save(dest/'training-complete.json',dict(passed=True,signature=sig,selected_epoch=epoch,checkpoint_sha256=sha(dest/'final.pt'),seconds=time.monotonic()-start))
    return model,opt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=DEFAULT_ROOT);a=ap.parse_args();setup();verify_freeze(a.root)
    x,y=load_training(a.root)
    for seed in [0,1,2]:
        for condition in ['single','multi']:
            dest=a.root/'runs'/f'{condition}-seed{seed}'
            if (dest/'training-complete.json').exists():
                done=read(dest/'training-complete.json');assert done['checkpoint_sha256']==sha(dest/'final.pt');continue
            m,o=fit(a.root,condition,seed,x,y);del m,o;torch.cuda.empty_cache()
        a0=read(a.root/'runs'/f'single-seed{seed}'/'initialization.json');b0=read(a.root/'runs'/f'multi-seed{seed}'/'initialization.json')
        assert a0['trainable_sha256']==b0['trainable_sha256'] and a0['parameters']==b0['parameters']
        aa=read(a.root/'runs'/f'single-seed{seed}'/'epochs.json');bb=read(a.root/'runs'/f'multi-seed{seed}'/'epochs.json')
        assert [r['permutation_sha256'] for r in aa]==[r['permutation_sha256'] for r in bb]
    progress(a.root,'all-training-complete')

if __name__=='__main__':main()

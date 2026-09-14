"""Matched repeated-seed study; consumes immutable caches and writes a new run directory."""
import argparse,hashlib,json,pickle,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class PhraseHead(nn.Module):
    def __init__(self,p,dim=1024):
        super().__init__();self.proj=nn.Linear(dim,p.shape[1]);self.register_buffer('P',F.normalize(p.float(),dim=-1));self.log_tau=nn.Parameter(torch.tensor(2.3));self.bias=nn.Parameter(torch.zeros(len(p)))
    def forward(self,x):return F.normalize(self.proj(x),dim=-1)@self.P.T*self.log_tau.exp()+self.bias

def focal(logits,y,alpha):
    prob=logits.sigmoid();pt=y*prob+(1-y)*(1-prob)
    return ((y*alpha+(1-y)*(1-alpha))*(1-pt).pow(2)*F.binary_cross_entropy_with_logits(logits,y,reduction='none')).mean()

def fit(make,x,y,idx,alpha,seed,epochs,bs,device):
    torch.manual_seed(seed);m=make().to(device);opt=torch.optim.Adam(m.parameters(),lr=.001)
    # Independent RNG makes batches identical across classifier variants.
    order=torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm=idx[torch.randperm(len(idx),generator=order).numpy()];loss_total=0;nb=0
        for start in range(0,len(perm),bs):
            b=perm[start:start+bs];loss=focal(m(x[b].to(device)),y[b].to(device),alpha.to(device))
            assert torch.isfinite(loss),'Nonfinite training loss'
            opt.zero_grad();loss.backward();opt.step();loss_total+=loss.item();nb+=1
        print(json.dumps({'epoch':ep+1,'loss':loss_total/max(nb,1)}),flush=True)
    return m.cpu().eval()

def save_model(model,path,meta):
    temp=path.with_suffix('.partial');torch.save(dict(meta,state=model.state_dict()),temp);temp.replace(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);ap.add_argument('--epochs',type=int,default=10);ap.add_argument('--batch-size',type=int,default=16384);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8);device='cuda' if torch.cuda.is_available() else 'cpu'
    if not a.smoke and device!='cuda':raise RuntimeError('Production study requires the allocated GPU')
    if a.smoke:
        rng=np.random.default_rng(9);keys=['v0_00001','v1_00001','v2_00001','v3_00001'];x=torch.tensor(rng.normal(size=(32,1024)),dtype=torch.float32);y=torch.tensor(rng.integers(0,2,size=(32,184)),dtype=torch.float32);vids=np.repeat(np.arange(4),8);p=torch.tensor(rng.normal(size=(184,512)),dtype=torch.float32);alpha=torch.full((184,),.5);a.epochs=1;a.batch_size=16;a.seeds=[0]
    else:
        print('Loading training cache',flush=True);d=pickle.load(open(a.data/'crop_feats_train.pkl','rb'));keys=[k for k in sorted(d['feats']) if len(d['feats'][k])]
        xa=np.concatenate([d['feats'][k] for k in keys]).astype(np.float32);ya=np.concatenate([d['targets'][k] for k in keys]).astype(np.float32)
        assert xa.shape[1]==1024 and ya.shape==(len(xa),184) and np.isfinite(ya).all()
        bad=int((~np.isfinite(xa)).sum());assert bad==0, 'Nonfinite paired features'
        vnames=sorted({k.rsplit('_',1)[0] for k in keys});vi={v:i for i,v in enumerate(vnames)};vids=np.concatenate([np.full(len(d['feats'][k]),vi[k.rsplit('_',1)[0]],np.int32) for k in keys]);del d
        x=torch.from_numpy(xa);y=torch.from_numpy(ya);p=torch.load(a.data/'phrase_embeds.pt',weights_only=False)['embeds'];alpha=torch.load(a.data/'flat_alphas.pt',weights_only=True).float()
        print(json.dumps({'rows':len(x),'videos':len(vnames),'nonfinite_elements_zeroed':bad}),flush=True)
    assert p.shape==(184,512) and alpha.shape==(184,)
    perm=torch.randperm(184,generator=torch.Generator().manual_seed(20260910));matrices={'phrase':p,'shuffled':p[perm]}
    meta={'seeds':a.seeds,'epochs':a.epochs,'batch_size':a.batch_size,'lr':.001,'loss':'focal gamma=2, fixed class alphas','device':device,'rows':len(x),'training_frames':len(keys),'training_keys_sha256':hashlib.sha256(('\n'.join(keys)+'\n').encode()).hexdigest(),'phrase_permutation':perm.tolist(),'fold_rule':'sorted video IDs alternate A/B; predict opposite fold','batch_rng':'independent seeded generator shared across classifier variants','smoke':a.smoke}
    (a.out/'training-plan.json').write_text(json.dumps(meta,indent=2)+'\n')
    full=np.arange(len(x));folds=[np.flatnonzero(vids%2==j) for j in [0,1]]
    for seed in a.seeds:
        dest=a.out/f'seed-{seed}';dest.mkdir(exist_ok=True);oof={}
        for kind in ['flat','phrase']:
            make=(lambda:nn.Linear(1024,184)) if kind=='flat' else (lambda kind=kind:PhraseHead(matrices[kind]))
            path=dest/f'head-{kind}.pt';om=dest/f'oof-{kind}.npy'
            if not path.exists():
                print(f'seed={seed} full head={kind}',flush=True);h=fit(make,x,y,full,alpha,seed,a.epochs,a.batch_size,device);save_model(h,path,dict(meta,seed=seed,head='flat' if kind=='flat' else 'phrase',control=kind,feat_dim=1024));del h
            if om.exists():oof[kind]=np.load(om,mmap_mode='r');continue
            z=np.zeros((len(x),184),np.float32)
            for j in [0,1]:
                print(f'seed={seed} OOF head={kind} train fold={j}',flush=True);h=fit(make,x,y,folds[j],alpha,seed,a.epochs,a.batch_size,device).to(device)
                with torch.no_grad():
                    for start in range(0,len(folds[1-j]),16384):
                        b=folds[1-j][start:start+16384];z[b]=h(x[b].to(device)).sigmoid().cpu().numpy()
                del h
            np.save(om,z);oof[kind]=z
        for variant,extra in [('stage5',None),('stage6','phrase')]:
            path=dest/f'{variant}.pt'
            if path.exists():continue
            parts=[oof['flat'][:,:49]]
            if extra:parts.append(oof[extra][:,49:184])
            parts.append(x.numpy());zin=torch.from_numpy(np.concatenate(parts,axis=1));dim=zin.shape[1]
            print(f'seed={seed} MLP={variant} dim={dim}',flush=True)
            m=fit(lambda:nn.Sequential(nn.Linear(dim,512),nn.ReLU(),nn.Linear(512,135)),zin,y[:,49:184],full,alpha[49:184],seed,a.epochs,a.batch_size,device)
            save_model(m,path,dict(meta,seed=seed,in_dim=dim,head_ckpt=str(dest/'head-flat.pt'),phrase_ckpt=str(dest/f'head-{extra}.pt') if extra else None,evidence=extra,oof=True));del zin,m
        print(f'COMPLETE seed={seed}',flush=True)
if __name__=='__main__':main()

import hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

VARIANTS=['head-flat','head-phrase','stage5','stage6','global-blend','class-gate','shuffled-global','shuffled-class-gate']
class PhraseHead(nn.Module):
    def __init__(self,p,dim=1024):
        super().__init__();self.proj=nn.Linear(dim,p.shape[1]);self.register_buffer('P',F.normalize(p.float(),dim=-1));self.log_tau=nn.Parameter(torch.tensor(2.3));self.bias=nn.Parameter(torch.zeros(len(p)))
    def forward(self,x):return F.normalize(self.proj(x),dim=-1)@self.P.T*self.log_tau.exp()+self.bias

def partitions(videos):
    ordered=sorted(videos,key=lambda v:hashlib.sha256(('class-gate-v1:'+v).encode()).hexdigest());n=len(ordered);a=int(.7*n);b=a+int(.15*n)
    assert a>=2 and b>a and n>b
    return {'expert':ordered[:a],'gate':ordered[a:b],'development':ordered[b:]}

def make_head(kind,p):return nn.Linear(1024,184) if kind=='flat' else PhraseHead(p)
def make_mlp(dim):return nn.Sequential(nn.Linear(dim,512),nn.ReLU(),nn.Linear(512,135))
def load_head(path):
    ck=torch.load(path,map_location='cpu',weights_only=False);m=make_head(ck['kind'],ck['state'].get('P'));m.load_state_dict(ck['state']);return m.eval()
def load_mlp(path):
    ck=torch.load(path,map_location='cpu',weights_only=False);m=make_mlp(ck['in_dim']);m.load_state_dict(ck['state']);return m.eval()
def save_model(m,path,**meta):
    path=Path(path);temp=path.with_suffix('.partial');torch.save(dict(meta,state=m.cpu().state_dict()),temp);temp.replace(path)

def focal(logits,y,alpha):
    p=logits.sigmoid();pt=y*p+(1-y)*(1-p)
    return ((y*alpha+(1-y)*(1-alpha))*(1-pt).pow(2)*F.binary_cross_entropy_with_logits(logits,y,reduction='none')).mean()

def fit_expert(make,x,y,rows,alpha,seed,epochs,bs,device):
    torch.manual_seed(seed);m=make().to(device);opt=torch.optim.Adam(m.parameters(),lr=.001);rng=torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        order=rows[torch.randperm(len(rows),generator=rng).numpy()];total=0;n=0
        for start in range(0,len(order),bs):
            b=order[start:start+bs];xb=torch.from_numpy(np.array(x[b],dtype=np.float32)).to(device);yb=torch.from_numpy(np.array(y[b],dtype=np.float32)).to(device);loss=focal(m(xb),yb,alpha.to(device));assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();opt.step();total+=loss.item()*len(b);n+=len(b)
        print(json.dumps({'epoch':ep+1,'focal_loss':total/n}),flush=True)
    return m.cpu().eval()

@torch.no_grad()
def predict(m,x,rows,device,bs=16384):
    m=m.to(device);out=[]
    for start in range(0,len(rows),bs):
        b=rows[start:start+bs];out.append(m(torch.from_numpy(np.array(x[b],dtype=np.float32)).to(device)).sigmoid().cpu().numpy())
    m.cpu();return np.concatenate(out)

class Gate(nn.Module):
    def __init__(self,anchor=0.,support=None,global_gate=False):
        super().__init__();self.global_gate=global_gate;self.a=nn.Parameter(torch.full((1 if global_gate else 135,),float(anchor)));self.register_buffer('anchor',torch.tensor(float(anchor)));self.register_buffer('support',torch.ones(135,dtype=torch.bool) if support is None else torch.as_tensor(support,dtype=torch.bool))
    def weights(self):return self.a.sigmoid().expand(135) if self.global_gate else torch.where(self.support,self.a,self.anchor).sigmoid()
    def forward(self,p5,pl):
        g=self.weights();return (1-g)*p5+g*pl
    def penalty(self):return ((torch.where(self.support,self.a,self.anchor)-self.anchor)**2).mean() if not self.global_gate else self.a.sum()*0

def fit_gate(p5,pl,y,seed,epochs,bs,device,anchor=0.,lam=0.,global_gate=False):
    support=y.sum(0)>0;m=Gate(anchor,support,global_gate).to(device);opt=torch.optim.Adam(m.parameters(),lr=.05);rng=np.random.default_rng(seed)
    for ep in range(epochs):
        order=rng.permutation(len(y));total=0
        for start in range(0,len(y),bs):
            b=order[start:start+bs];a=torch.from_numpy(np.asarray(p5[b],np.float32)).to(device);l=torch.from_numpy(np.asarray(pl[b],np.float32)).to(device);t=torch.from_numpy(np.asarray(y[b],np.float32)).to(device);p=m(a,l).clamp(1e-7,1-1e-7);loss=F.binary_cross_entropy(p,t)+lam*m.penalty();assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();opt.step()
            with torch.no_grad():m.a.clamp_(-8,8)
            total+=loss.item()*len(b)
        if ep==0 or ep==epochs-1:print(json.dumps({'gate_global':global_gate,'lambda':lam,'epoch':ep+1,'loss':total/len(y)}),flush=True)
    return m.cpu().eval()

def binary_ap(y,p):
    order=np.argsort(-p);yy=y[order]>0;n=int(yy.sum())
    if not n:return 0.
    tp=np.cumsum(yy);prec=tp/np.arange(1,len(yy)+1);env=np.maximum.accumulate(prec[::-1])[::-1]
    return float(env[yy].sum()/n*100)
def crop_ap(y,p):return [binary_ap(y[:,c],p[:,c]) for c in range(y.shape[1])]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

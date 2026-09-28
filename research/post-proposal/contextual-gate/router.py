"""Actor/class gates with shared primitive reliability; no labels in forward."""
import torch
from torch import nn
from torch.nn import functional as F

def composition_map(labels):
 out=[]
 for kind in ['duplex','triplet']:
  for label in labels[kind]:
   parts=label.split('-');assert len(parts)==(2 if kind=='duplex' else 3)
   ids=[1+labels['agent'].index(parts[0]),11+labels['action'].index(parts[1])]
   ids.append(33+labels['loc'].index(parts[2]) if len(parts)==3 else 0);out.append(ids)
 assert len(out)==135
 return torch.tensor(out,dtype=torch.long)

class Router(nn.Module):
 def __init__(self,mapping,support,anchor,kind='factorized'):
  super().__init__();assert kind in ['class','generic','factorized'];self.kind=kind
  self.register_buffer('mapping',mapping.long());self.register_buffer('shrink',support.float()/(support.float()+5));self.register_buffer('anchor',torch.tensor(float(anchor)).clamp(1e-4,1-1e-4).logit())
  if kind=='class':self.offset=nn.Parameter(torch.zeros(135))
  elif kind=='generic':
   self.embedding=nn.Embedding(135,8);self.net=nn.Sequential(nn.Linear(40,30),nn.GELU(),nn.Linear(30,1));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
  else:
   self.embedding=nn.Embedding(49,8,padding_idx=0)
   self.primitive=nn.Sequential(nn.Linear(28,32),nn.GELU(),nn.Linear(32,1))
   self.interaction=nn.Sequential(nn.Linear(28,32),nn.GELU(),nn.Linear(32,1))
   for net in [self.primitive,self.interaction]:nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias)
 def forward(self,a,b,actor,classes):
  a,b,actor=[v.detach().float() for v in (a,b,actor)];classes=classes.long();rows=torch.arange(len(a),device=a.device);pa=a[rows,classes+49];pb=b[rows,classes+49]
  own=torch.stack([pa,pb,pa-pb,(pa-pb).abs()],-1)
  if self.kind=='class':delta=self.offset[classes]
  elif self.kind=='generic':
   ids=self.mapping[classes];x=a.gather(1,ids);y=b.gather(1,ids);ev=torch.stack([x,y,x-y,(x-y).abs()],-1)*(ids!=0)[:,:,None]
   delta=self.net(torch.cat([actor,own,ev.flatten(1),self.embedding(classes)],-1)).squeeze(-1)
  else:
   ids=self.mapping[classes];valid=(ids!=0).float();ea=self.embedding(ids);x=a.gather(1,ids);y=b.gather(1,ids)
   ev=torch.stack([x,y,x-y,(x-y).abs()],-1);pr=self.primitive(torch.cat([actor[:,None].expand(-1,3,-1),ev,ea],-1)).squeeze(-1)
   count=valid.sum(-1).clamp_min(1);shared=(pr*valid).sum(-1)/count;emb=(ea*valid[:,:,None]).sum(1)/count[:,None]
   delta=shared+self.interaction(torch.cat([actor,own,emb],-1)).squeeze(-1)
  g=(self.anchor+self.shrink[classes]*delta).sigmoid();return (1-g)*pa+g*pb,g

def actor_features(crop,context,boxes):
 # Deterministic channel pooling supplies16 context/crop descriptors without a new encoder pass.
 a=torch.cat([crop.float().reshape(-1,8,128).mean(-1),context.float().reshape(-1,8,128).mean(-1)],-1)
 return F.layer_norm(a,(16,)).detach()

def ranking_loss(prob,groups,pairs):
 z=prob.clamp(1e-5,1-1e-5).logit().reshape(groups,2,pairs)
 return F.softplus(-(z[:,0,:,None]-z[:,1,None,:])).mean()

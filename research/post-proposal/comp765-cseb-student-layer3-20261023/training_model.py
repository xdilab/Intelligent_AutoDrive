"""Frame-independent frozen prefix and optional trainable layer3; fixed BN stats."""
from common import *
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50
from torchvision.ops import roi_align

def encoder():
 with torch.random.fork_rng(devices=[]):
  m=resnet50(weights=None);m.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True))
 return m
class Prefix(nn.Module):
 def __init__(self):
  super().__init__();self.body=nn.Sequential(*list(encoder().children())[:6]);self.requires_grad_(False).eval()
 @torch.no_grad()
 def forward(self,image):
  x=torch.as_tensor(image,device='cuda').permute(2,0,1)[None].float()/255;x=F.interpolate(x,(448,672),mode='bilinear',align_corners=False)
  mean=x.new_tensor([.485,.456,.406])[None,:,None,None];std=x.new_tensor([.229,.224,.225])[None,:,None,None]
  with torch.autocast('cuda',dtype=torch.bfloat16):return self.body((x-mean)/std)

def features(stage,x,rows):
 with torch.autocast('cuda',dtype=torch.bfloat16):f=stage(x)
 f=f.float();rois=[]
 for i,r in enumerate(rows):
  b=r['boxes'];rois.append(torch.cat((b.new_full((len(b),1),i),b*b.new_tensor([42,28,42,28])),1))
 roi=roi_align(f,torch.cat(rois),output_size=3,aligned=True).mean((-1,-2)).half().float();scene=f.mean((-1,-2)).half().float();parts=roi.split([len(r['boxes']) for r in rows])
 return [dict(r,roi=a,scene=s) for r,a,s in zip(rows,parts,scene)]
class TrainModel(nn.Module):
 def __init__(self,tuned):
  super().__init__();self.stage=encoder().layer3;self.stage.requires_grad_(tuned);self.head=Student();self.tuned=tuned;self.train()
 def train(self,mode=True):
  super().train(mode)
  for m in self.stage.modules():
   if isinstance(m,nn.modules.batchnorm._BatchNorm):m.eval()
  return self
 def rows(self,seq):return features(self.stage,seq['prefix'],seq['rows']) if self.tuned else seq['pooled']

def optimizer(model):
 groups=[dict(params=model.head.parameters(),lr=.001)]
 if model.tuned:groups.append(dict(params=model.stage.parameters(),lr=.00001))
 return torch.optim.AdamW(groups,weight_decay=.01)

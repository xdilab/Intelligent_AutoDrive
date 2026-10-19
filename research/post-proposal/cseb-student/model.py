"""Small causal student: shared frozen spatial features and scene GRU memory."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50
from torchvision.ops import roi_align

class SharedBackbone(nn.Module):
    def __init__(self, weights):
        super().__init__()
        model=resnet50(weights=None)
        model.load_state_dict(torch.load(Path(weights),map_location='cpu',weights_only=True),strict=True)
        self.body=nn.Sequential(*list(model.children())[:7]) # through layer3, stride16, 1024 channels
        self.requires_grad_(False).eval()
    @torch.inference_mode()
    def forward(self, image, boxes):
        x=torch.as_tensor(image,device='cuda').permute(2,0,1)[None].float()/255
        x=F.interpolate(x,size=(448,672),mode='bilinear',align_corners=False)
        mean=x.new_tensor([.485,.456,.406])[None,:,None,None]
        std=x.new_tensor([.229,.224,.225])[None,:,None,None]
        with torch.autocast('cuda',dtype=torch.bfloat16): fmap=self.body((x-mean)/std)
        fmap=fmap.float();b=torch.as_tensor(boxes,device='cuda',dtype=torch.float32).reshape(-1,4)
        rois=torch.cat((b.new_zeros((len(b),1)),b*b.new_tensor([42,28,42,28])),1)
        roi=roi_align(fmap,rois,output_size=3,aligned=True).mean((-1,-2))
        return roi, fmap.mean((-1,-2))[0]

class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.scene=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,256),nn.GELU())
        self.memory=nn.GRUCell(256,256)
        self.actor=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,256),nn.GELU())
        self.agent=nn.Embedding(10,16)
        self.head=nn.Sequential(nn.Linear(256+256+16+8,256),nn.GELU(),nn.Linear(256,184))
    def forward(self,roi,scene,boxes,agents,state=None):
        if state is None:state=scene.new_zeros((1,256))
        state=self.memory(self.scene(scene[None]),state)
        geometry=torch.cat((boxes,(boxes[:,:2]+boxes[:,2:])/2,boxes[:,2:]-boxes[:,:2]),1)
        x=torch.cat((self.actor(roi),state.expand(len(roi),-1),self.agent(agents),geometry),1)
        return self.head(x),state

def objective(logits,labels,teacher,kd_weight):
    # Multi-label Bernoulli targets; no softmax competition between valid labels.
    supervised=F.binary_cross_entropy_with_logits(logits[:,1:],labels[:,1:])
    kd=F.binary_cross_entropy_with_logits(logits[:,1:],teacher[:,1:])
    return supervised+kd_weight*kd

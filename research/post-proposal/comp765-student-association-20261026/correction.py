"""One-pass, stateless other-actor correction. No teacher dependency."""
import torch
from torchvision.ops import nms

def neighbors(boxes,context):
 lo=torch.maximum(boxes[:,None,:2],context[None,:,:2]);hi=torch.minimum(boxes[:,None,2:],context[None,:,2:])
 inter=(hi-lo).clamp_min(0).prod(-1);a=(boxes[:,2:]-boxes[:,:2]).clamp_min(0).prod(-1)[:,None];b=(context[:,2:]-context[:,:2]).clamp_min(0).prod(-1)[None,:]
 iou=inter/(a+b-inter).clamp_min(1e-12);iom=inter/torch.minimum(a,b).clamp_min(1e-12)
 distance=torch.linalg.vector_norm((boxes[:,:2]+boxes[:,2:])[:,None,:]/2-(context[:,:2]+context[:,2:])[None,:,:]/2,dim=-1)
 return (distance<=.5)&(iou<.3)&(iom<.5)

class Correction:
 def __init__(self,rules,device='cpu'):
  self.weights=torch.as_tensor(rules['weights'],dtype=torch.float32,device=device)
  self.mask=(self.weights!=0).float()
  self.event_agents=torch.as_tensor(rules['event_agents'],device=device,dtype=torch.long)
  self.marginal=torch.as_tensor(rules['context_marginal'],dtype=torch.float32,device=device)
  self.static=(self.marginal@self.weights)/(1+self.marginal@self.mask)
 def delta(self,logits,boxes,agents,q,mode):
  if not len(boxes):return logits.new_empty((0,135))
  target=(agents[:,None]==self.event_agents[None,:])
  if mode=='frequency':return self.static[agents]*target
  assert mode=='association'
  eligible=torch.where(q>=.25)[0]
  if not len(eligible):return logits.new_zeros((len(boxes),135))
  selected=eligible[nms(boxes[eligible],q[eligible],.5)[:30]]
  support=logits[selected,49:].sigmoid()*q[selected,None]
  support=support*(agents[selected,None]==self.event_agents[None,:])
  adj=neighbors(boxes,boxes[selected]);adj[torch.arange(len(boxes),device=boxes.device)[:,None]==selected[None,:]]=False
  evidence=(support[None,:,:]*adj[:,:,None]).amax(1)
  return ((evidence@self.weights)/(1+evidence@self.mask))*target
 def scores(self,logits,q,delta=None,strength=0):
  prob=logits.sigmoid()
  if strength and delta is not None:prob[:,49:]=(logits[:,49:]+strength*delta).sigmoid()
  scores=prob*q[:,None];scores[:,0]=q
  return scores

"""One-pass full-video single rules plus bounded residual two-event conjunctions."""
import torch
from torchvision.ops import nms
from single_correction import Correction as Single,neighbors
class Correction(Single):
 def __init__(self,rules,device='cpu'):
  super().__init__(rules,device)
  self.pairs=torch.as_tensor(rules['pair_antecedents'],device=device,dtype=torch.long).reshape(-1,2)
  self.pw=torch.as_tensor(rules['pair_weights'],device=device,dtype=torch.float32).reshape(-1,135);self.pm=(self.pw!=0).float()
  marginal=torch.as_tensor(rules['pair_context_marginal'],device=device,dtype=torch.float32).reshape(10,-1)
  self.pstatic=(marginal@self.pw)/(1+marginal@self.pm)
 def evidence(self,logits,boxes,agents,q):
  eligible=torch.where(q>=.25)[0]
  if not len(boxes) or not len(eligible):return logits.new_zeros((len(boxes),135))
  selected=eligible[nms(boxes[eligible],q[eligible],.5)[:30]];support=logits[selected,49:].sigmoid()*q[selected,None];support*=agents[selected,None]==self.event_agents[None,:]
  adj=neighbors(boxes,boxes[selected]);adj[torch.arange(len(boxes),device=boxes.device)[:,None]==selected[None,:]]=False
  return (support[None]*adj[:,:,None]).amax(1)
 def components(self,logits,boxes,agents,q):
  target=agents[:,None]==self.event_agents[None,:];e=self.evidence(logits,boxes,agents,q);single=((e@self.weights)/(1+e@self.mask))*target;pe=torch.minimum(e[:,self.pairs[:,0]],e[:,self.pairs[:,1]]);pair=((pe@self.pw)/(1+pe@self.pm))*target
  return dict(full_single=single,full_frequency=self.static[agents]*target,full_pair=(single+pair).clamp(-1,1),full_pair_frequency=((self.static+self.pstatic)[agents]*target).clamp(-1,1))
 def delta(self,logits,boxes,agents,q,mode='full_pair'):
  return self.components(logits,boxes,agents,q)[mode]

"""FRCB-inspired DCB: past-batch positive residual history, detached weights."""
import torch
import torch.nn.functional as F
from model import objective as focal_objective
class DCBState(torch.nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer('S',torch.zeros(184,dtype=torch.float64));self.register_buffer('N',torch.zeros(184,dtype=torch.float64))
    def weights(self,z,y):
        assert z.shape==y.shape and z.shape[-1]==184
        with torch.no_grad():
            wk=torch.where(y>0,1-z.float().sigmoid(),torch.zeros_like(z.float()))
            wa=torch.where(self.N>0,self.S/self.N.clamp_min(1),0.)
            return .5+wk+wa.float(),wk
    def classification(self,z,y,update=True):
        b,wk=self.weights(z,y);loss=(b*F.binary_cross_entropy_with_logits(z.float(),y,reduction='none')).mean()
        if update:
            with torch.no_grad():self.S.add_(wk.sum(0,dtype=torch.float64));self.N.add_(y.sum(0,dtype=torch.float64))
        return loss
    def report(self):
        return {'S':self.S.cpu().tolist(),'N':self.N.cpu().tolist(),'Wa':torch.where(self.N>0,self.S/self.N.clamp_min(1),0.).cpu().tolist()}
def objective(outputs,targets,alpha,weight,state,loss_kind='dcb',update=True):
    if loss_kind=='focal':return focal_objective(outputs,targets,alpha,weight)
    cls=state.classification(outputs['logits'],targets,update=update)
    sim=outputs['contrastive_logits'];assert targets.shape==sim.shape and sim.shape[-1]==184
    valid=targets.sum(-1)>0
    aux=-(targets[valid]*sim[valid].log_softmax(-1)).sum(-1).div(targets[valid].sum(-1)).mean() if valid.any() else sim.sum()*0
    return cls+weight*aux,cls,aux

import torch
from train import loss_fn
v=torch.eye(4,requires_grad=True);t=torch.eye(4,requires_grad=True);temp=torch.tensor(.1,requires_grad=True)
texts=['a','b','c','d'];good=loss_fn(v,t,texts,temp);bad=loss_fn(v,t.flip(0),texts,temp)
assert good<bad
bad.backward();assert v.grad.abs().sum()>0 and t.grad.abs().sum()>0
same=loss_fn(v.detach(),t.detach(),['same']*4,temp.detach());assert abs(float(same))<1e-6
assert torch.isfinite(good)
print('PASS: correct pairing preferred; gradients reach both embeddings; duplicate captions treated as positives.')

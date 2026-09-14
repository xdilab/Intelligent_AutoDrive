import torch
from experiment import multi_positive_loss,crop_ap
z=torch.tensor([[1.,2.,3.],[3.,2.,1.]],requires_grad=True);y=torch.tensor([[1.,0.,1.],[0.,0.,0.]])
l=multi_positive_loss(z,y);expected=-(torch.log_softmax(z[0],0)[0]+torch.log_softmax(z[0],0)[2])/2;assert torch.allclose(l,expected);l.backward();assert torch.equal(z.grad[1],torch.zeros(3))
assert multi_positive_loss(z,torch.zeros_like(y))==0
perm=torch.tensor([2,0,1]);assert torch.allclose(multi_positive_loss(z[:,perm],y[:,perm]),l)
assert torch.allclose(multi_positive_loss(z.repeat(2,1),y.repeat(2,1)),l)
print('PASS: multiple positives, no-positive masking, prototype order, duplicate crop invariance')

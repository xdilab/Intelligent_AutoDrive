import unittest,torch
from router import Router,ranking_loss,composition_map
class Contracts(unittest.TestCase):
 def test_shared_and_frozen(self):
  torch.manual_seed(0);mapping=torch.tensor([[1,11,33]]*135);m=Router(mapping,torch.ones(135)*5,.5);a=torch.rand(8,184,requires_grad=True);b=torch.rand(8,184,requires_grad=True);actor=torch.randn(8,16,requires_grad=True);c=torch.arange(8)
  p,g=m(a,b,actor,c);torch.testing.assert_close(g,torch.ones(8)*.5);p.sum().backward();self.assertIsNone(a.grad);self.assertIsNone(b.grad);self.assertIsNone(actor.grad);self.assertGreater(m.primitive[-1].weight.grad.abs().sum(),0)
 def test_zero_support_fallback(self):
  m=Router(torch.tensor([[1,11,0]]*135),torch.zeros(135),.25,'class');m.offset.data.fill_(100);p,g=m(torch.rand(4,184),torch.rand(4,184),torch.rand(4,16),torch.arange(4));torch.testing.assert_close(g,torch.ones(4)*.25)
 def test_multilabel_independent_and_convex(self):
  for kind in ['class','generic','factorized']:
   m=Router(torch.tensor([[1,11,0]]*135),torch.ones(135),.5,kind);a=torch.rand(4,184);b=torch.rand(4,184);c=torch.arange(4);p,g=m(a,b,torch.rand(4,16),c);rows=torch.arange(4)
   self.assertTrue(torch.all(p>=torch.minimum(a[rows,c+49],b[rows,c+49])));self.assertTrue(torch.all(p<=torch.maximum(a[rows,c+49],b[rows,c+49])))
 def test_rank_direction(self):
  self.assertLess(ranking_loss(torch.tensor([.9,.8,.1,.2]),1,2),ranking_loss(torch.tensor([.1,.2,.9,.8]),1,2))
if __name__=='__main__':unittest.main()

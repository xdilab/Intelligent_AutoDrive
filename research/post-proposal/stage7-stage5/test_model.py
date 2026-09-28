import unittest
import torch
from torch import nn
from model import Stage7,objective

class Stage7Contracts(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(2);torch.manual_seed(12)
  self.m=Stage7(torch.randn(184,32),nn.Linear(64,184),nn.Sequential(nn.Linear(113,32),nn.ReLU(),nn.Linear(32,135)),visual_dim=64,dim=32)
  self.x=[torch.randn(5,64,requires_grad=True),torch.randn(5,64,requires_grad=True),torch.randn(5,16,64,requires_grad=True),torch.tensor([[.1,.2,.7,.8]]*5)]
  self.y=torch.zeros(5,184);self.y[:,[0,1,11,33,49,98]]=1;self.alpha=torch.full((184,),.5)
 def test_initial_parity_and_actual_stage5_wiring(self):
  out=self.m(*self.x);raw=self.m.flat(self.x[0]);expected=torch.cat([raw[:,:49],self.m.comp(torch.cat([raw[:,:49].sigmoid(),self.x[0]],1))],1)
  torch.testing.assert_close(out['logits'],expected,atol=0,rtol=0)
  torch.testing.assert_close(out['refined_features'],self.x[0],atol=0,rtol=0)
  self.assertFalse(any(n.startswith('context.classifier.') for n,_ in self.m.named_parameters()))
 def test_classification_crosses_frozen_heads_and_zero_bridge(self):
  out=self.m(*self.x);_,loss,_=objective(out,self.y,self.alpha);loss.backward()
  self.assertGreater(self.m.bridge[-1].weight.grad.abs().sum().item(),0)
  self.assertEqual(self.m.bridge[0].weight.grad.abs().sum().item(),0)
  self.assertTrue(all(p.grad is None for p in self.m.flat.parameters()))
  self.assertTrue(all(p.grad is None for p in self.m.comp.parameters()))
  self.assertTrue(all(v.grad is None for v in self.x[:3]))
  opt=torch.optim.AdamW([p for p in self.m.parameters() if p.requires_grad],lr=.001)
  frozen={n:p.clone() for n,p in self.m.named_parameters() if not p.requires_grad}
  opt.step();opt.zero_grad();_,cls,_=objective(self.m(*self.x),self.y,self.alpha);cls.backward()
  self.assertGreater(self.m.context.visual_fusion[-1].weight.grad.abs().sum().item(),0)
  self.assertGreater(self.m.context.text_adapter.net[-1].weight.grad.abs().sum().item(),0)
  for n,p in self.m.named_parameters():
   if n in frozen:torch.testing.assert_close(p,frozen[n],atol=0,rtol=0)
 def test_contrastive_both_adapters_before_language(self):
  _,_,aux=objective(self.m(*self.x),self.y,self.alpha);aux.backward()
  for module in [self.m.context.visual_fusion,self.m.context.text_adapter.net]:self.assertGreater(module[-1].weight.grad.abs().sum().item(),0)
  self.assertTrue(all(p.grad is None for p in self.m.context.language_attention.parameters()))
  self.assertTrue(all(p.grad is None for p in self.m.bridge.parameters()))
  self.assertFalse(self.m.context.phrase_bank.requires_grad)
if __name__=='__main__':unittest.main()

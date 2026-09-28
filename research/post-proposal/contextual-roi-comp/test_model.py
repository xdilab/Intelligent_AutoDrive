import importlib.util,unittest,tempfile
from pathlib import Path
import torch
from model import ContextualRoIHead,objective
from train_cached import atomic_torch

def source_module(name,folder):
 p=Path('/data/repos/ROAD_Reason/research/post-proposal')/folder/'model.py'
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
OLD=source_module('flat_original','contextual-roi');CORRECTED=source_module('flat_corrected','contextual-roi-all184')
class CompositionContracts(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(2);torch.manual_seed(99);self.bank=torch.randn(184,32)
  self.inputs=[torch.randn(5,64,requires_grad=True),torch.randn(5,64,requires_grad=True),torch.randn(5,16,64,requires_grad=True),torch.tensor([[.1,.2,.7,.8]]*5)]
  self.y=torch.zeros(5,184);self.y[:,[0,1,11,33,49,98,183]]=1;self.alpha=torch.full((184,),.5)
 def build(self,cls,seed):
  torch.manual_seed(seed);return cls(self.bank,visual_dim=64,dim=32,fusion='attention',heads=4)
 def test_seed_matched_backbones_and_new_arms(self):
  for seed in range(3):
   new=self.build(ContextualRoIHead,seed);other=self.build(ContextualRoIHead,seed)
   for n,v in new.state_dict().items():torch.testing.assert_close(v,other.state_dict()[n],atol=0,rtol=0)
   for module in [OLD,CORRECTED]:
    old=self.build(module.ContextualRoIHead,seed)
    for n,v in old.state_dict().items():
     if not n.startswith('classifier.'):torch.testing.assert_close(v,new.state_dict()[n],atol=0,rtol=0)
 def test_classification_only_parent_equivalence(self):
  old=self.build(OLD.ContextualRoIHead,0);new=self.build(CORRECTED.ContextualRoIHead,0)
  a=old(*self.inputs);b=new(*self.inputs);torch.testing.assert_close(a['logits'],b['logits'],atol=0,rtol=0)
  la=OLD.objective(a,self.y,self.alpha,0)[0];lb=CORRECTED.objective(b,self.y,self.alpha,0)[0];torch.testing.assert_close(la,lb,atol=0,rtol=0);la.backward();lb.backward()
  for (_,p),(_,q) in zip(old.named_parameters(),new.named_parameters()):
   if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=1e-7,rtol=1e-6)
 def test_composition_order_and_joint_gradient(self):
  m=self.build(ContextualRoIHead,0);seen={}
  m.flat.register_forward_hook(lambda module,inputs,out:seen.update(raw=out))
  m.comp.register_forward_hook(lambda module,inputs,out:seen.update(compositions=out,comp_input=inputs[0]))
  out=m(*self.inputs);self.assertEqual(out['logits'].shape,(5,184))
  torch.testing.assert_close(out['logits'][:,:49],seen['raw']);torch.testing.assert_close(out['logits'][:,49:],seen['compositions']);torch.testing.assert_close(seen['comp_input'][:,:49],seen['raw'].sigmoid())
  out['logits'][:,49:].square().mean().backward()
  self.assertGreater(m.flat[-1].weight.grad.abs().sum().item(),0)
  self.assertGreater(m.context_projection.weight.grad.abs().sum().item(),0)
  self.assertTrue(all(x.grad is None for x in self.inputs[:3]))
 def test_contrastive_trains_both_adapters_not_readout(self):
  m=self.build(ContextualRoIHead,0);aux=objective(m(*self.inputs),self.y,self.alpha,.001)[2];aux.backward()
  for mod in [m.visual_fusion,m.text_adapter.net]:self.assertGreater(mod[-1].weight.grad.abs().sum().item(),0)
  self.assertTrue(all(p.grad is None for p in list(m.flat.parameters())+list(m.comp.parameters())+list(m.language_attention.parameters())))
  self.assertFalse(m.phrase_bank.requires_grad)
 def test_atomic_optimizer_roundtrip(self):
  m=self.build(ContextualRoIHead,0);opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
  def step(model,optim):
   optim.zero_grad();loss=objective(model(*self.inputs),self.y,self.alpha,.001)[0];loss.backward();optim.step()
  step(m,opt)
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'resume.pt';atomic_torch({'model':m.state_dict(),'optimizer':opt.state_dict(),'signature':{'head':'flat49+comp135','comp_gradient':'joint'}},p)
   ck=torch.load(p,weights_only=False);n=self.build(ContextualRoIHead,2);n.load_state_dict(ck['model']);o=torch.optim.AdamW(n.parameters(),lr=1e-4);o.load_state_dict(ck['optimizer']);step(m,opt);step(n,o)
   for k,v in m.state_dict().items():torch.testing.assert_close(v,n.state_dict()[k],atol=0,rtol=0)
if __name__=='__main__':unittest.main()

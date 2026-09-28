import unittest,importlib.util,copy,tempfile
from pathlib import Path
import torch
from model import ContextualRoIHead,objective
from compare import stats
s=importlib.util.spec_from_file_location('original',Path('/data/repos/ROAD_Reason/research/post-proposal/contextual-roi/model.py'));original=importlib.util.module_from_spec(s);s.loader.exec_module(original)
class Controls(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(2);torch.manual_seed(44);self.bank=torch.randn(184,32);self.x=[torch.randn(4,64),torch.randn(4,64),torch.randn(4,16,64),torch.tensor([[.1,.2,.6,.8]]*4)];self.y=torch.randint(0,2,(4,184)).float();self.alpha=torch.full((184,),.5)
 def build(self,mode,seed=0):
  torch.manual_seed(seed);return ContextualRoIHead(self.bank,64,32,'attention',4,mode,seed)
 def test_init_and_rng(self):
  for seed in range(3):
   real=self.build('real',seed);rng=torch.get_rng_state()
   for mode in ['bypass','randbank']:
    m=self.build(mode,seed);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
    for name,value in real.state_dict().items():
     if name!='phrase_bank':self.assertTrue(torch.equal(value,m.state_dict()[name]),name)
    if mode=='randbank':torch.testing.assert_close(m.phrase_bank.norm(dim=1),torch.ones(184));self.assertFalse(torch.equal(m.phrase_bank,real.phrase_bank))
 def test_original_lambda0_parity(self):
  m=self.build('real');torch.manual_seed(0);old=original.ContextualRoIHead(self.bank,64,32,'attention',4)
  a=m(*self.x);b=old(*self.x);torch.testing.assert_close(a['logits'],b['logits'],atol=0,rtol=0);loss=objective(a,self.y,self.alpha,0)[0];oldloss=original.objective(b,self.y,self.alpha,0)[0];loss.backward();oldloss.backward()
  for (_,p),(_,q) in zip(m.named_parameters(),old.named_parameters()):
   if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=1e-7,rtol=1e-6)
 def test_bypass_invariance_gradients_freezing(self):
  a=self.build('bypass');b=copy.deepcopy(a);b.phrase_bank.copy_(torch.nn.functional.normalize(torch.randn_like(b.phrase_bank),dim=-1));oa=a(*self.x);ob=b(*self.x);torch.testing.assert_close(oa['logits'],ob['logits'],atol=0,rtol=0)
  loss,cls,aux=objective(oa,self.y,self.alpha,0);self.assertTrue(torch.equal(loss,cls));loss.backward();objective(ob,self.y,self.alpha,0)[0].backward()
  for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
   if name.startswith(('text_adapter.','language_attention.','language_scale')):self.assertFalse(p.requires_grad);self.assertIsNone(p.grad)
   elif p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)
  frozen={k:v.clone() for k,v in a.state_dict().items() if k.startswith(('text_adapter.','language_attention.','language_scale'))};opt=torch.optim.AdamW([p for p in a.parameters() if p.requires_grad],lr=.001);opt.step()
  for k,v in frozen.items():self.assertTrue(torch.equal(v,a.state_dict()[k]))
 def test_random_active_and_checkpoint(self):
  m=self.build('randbank');opt=torch.optim.AdamW(m.parameters(),lr=.001)
  def step(m,o):
   o.zero_grad();objective(m(*self.x),self.y,self.alpha,0)[0].backward();o.step()
  step(m,opt);self.assertGreater(m.text_adapter.net[-1].weight.grad.abs().sum().item(),0)
  with tempfile.TemporaryDirectory() as td:
   path=Path(td)/'state.pt';torch.save({'model':m.state_dict(),'opt':opt.state_dict()},path);ck=torch.load(path,weights_only=False);n=self.build('randbank',2);n.load_state_dict(ck['model']);o=torch.optim.AdamW(n.parameters(),lr=.001);o.load_state_dict(ck['opt']);step(m,opt);step(n,o)
   for k,v in m.state_dict().items():torch.testing.assert_close(v,n.state_dict()[k],atol=0,rtol=0)
 def test_equivalence_not_nonsignificance(self):
  self.assertTrue(stats([.01,.02,.03],.30)['equivalent']);self.assertFalse(stats([-1,0,1],.30)['equivalent']);self.assertNotIn('equivalent',stats([0,0,0]))
if __name__=='__main__':unittest.main()

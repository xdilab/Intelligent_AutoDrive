import io,unittest
import torch
from dcb import DCBState,objective
from model import ContextualRoIHead,objective as focal
class Tests(unittest.TestCase):
 def test_weights_gradient(self):
  state=DCBState();state.S.fill_(.49);state.N.fill_(1);z=torch.full((2,184),torch.logit(torch.tensor(.54)).item(),requires_grad=True);y=torch.zeros_like(z);y[0]=1;b,w=state.weights(z,y);self.assertFalse(b.requires_grad);torch.testing.assert_close(b[0],torch.full((184,),1.45));torch.testing.assert_close(b[1],torch.full((184,),.99));old=state.S.clone();loss=state.classification(z,y);loss.backward();torch.testing.assert_close(z.grad,b*(z.sigmoid().detach()-y)/z.numel());torch.testing.assert_close(state.S,old+w.sum(0,dtype=torch.float64))
 def test_zero_no_update_and_resume(self):
  state=DCBState();z=torch.zeros(2,184,requires_grad=True);y=torch.zeros_like(z);y[0,:3]=1;b,_=state.weights(z,y);torch.testing.assert_close(b,.5+y*.5);state.classification(z,y,False);self.assertEqual(state.N.sum().item(),0);state.classification(z,y);buf=io.BytesIO();torch.save(state.state_dict(),buf);buf.seek(0);other=DCBState();other.load_state_dict(torch.load(buf,weights_only=True));torch.testing.assert_close(other.weights(z,y)[0],state.weights(z,y)[0]);self.assertTrue(torch.isfinite(other.weights(z,y)[0]).all())
 def test_focal_path(self):
  z=torch.randn(3,184,requires_grad=True);sim=torch.randn(3,184,requires_grad=True);y=(torch.rand(3,184)>.9).float();alpha=torch.rand(184);o={'logits':z,'contrastive_logits':sim};state=DCBState()
  for weight in [0,.001]:
   a=objective(o,y,alpha,weight,state,'focal');b=focal(o,y,alpha,weight)
   for x,v in zip(a,b):self.assertTrue(torch.equal(x,v))
  self.assertEqual(state.N.sum().item(),0)
 def test_next_step_resume(self):
  torch.manual_seed(9);m=torch.nn.Linear(4,184);opt=torch.optim.AdamW(m.parameters());st=DCBState();x=torch.randn(3,4);y=(torch.rand(3,184)>.8).float()
  def step(m,opt,st):opt.zero_grad();loss=st.classification(m(x),y);loss.backward();opt.step()
  step(m,opt,st);buf=io.BytesIO();torch.save({'m':m.state_dict(),'o':opt.state_dict(),'s':st.state_dict()},buf);buf.seek(0);ck=torch.load(buf,weights_only=False);m2=torch.nn.Linear(4,184);m2.load_state_dict(ck['m']);o2=torch.optim.AdamW(m2.parameters());o2.load_state_dict(ck['o']);s2=DCBState();s2.load_state_dict(ck['s']);step(m,opt,st);step(m2,o2,s2)
  for a,b in zip(m.parameters(),m2.parameters()):self.assertTrue(torch.equal(a,b))
  self.assertTrue(torch.equal(st.S,s2.S));self.assertTrue(torch.equal(st.N,s2.N))
if __name__=='__main__':unittest.main()

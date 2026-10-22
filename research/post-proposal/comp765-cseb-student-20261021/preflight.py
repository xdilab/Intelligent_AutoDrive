"""Focused correctness tests before substantial work; no final metrics."""
from common import *
from train import update,capture,restore,sequence
import copy,unittest,importlib.util
def main():
 setup();old=read(EXPANDED/'manifest.json');r=next(r for r in old['sequences'] if r['split']=='train');seq=[to_gpu(v) for v in torch.load(r['path'],weights_only=True)]*4;checks={}
 # Preserve complete existing test suite.
 spec=importlib.util.spec_from_file_location('legacy_student_tests',OLD/'test_student.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 res=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromModule(mod));assert res.wasSuccessful();checks['legacy_suite']=res.testsRun
 initials=[];losses=[]
 for kd in [0.,1.]:
  setup(0);m=Student().cuda();initials.append(tensor_hash(m.state_dict()));opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);losses.append(update(m,opt,seq,kd))
 assert initials[0]==initials[1];checks['paired_initialization']=True;checks['finite_32frame_smoke_losses']=losses
 # Save/reload actual optimizer+all RNG, verify exact next update and random draws.
 ck=copy.deepcopy(capture(m,opt));test=ROOT/'preflight-resume.pt';p.atomic_save(ck,test)
 expected_loss=update(m,opt,seq,1);expected=tensor_hash(m.state_dict());expected_rand=[torch.rand(3).tolist(),torch.rand(3,device='cuda').tolist(),np.random.rand(3).tolist(),random.random()]
 n=Student().cuda();o=torch.optim.AdamW(n.parameters(),lr=.001,weight_decay=.01);restore(n,o,torch.load(test,map_location='cpu',weights_only=False));actual_loss=update(n,o,seq,1);actual=tensor_hash(n.state_dict());actual_rand=[torch.rand(3).tolist(),torch.rand(3,device='cuda').tolist(),np.random.rand(3).tolist(),random.random()]
 assert expected==actual and expected_rand==actual_rand and expected_loss==actual_loss;checks['optimizer_rng_resume_exact']=True
 # Prefix predictions cannot be affected by a different future; state-only frame works.
 m.eval()
 with torch.no_grad():
  z,y=sequence(m,seq);changed=[dict(x) for x in seq];changed[-1]=dict(changed[-1],scene=changed[-1]['scene']+20)
  z2,_=sequence(m,changed);prefix=sum(len(r['boxes']) for r in seq[:-1]);torch.testing.assert_close(z[:prefix],z2[:prefix],rtol=0,atol=0)
  reset,_=sequence(m,seq);torch.testing.assert_close(z,reset,rtol=0,atol=0)
 checks['causal_prefix_and_reset_exact']=True
 raw=torch.full((2,184),.6,device='cuda');q=torch.tensor([.25,.5],device='cuda');scores=p.detector_scores(raw,q);torch.testing.assert_close(scores[:,1],raw[:,1]*q);torch.testing.assert_close(scores[:,0],q);checks['detector_confidence_once']=True
 logits=torch.zeros((2,184),device='cuda',requires_grad=True);y=torch.zeros_like(logits);t=y.clone();y[:,[11,12,98]]=1;t[:,[11,12,98]]=.8;objective(logits,y,t,1).backward();assert (logits.grad[:,[11,12,98]]<0).all() and (logits.grad[:,0]==0).all();checks['multilabel_and_agentness_mask']=True
 assert sum(v.numel() for v in m.parameters())==1108568;checks['parameter_count']=1108568
 save(ROOT/'preflight.json',dict(passed=True,time=time.time(),checks=checks,code_sha256={str(x):sha(x) for x in [HERE/'train.py',OLD/'model.py',OLD/'pilot.py']}));progress('preflight-passed',checks=checks)
if __name__=='__main__':main()

from assoc_common import *
from correction import Correction

def main():
 setup();rules=read(ROOT/'rules.json');c=Correction(rules,'cuda');checks={}
 n=4;z=torch.randn(n,184,device='cuda');boxes=torch.tensor([[.1,.1,.2,.2],[.3,.1,.4,.2],[.6,.6,.7,.7],[.4,.2,.45,.3]],device='cuda');agents=torch.tensor([0,1,0,2],device='cuda');q=torch.tensor([.9,.8,.7,.6],device='cuda')
 base=p.detector_scores(z.sigmoid(),q);assert torch.equal(c.scores(z,q),base);assert torch.equal(c.scores(z,q,c.delta(z,boxes,agents,q,'association'),0),base);checks['zero_strength_exact_original_score_assembly']=True
 d=c.delta(z,boxes,agents,q,'association');out=c.scores(z,q,d,1)
 assert torch.equal(out[:,:49],base[:,:49]);assert torch.equal(out[:,0],q);assert torch.isfinite(out).all();assert d.abs().max()<=1;checks['primitive_slots_q_once_finite_bounded']=True
 incompatible=agents[:,None]!=c.event_agents[None,:];assert (d[incompatible]==0).all();checks['agent_compatible_rows']=True
 single=c.delta(z[:1],boxes[:1],agents[:1],q[:1],'association');assert (single==0).all();assert torch.equal(c.scores(z[:1],q[:1],single,1),base[:1]);checks['single_actor_no_self_reinforcement']=True
 perm=torch.tensor([2,0,3,1],device='cuda');assert torch.allclose(c.delta(z[perm],boxes[perm],agents[perm],q[perm],'association'),d[perm],atol=1e-7,rtol=0);checks['permutation_invariance']=True
 duplicated=c.delta(torch.cat([z,z[1:2]]),torch.cat([boxes,boxes[1:2]]),torch.cat([agents,agents[1:2]]),torch.cat([q,q[1:2]]),'association');assert torch.allclose(duplicated[:n],d,atol=1e-7,rtol=0);checks['duplicate_context_invariance']=True
 assert c.delta(z[:0],boxes[:0],agents[:0],q[:0],'association').shape==(0,135);assert (c.delta(z,boxes,agents,q*0,'association')==0).all();checks['empty_and_unreliable_context_identity']=True
 # Deliberately nonzero synthetic rule verifies direction and target attribution, independently of mined effects.
 fake=dict(rules);w=np.zeros((135,135));w[9,0]=.8;fake['weights']=w.tolist();f=Correction(fake,'cuda');zz=torch.zeros_like(z);dd=f.delta(zz,boxes,agents,q,'association');assert dd[0,0]>0 and (dd[agents!=0,0]==0).all();checks['nonzero_rule_target_attribution']=True
 manifest=read(BASE_ROOT/'data-manifest.json');assert rules['events']==manifest['labels']['duplex']+manifest['labels']['triplet'];checks['label_order_and_training_sources']=True
 checkpoints={}
 for seed in SEEDS:
  path=checkpoint(seed);h=sha(path);done=read(path.parent/'training-complete.json');assert h==done['checkpoint_sha256'];backup=BASE_ROOT/'verified-checkpoint-backup'/path.parent.name/'best.pt';assert sha(backup)==h
  ck=torch.load(path,map_location='cpu',weights_only=False);m=Student().cuda().eval();m.load_state_dict(ck['model']);assert all(torch.isfinite(v).all() for v in m.parameters());assert sum(v.numel() for v in m.parameters())==1108568
  seq=torch.load(manifest['sequences'][0]['path'],map_location='cpu',weights_only=False);row=to_gpu(seq[0]);nextrow=to_gpu(seq[1])
  with torch.inference_mode():
   first,state=m(row['roi'],row['scene'],row['boxes'],row['agents']);repeat,again=m(row['roi'],row['scene'],row['boxes'],row['agents']);assert torch.equal(first,repeat) and torch.equal(state,again)
   later,_=m(nextrow['roi'],nextrow['scene']+7,nextrow['boxes'],nextrow['agents'],state);assert torch.equal(first,repeat)
  checkpoints[str(path)]=dict(sha256=h,backup=str(backup),backup_sha256=sha(backup),selected_epoch=done['selected_epoch'])
 checks['unchanged_checkpoints_shapes_reset_causal_function']=True
 # A representative cached frame benchmark is setup-only; full-candidate raw-image benchmark follows.
 row=to_gpu(seq[0]);m=Student().cuda().eval();m.load_state_dict(ck['model']);times=[]
 with torch.inference_mode():
  logits,_=m(row['roi'],row['scene'],row['boxes'],row['agents'])
  for i in range(110):
   torch.cuda.synchronize();t=time.perf_counter();delta=c.delta(logits,row['boxes'],row['agents'],row['q'],'association');score=c.scores(logits,row['q'],delta,1);score.cpu();torch.cuda.synchronize()
   if i>=10:times.append(1000*(time.perf_counter()-t))
 save(ROOT/'preflight.json',dict(passed=True,time=time.time(),checks=checks,checkpoints=checkpoints,benchmark=dict(scope='association correction plus score assembly/export only; cached proposal frame',samples=100,warmup=10,candidates=len(row['boxes']),median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95))),runtime_notes='No student retraining; optimizer/RNG resume unchanged and not invoked. Official inference resumes at video boundaries after flushing all arrays.'))
 progress('preflight-passed',checks=len(checks),correction_ms=float(np.median(times)))
if __name__=='__main__':main()

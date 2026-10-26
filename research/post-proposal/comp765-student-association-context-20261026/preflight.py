from assoc_common import *
from correction import Correction
from single_correction import Correction as Single
from mine import related

def main():
 setup();rules=read(ROOT/'rules.json');c=Correction(rules,'cuda');single=Single(rules,'cuda');checks={};n=8;z=torch.randn(n,184,device='cuda');boxes=torch.tensor([[.1,.1,.2,.2],[.3,.1,.4,.2],[.6,.6,.7,.7],[.4,.2,.45,.3],[.2,.2,.27,.3],[.8,.8,.9,.9],[.15,.3,.2,.4],[.35,.4,.4,.5]],device='cuda');agents=torch.tensor([0,1,0,2,3,4,5,6],device='cuda');q=torch.linspace(.95,.3,n,device='cuda');base=p.detector_scores(z.sigmoid(),q);parts=c.components(z,boxes,agents,q)
 assert torch.equal(c.scores(z,q),base);checks['original_score_assembly_exact']=True
 for mode,d in parts.items():
  out=c.scores(z,q,d,1);assert torch.isfinite(out).all() and d.abs().max()<=1;assert torch.equal(out[:,:49],base[:,:49]);assert torch.equal(out[:,0],q);assert (d[agents[:,None]!=c.event_agents[None,:]]==0).all();assert torch.equal(c.scores(z,q,d,0),base)
 checks['finite_bounded_agent_mask_primitives_q_once_zero_strength']=True
 assert torch.equal(parts['full_single'],single.delta(z,boxes,agents,q,'association'));assert torch.equal(parts['full_frequency'],single.delta(z,boxes,agents,q,'frequency'));checks['single_context_and_frequency_parity']=True
 for mode in ['full_single','full_pair']:
  assert (c.delta(z[:1],boxes[:1],agents[:1],q[:1],mode)==0).all();assert (c.delta(z,boxes,agents,q*0,mode)==0).all();assert c.delta(z[:0],boxes[:0],agents[:0],q[:0],mode).shape==(0,135)
  perm=torch.tensor([2,0,7,3,1,5,6,4],device='cuda');assert torch.allclose(c.delta(z[perm],boxes[perm],agents[perm],q[perm],mode),parts[mode][perm],atol=1e-7,rtol=0)
  d=c.delta(torch.cat([z,z[1:2]]),torch.cat([boxes,boxes[1:2]]),torch.cat([agents,agents[1:2]]),torch.cat([q,q[1:2]]),mode);assert torch.allclose(d[:n],parts[mode],atol=1e-7,rtol=0)
 checks['no_self_no_context_empty_permutation_duplicate']=True
 # Removing incremental pair effects must exactly recover the full single-event condition.
 empty=dict(rules,pair_weights=np.zeros_like(rules['pair_weights']).tolist());ce=Correction(empty,'cuda');ee=ce.components(z,boxes,agents,q);assert torch.equal(ee['full_pair'],ee['full_single']);assert torch.equal(ee['full_pair_frequency'],ee['full_frequency']);checks['zero_pair_exact_single_recovery']=True
 # Independent numpy arithmetic on evidence; same evidence is immutable before all corrections.
 e=c.evidence(z,boxes,agents,q).cpu().numpy();w=np.asarray(rules['weights'],np.float32);pw=np.asarray(rules['pair_weights'],np.float32);pairs=np.asarray(rules['pair_antecedents']);pe=np.minimum(e[:,pairs[:,0]],e[:,pairs[:,1]]);reference=np.clip((e@w)/(1+e@(w!=0).astype(np.float32))+(pe@pw)/(1+pe@(pw!=0).astype(np.float32)),-1,1)*(agents.cpu().numpy()[:,None]==np.array(rules['event_agents'])[None,:]);assert np.max(np.abs(reference-parts['full_pair'].cpu().numpy()))<1e-6;checks['independent_numpy_pair_arithmetic']=True
 # Boolean min reproduces AND, and one missing antecedent suppresses the pair exactly.
 x=torch.tensor([0.,0.,1.,1.]);y=torch.tensor([0.,1.,0.,1.]);assert torch.equal(torch.minimum(x,y),x*y);checks['conjunction_truth_table']=True
 ev=rules['events'];weights=np.array(rules['pair_weights']);assert (weights!=0).sum(0).max()<=16
 for i,j in zip(*np.where(weights!=0)):
  a,b=pairs[i];assert a<b and not related(ev[a],ev[b]) and not related(ev[a],ev[j]) and not related(ev[b],ev[j])
 assert ev==read(BASE_ROOT/'data-manifest.json')['labels']['duplex']+read(BASE_ROOT/'data-manifest.json')['labels']['triplet'];checks['class_order_nonhierarchical_cap']=True
 mining=read(ROOT/'mining-summary.json');assert mining['short_max_abs_reproduction_error']<1e-12 and mining['short_actor_frames']==45766 and mining['annotated_frames']==12270;checks['pilot_mining_reproduction_and_full_coverage']=True
 splits=read(ROOT/'split-manifest.json');train=set(splits['groups']['train']);assert not train.intersection(splits['groups']['selection']+splits['groups']['final']);assert set(r['video'] for r in rules['video_records'])==train;checks['training_only_rule_inputs']=True
 checkpoints={};manifest=read(BASE_ROOT/'data-manifest.json');seq=torch.load(manifest['sequences'][0]['path'],map_location='cpu',weights_only=False);row=to_gpu(seq[0])
 for seed in SEEDS:
  path=checkpoint(seed);h=sha(path);done=read(path.parent/'training-complete.json');backup=BASE_ROOT/'verified-checkpoint-backup'/path.parent.name/'best.pt';assert h==done['checkpoint_sha256']==sha(backup);ck=torch.load(path,map_location='cpu',weights_only=False);m=Student().cuda().eval();m.load_state_dict(ck['model']);assert sum(v.numel() for v in m.parameters())==1108568
  with torch.inference_mode():
   z1,h1=m(row['roi'],row['scene'],row['boxes'],row['agents']);z2,h2=m(row['roi'],row['scene'],row['boxes'],row['agents']);assert torch.equal(z1,z2) and torch.equal(h1,h2) and torch.isfinite(z1).all()
  checkpoints[str(path)]=dict(sha256=h,backup=str(backup),backup_sha256=sha(backup),selected_epoch=done['selected_epoch'])
 checks['unchanged_checkpoints_shapes_reset']=True
 times={k:[] for k in ['full_single','full_pair']}
 with torch.inference_mode():
  for mode in times:
   for i in range(110):
    torch.cuda.synchronize();t=time.perf_counter();dd=single.delta(z1,row['boxes'],row['agents'],row['q'],'association') if mode=='full_single' else c.delta(z1,row['boxes'],row['agents'],row['q'],'full_pair');c.scores(z1,row['q'],dd,1).cpu();torch.cuda.synchronize()
    if i>=10:times[mode].append(1000*(time.perf_counter()-t))
 save(ROOT/'preflight.json',dict(passed=True,time=time.time(),checks=checks,checkpoints=checkpoints,benchmark={k:dict(candidates=len(row['boxes']),samples=100,warmup=10,median_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95))) for k,v in times.items()},resume='Unchanged student; no training/optimizer state to resume. Score arrays flush all7conditions/all3seeds before video receipt commit. Benchmark receipt replayed by full run.'))
 progress('preflight-passed',checks=len(checks))
if __name__=='__main__':main()

"""Training-only full-video transactions and bounded residual conjunction rules."""
from assoc_common import *
from single_correction import neighbors
import argparse,csv,itertools

def logit(x):return np.log(np.clip(x,1e-6,1-1e-6)/np.clip(1-x,1e-6,1-1e-6))
def related(a,b):return a==b or a.startswith(b+'-') or b.startswith(a+'-')
def transactions(ann,video,labels):
 look={k:{v:i for i,v in enumerate(labels[k])} for k in ['agent','duplex','triplet']};ea=np.array([labels['agent'].index(e.split('-')[0]) for e in labels['duplex']+labels['triplet']]);xs=[];ys=[];ags=[];fids=[];frames=0
 for fid,frame in sorted(ann['db'][video]['frames'].items(),key=lambda kv:int(kv[0])):
  if frame.get('annotated')!=1:continue
  frames+=1
  boxes=[];yy=[];aa=[]
  for item in frame.get('annos',{}).values():
   aids=[look['agent'][ann['all_agent_labels'][i]] for i in item.get('agent_ids',[]) if ann['all_agent_labels'][i] in look['agent']]
   if not aids:continue
   assert len(aids)==1
   y=np.zeros(135,dtype=np.float64)
   for kind,off in [('duplex',0),('triplet',49)]:
    for i in item.get(kind+'_ids',[]):
     name=ann['all_'+kind+'_labels'][i]
     if name in look[kind]:y[off+look[kind][name]]=1
   assert np.all(ea[y>0]==aids[0]);boxes.append(item['box']);yy.append(y);aa.append(aids[0])
  if not boxes:continue
  b=torch.tensor(boxes,dtype=torch.float32);adj=neighbors(b,b).numpy();np.fill_diagonal(adj,False);y=np.array(yy);x=(adj.astype(float)@y>0)
  xs.append(x);ys.append(y.astype(bool));ags.extend(aa);fids.extend([int(fid)]*len(aa))
 return np.concatenate(xs),np.concatenate(ys),np.array(ags),np.array(fids),frames

def init(n):return dict(totals=np.zeros(10),a=np.zeros((10,n)),y=np.zeros(135),joint=np.zeros((n,135)),raw=np.zeros((n,135),np.int64),videos=np.zeros((n,135),np.int64))
def add(s,X,Y,A,pairs=None):
 # CPU float64 matrix products. Per-agent/video total weight1, identical to pilot.
 xx=X.astype(float) if pairs is None else (X[:,pairs[:,0]]&X[:,pairs[:,1]]).astype(float);yy=Y.astype(float)
 tx=torch.from_numpy(xx);ty=torch.from_numpy(yy);raw=(tx.T@ty).numpy();s['raw']+=raw.astype(np.int64);s['videos']+=(raw>0)
 for agent in range(10):
  take=A==agent;n=int(take.sum())
  if n:s['totals'][agent]+=1;s['a'][agent]+=xx[take].mean(0);s['y']+=yy[take].mean(0);s['joint']+=(torch.from_numpy(xx[take]).T@torch.from_numpy(yy[take])).numpy()/n

def probabilities(s,ea):
 marginal=s['y']/np.maximum(s['totals'][ea],1);context=s['a']/np.maximum(s['totals'][:,None],1);ante=s['a'][ea].T;conditional=(s['joint']+5*marginal[None,:])/(ante+5)
 return marginal,context,ante,conditional

def singles(s,events,ea):
 marginal,context,ante,conditional=probabilities(s,ea);effect=np.clip(logit(conditional)-logit(marginal)[None,:],-1,1);lift=conditional/np.maximum(marginal[None,:],1e-12);keep=(s['raw']>=5)&(s['videos']>=3)&(np.abs(np.log(np.maximum(lift,1e-12)))>=np.log(1.1))
 for i,a in enumerate(events):
  for j,b in enumerate(events):
   if related(a,b):keep[i,j]=False
 effect[~keep]=0
 return dict(events=events,event_agents=ea.tolist(),weights=effect.tolist(),context_marginal=context.tolist(),target_marginal=marginal.tolist(),rules=int(keep.sum())),conditional

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--benchmark-videos',type=int);a=ap.parse_args();setup();assert not (ROOT/'freeze.json').exists();start=time.perf_counter();protocol_sha=sha(HERE/'protocol.json');before=ROOT/'protocol-before-mining.json';prior=read(before);assert prior['protocol']==read(HERE/'protocol.json');prior.update(sha256=protocol_sha,time=time.time());save(before,prior)
 labels=read(BASE_ROOT/'data-manifest.json')['labels'];splits=read(ROOT/'split-manifest.json');events=labels['duplex']+labels['triplet'];ea=np.array([labels['agent'].index(e.split('-')[0]) for e in events]);pairs=np.array([(i,j) for i in range(135) for j in range(i+1,135) if not related(events[i],events[j])]);short=init(135);full=init(135);pair=init(len(pairs));ann=read(p.ROOT/'road_waymo_trainval_v1.1.json');records=[];compute_start=time.perf_counter()
 for video in splits['groups']['train'][:a.benchmark_videos]:
  t=time.perf_counter();X,Y,A,F,nf=transactions(ann,video,labels);first=splits['start_frames'][video];take=(F>=first)&(F<first+32);add(short,X[take],Y[take],A[take]);add(full,X,Y,A);add(pair,X,Y,A,pairs);rec=dict(video=video,annotated_frames=nf,nonempty_target_frames=len(set(F)),target_actor_frames=len(A),short_nonempty_target_frames=len(set(F[take])),short_frames=sum(ann['db'][video]['frames'].get(str(f),{}).get('annotated')==1 for f in range(first,first+32)),short_actor_frames=int(take.sum()),seconds=time.perf_counter()-t);records.append(rec);progress('mining-training-only',videos=len(records),total=64,**rec)
 if a.benchmark_videos:
  save(ROOT/'mining-benchmark.json',dict(videos=len(records),seconds=time.perf_counter()-start,compute_seconds=time.perf_counter()-compute_start,pair_candidates=len(pairs),records=records,projected64video_compute_seconds=(time.perf_counter()-compute_start)*64/len(records)));return
 assert len(records)==64;old,_=singles(short,events,ea);pilot=read(PILOT_ROOT/'rules.json');error=max(float(np.max(np.abs(np.array(old[k])-np.array(pilot[k])))) for k in ['weights','context_marginal','target_marginal']);assert error<1e-12,('short reproduction',error);assert sum(r['short_actor_frames'] for r in records)==45766
 rule,conditional=singles(full,events,ea);marginal,context,ante,pc=probabilities(pair,ea);assert np.allclose(marginal,rule['target_marginal'],atol=1e-15);lo=np.minimum(conditional[pairs[:,0]],conditional[pairs[:,1]]);hi=np.maximum(conditional[pairs[:,0]],conditional[pairs[:,1]]);effect=np.where(pc>hi,logit(pc)-logit(hi),np.where(pc<lo,logit(pc)-logit(lo),0));effect=np.clip(effect,-1,1);eligible=(pair['raw']>=5)&(pair['videos']>=3)&(np.abs(effect)>=np.log(1.1))
 for i,(a,b) in enumerate(pairs):
  for j,c in enumerate(events):
   if related(events[a],c) or related(events[b],c):eligible[i,j]=False
 keep=np.zeros_like(eligible)
 for j in range(135):
  ii=np.flatnonzero(eligible[:,j]);order=sorted(ii,key=lambda i:(-int(pair['videos'][i,j]),-abs(effect[i,j]),int(pairs[i,0]),int(pairs[i,1])));keep[order[:16],j]=True
 used=np.flatnonzero(keep.any(1));effect[~keep]=0;rule.update(pair_antecedents=pairs[used].tolist(),pair_weights=effect[used].tolist(),pair_context_marginal=context[:,used].tolist(),pair_rules=int(keep.sum()),pair_candidate_rules=int(eligible.sum()),pair_distinct_antecedents=len(used),protocol_sha256=protocol_sha,annotation_sha256=sha(p.ROOT/'road_waymo_trainval_v1.1.json'),video_records=records)
 save(ROOT/'rules.json',rule);save(ROOT/'short-rules.json',pilot)
 # All support-qualified, nonhierarchical candidate effects, including discarded/capped rules, are auditable.
 with (ROOT/'pair-candidates.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['antecedent_a','antecedent_b','consequent','joint_actor_frames','joint_videos','pair_confidence','single_a_confidence','single_b_confidence','incremental_effect','retained'])
  for i,j in zip(*np.where(eligible)):w.writerow([events[pairs[i,0]],events[pairs[i,1]],events[j],pair['raw'][i,j],pair['videos'][i,j],pc[i,j],conditional[pairs[i,0],j],conditional[pairs[i,1],j],float(np.clip((logit(pc[i,j])-logit(hi[i,j])) if pc[i,j]>hi[i,j] else (logit(pc[i,j])-logit(lo[i,j])), -1,1)),bool(keep[i,j])])
 with (ROOT/'single-rules.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['antecedent','consequent','joint_actor_frames','joint_videos','shrunk_conditional','bounded_effect'])
  for i,j in zip(*np.where(np.array(rule['weights'])!=0)):w.writerow([events[i],events[j],full['raw'][i,j],full['videos'][i,j],conditional[i,j],rule['weights'][i][j]])
 summary=dict(time=time.time(),seconds=time.perf_counter()-start,training_videos=64,annotated_frames=sum(r['annotated_frames'] for r in records),target_actor_frames=sum(r['target_actor_frames'] for r in records),short_frames=sum(r['short_frames'] for r in records),short_actor_frames=sum(r['short_actor_frames'] for r in records),short_max_abs_reproduction_error=error,single_rules=rule['rules'],pair_rules=rule['pair_rules'],pair_distinct_antecedents=len(used),pair_candidates=len(pairs),pair_candidate_rules=rule['pair_candidate_rules'],records=records);save(ROOT/'mining-summary.json',summary);progress('mining-complete',**{k:v for k,v in summary.items() if k not in ['records','time']})
if __name__=='__main__':main()

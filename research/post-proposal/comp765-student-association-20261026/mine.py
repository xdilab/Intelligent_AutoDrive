"""Mine only training GT, balanced by target agent/video; no AP is loaded."""
from assoc_common import *
from correction import neighbors
import csv,shutil

def main():
 setup();ROOT.mkdir(exist_ok=True)
 protocol=read(HERE/'protocol.json');save(ROOT/'protocol-before-mining.json',dict(time=time.time(),sha256=sha(HERE/'protocol.json'),protocol=protocol))
 manifest=read(BASE_ROOT/'data-manifest.json');splits=read(BASE_ROOT/'split-manifest.json');labels=manifest['labels'];events=labels['duplex']+labels['triplet'];event_agents=np.array([labels['agent'].index(x.split('-')[0]) for x in events])
 assert len(events)==135 and len(set(events))==135
 assert not set(splits['groups']['train'])&set(splits['groups']['selection']) and not set(splits['groups']['final'])&set(splits['groups']['train']+splits['groups']['selection'])
 shutil.copyfile(BASE_ROOT/'split-manifest.json',ROOT/'split-manifest.json')
 ann=read(p.ROOT/'road_waymo_trainval_v1.1.json');lookups={k:{x:i for i,x in enumerate(labels[k])} for k in ['agent','duplex','triplet']}
 totals=np.zeros(10);a_sum=np.zeros((10,135));y_sum=np.zeros(135);joint=np.zeros((135,135));joint_frames=np.zeros((135,135),int);joint_videos=np.zeros((135,135),int);video_records=[];cache_hashes={}
 for rec in manifest['sequences']:
  if rec['split']!='train':continue
  assert rec['video'] in splits['groups']['train'];path=Path(rec['path']);assert sha(path)==rec['sha256'];cache_hashes[str(path)]=rec['sha256']
  seq=torch.load(path,map_location='cpu',weights_only=False);Xs=[];Ys=[];As=[];nframes=0
  for row in seq:
   frame=ann['db'][rec['video']]['frames'].get(str(row['fid']),{})
   if frame.get('annotated')!=1:continue
   boxes=[];ys=[];agents=[]
   for item in frame.get('annos',{}).values():
    aids=[lookups['agent'][ann['all_agent_labels'][i]] for i in item.get('agent_ids',[]) if ann['all_agent_labels'][i] in lookups['agent']]
    if not aids:continue
    assert len(aids)==1,(rec['video'],row['fid'],aids)
    y=np.zeros(135)
    for kind,offset in [('duplex',0),('triplet',49)]:
     for i in item.get(kind+'_ids',[]):
      name=ann['all_'+kind+'_labels'][i]
      if name in lookups[kind]:y[offset+lookups[kind][name]]=1
    assert np.all(event_agents[y>0]==aids[0])
    boxes.append(item['box']);ys.append(y);agents.append(aids[0])
   if not boxes:continue
   box=torch.tensor(boxes,dtype=torch.float32);adj=neighbors(box,box).numpy();np.fill_diagonal(adj,False);y=np.array(ys);x=(adj.astype(float)@y>0).astype(float)
   Xs.append(x);Ys.append(y);As.append(np.array(agents));nframes+=1
  X=np.concatenate(Xs);Y=np.concatenate(Ys);A=np.concatenate(As);raw=X.T@Y;joint_frames+=raw.astype(int);joint_videos+=(raw>0)
  for agent in range(10):
   take=A==agent;n=int(take.sum())
   if not n:continue
   totals[agent]+=1;a_sum[agent]+=X[take].mean(0);y_sum+=Y[take].mean(0);joint+=X[take].T@Y[take]/n
  video_records.append(dict(video=rec['video'],annotated_frames=nframes,target_actor_frames=len(A),event_actor_frames=int((Y.sum(1)>0).sum())))
  progress('mining-training-only',videos=len(video_records),total=64,video=rec['video'])
 assert len(video_records)==64
 marginal=y_sum/np.maximum(totals[event_agents],1);context=a_sum/np.maximum(totals[:,None],1);ante=a_sum[event_agents].T
 conditional=(joint+5*marginal[None,:])/(ante+5)
 logit=lambda x:np.log(np.clip(x,1e-6,1-1e-6)/np.clip(1-x,1e-6,1-1e-6))
 effect=np.clip(logit(conditional)-logit(marginal)[None,:],-1,1);lift=conditional/np.maximum(marginal[None,:],1e-12)
 keep=(joint_frames>=5)&(joint_videos>=3)&(np.abs(np.log(np.maximum(lift,1e-12)))>=protocol['rule_mining']['min_absolute_log_lift'])
 for i,a in enumerate(events):
  for j,b in enumerate(events):
   if a==b or a.startswith(b+'-') or b.startswith(a+'-'):keep[i,j]=False
 effect[~keep]=0
 data=dict(events=events,event_agents=event_agents.tolist(),weights=effect.tolist(),context_marginal=context.tolist(),target_marginal=marginal.tolist(),rules=int(keep.sum()),training_videos=64,video_records=video_records,protocol_sha256=sha(HERE/'protocol.json'),annotation_sha256=sha(p.ROOT/'road_waymo_trainval_v1.1.json'),training_cache_hashes=cache_hashes)
 save(ROOT/'rules.json',data)
 with (ROOT/'rules.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['antecedent_on_other_actor','consequent_on_target','target_agent','joint_actor_frames','joint_videos','video_balanced_joint','antecedent_video_weight','target_probability','raw_confidence','shrunk_confidence','shrunk_lift','bounded_logodds_effect'])
  for i,j in zip(*np.where(keep)):w.writerow([events[i],events[j],labels['agent'][event_agents[j]],joint_frames[i,j],joint_videos[i,j],joint[i,j],ante[i,j],marginal[j],joint[i,j]/max(ante[i,j],1e-12),conditional[i,j],lift[i,j],effect[i,j]])
 save(ROOT/'mining-summary.json',dict(time=time.time(),rules=int(keep.sum()),positive=int((effect>0).sum()),negative=int((effect<0).sum()),consequents_covered=int(keep.any(0).sum()),duplex_covered=int(keep[:,:49].any(0).sum()),triplet_covered=int(keep[:,49:].any(0).sum()),target_actor_frames=sum(x['target_actor_frames'] for x in video_records),target_agents_videos=totals.tolist(),rules_sha256=sha(ROOT/'rules.json')))
 progress('mining-complete',rules=int(keep.sum()))
if __name__=='__main__':main()

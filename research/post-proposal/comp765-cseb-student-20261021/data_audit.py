"""Validate every available prepared row; no metric inspection or tuning."""
from common import *
def main():
 torch.set_num_threads(2)
 m=read(ROOT/'data-manifest.json');s=read(ROOT/'split-manifest.json');support={k:np.zeros(184,dtype=np.int64) for k in ['train','selection']};counts={k:dict(videos=0,frames=0,annotated=0,proposals=0,matched=0) for k in support};unannotated=[]
 for rec in m['sequences']:
  assert rec['video'] in s['groups'][rec['split']] and sha(rec['path'])==rec['sha256'];rows=torch.load(rec['path'],map_location='cpu',weights_only=True);first=s['start_frames'][rec['video']];assert [r['fid'] for r in rows]==list(range(first,first+32));c=counts[rec['split']];c['videos']+=1
  for r in rows:
   n=len(r['boxes']);assert r['roi'].shape==(n,1024) and r['scene'].shape==(1024,) and r['boxes'].shape==(n,4) and r['labels'].shape==(n,184) and r['agents'].shape==(n,) and r['q'].shape==(n,)
   assert all(torch.isfinite(v).all() for v in r.values() if torch.is_tensor(v));assert ((r['labels']==0)|(r['labels']==1)).all();assert ((r['q']>0)&(r['q']<=1)).all();assert ((r['agents']>=0)&(r['agents']<10)).all();assert ((r['boxes']>=0)&(r['boxes']<=1)).all();assert (r['boxes'][:,2:]>=r['boxes'][:,:2]).all()
   assert ('teacher' in r)==(rec['split']=='train')
   if 'teacher' in r:assert r['teacher'].shape==(n,184) and ((r['teacher']>=0)&(r['teacher']<=1)).all() and (r['teacher'][:,0]==1).all()
   if not r['annotated']:assert n==0;unannotated.append(dict(video=rec['video'],fid=r['fid'],split=rec['split']))
   c['frames']+=1;c['annotated']+=int(r['annotated']);c['proposals']+=n;c['matched']+=int(r['labels'][:,0].sum());support[rec['split']]+=r['labels'].sum(0).numpy().astype(np.int64)
 complete=len(m['sequences'])==76
 if complete:assert counts['train']['videos']==64 and counts['selection']['videos']==12
 save(ROOT/'data-audit.json',dict(passed=True,complete=complete,time=time.time(),counts=counts,positive_class_counts={k:v.tolist() for k,v in support.items()},unannotated_state_only_frames=unannotated,data_manifest_sha256=sha(ROOT/'data-manifest.json'),teacher_targets_only_train=True));print(json.dumps(dict(passed=True,complete=complete,counts=counts)),flush=True)
if __name__=='__main__':main()

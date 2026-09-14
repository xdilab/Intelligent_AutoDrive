import argparse,json,pickle,hashlib,time
from pathlib import Path
import numpy as np
from common import partitions,sha
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root;cfg=json.loads((root/'code/protocol.json').read_text());src=Path(cfg['source_study']);out=root/'data';out.mkdir(exist_ok=True,parents=True)
print('Loading immutable original training cache',flush=True);d=pickle.load(open(src/'data/crop_feats_train.pkl','rb'));keys=[k for k in sorted(d['feats']) if len(d['feats'][k])];videos=sorted({k.rsplit('_',1)[0] for k in keys});split=partitions(videos);mapping={v:j for j,name in enumerate(['expert','gate','development']) for v in split[name]}
manifest=json.loads((src/'code/shared-frames.json').read_text());finalvideos={k.rsplit('_',1)[0] for k in manifest['frames']};assert not finalvideos&set(videos)
assert not(set(split['expert'])&set(split['gate']) or set(split['expert'])&set(split['development']) or set(split['gate'])&set(split['development']))
N=sum(len(d['feats'][k]) for k in keys);x=np.lib.format.open_memmap(out/'x.npy',mode='w+',dtype=np.float32,shape=(N,1024));y=np.lib.format.open_memmap(out/'y.npy',mode='w+',dtype=np.float32,shape=(N,184));part=np.lib.format.open_memmap(out/'partition.npy',mode='w+',dtype=np.int8,shape=(N,));fold=np.lib.format.open_memmap(out/'expert-fold.npy',mode='w+',dtype=np.int8,shape=(N,));fold[:]=-1
vidfold={v:i%2 for i,v in enumerate(sorted(split['expert']))};pos=0;bad=0;counts=np.zeros((3,184),np.int64);frame_counts=[0,0,0]
for k in keys:
    f=np.array(d['feats'][k],np.float32);t=np.asarray(d['targets'][k],np.float32);assert f.shape[1]==1024 and t.shape==(len(f),184) and np.isfinite(t).all() and ((t==0)|(t==1)).all();bad+=int((~np.isfinite(f)).sum());np.nan_to_num(f,copy=False,nan=0,posinf=0,neginf=0);v=k.rsplit('_',1)[0];j=mapping[v];n=len(f);x[pos:pos+n]=f;y[pos:pos+n]=t;part[pos:pos+n]=j
    if j==0:fold[pos:pos+n]=vidfold[v]
    counts[j]+=t.sum(0).astype(np.int64);frame_counts[j]+=1;pos+=n
for z in [x,y,part,fold]:z.flush()
assert pos==N
report={'protocol':cfg,'videos':split,'video_counts':{k:len(v) for k,v in split.items()},'frame_counts':frame_counts,'row_counts':[int((part==j).sum()) for j in range(3)],'positive_counts':counts.tolist(),'expert_fold_video_counts':[sum(v==j for v in vidfold.values()) for j in [0,1]],'final_video_overlap':0,'nonfinite_elements_zeroed_as_original_recipe':bad,'training_keys_sha256':hashlib.sha256(('\n'.join(keys)+'\n').encode()).hexdigest(),'source_files':{str(p):sha(p) for p in [src/'data/crop_feats_train.pkl',src/'data/phrase_embeds.pt',src/'data/flat_alphas.pt',src/'code/shared-frames.json']},'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
(root/'results/preparation.json').write_text(json.dumps(report,indent=2)+'\n');print('PREPARED',json.dumps({k:report[k] for k in ['video_counts','row_counts','final_video_overlap']}),flush=True)

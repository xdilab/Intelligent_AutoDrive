"""Prepare only reserved gate rows. No expert/development/validation rows are fit data."""
from pathlib import Path
import hashlib,json,time
import numpy as np
ROOT=Path('/data/repos/wiki/artifacts/contextual-gate-20260919');PARENT=ROOT.parent/'contextual-roi-20260917'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()
def main():
 src=ROOT/'export';report=json.loads((src/'export.json').read_text());assert report['passed'];parent=json.loads((PARENT/'preparation.json').read_text());assert report['videos']==parent['split']
 videos=sorted(report['videos']['gate'],key=lambda v:hashlib.sha256(('contextual-gate-v1:'+v).encode()).hexdigest());fit=set(videos[:45]);select=set(videos[45:]);assert len(fit)==45 and len(select)==45 and not fit&select
 for s in ['expert','development']:assert not (fit|select)&set(report['videos'][s])
 official=json.loads((PARENT/'data/shared-frames.json').read_text());assert not (fit|select)&{k.rsplit('_',1)[0] for k in official['frames']}
 ann=json.loads(Path('/data/datasets/ROAD_plusplus/road_waymo_trainval_v1.1.json').read_text());data=ROOT/'data';data.mkdir(exist_ok=True);n=report['rows'];arrays={}
 for key,shape,dtype in [('crop',(n,1024),'float16'),('boxes',(n,4),'float32'),('targets',(n,184),'uint8'),('frame',(n,),'int32'),('partition',(n,),'uint8')]:arrays[key]=np.lib.format.open_memmap(data/f'gate-{key}.npy',mode='w+',dtype=dtype,shape=shape)
 pos=0;frames=0;seen=set();digests={}
 with (data/'gate.jsonl').open('w') as out:
  for shard in report['shards']:
   path=src/shard['file'];assert sha(path)==shard['sha256'];d=np.load(path);length=shard['n']
   for key in ['crop','boxes','targets']:arrays[key][pos:pos+length]=d[key]
   for row in shard['rows']:
    v=row['video'];fid=row['fid'];key=(v,fid);assert key not in seen;seen.add(key);nf=ann['db'][v]['numf'];fids=[min(max(fid-3+j,1),nf) for j in range(8)];hh=[]
    for fj in fids:
     rel=f'{v}/{fj:05d}.jpg'
     if rel not in digests:digests[rel]=sha(Path('/data/datasets/ROAD_plusplus/rgb-images')/rel)
     hh.append(digests[rel])
    start,end=pos+row['start'],pos+row['end'];arrays['frame'][start:end]=frames;arrays['partition'][start:end]=0 if v in fit else 1
    out.write(json.dumps({'video':v,'fid':fid,'fids':fids,'key_t':fids.index(fid),'boxes':arrays['boxes'][start:end].tolist(),'frame_sha256':hh,'row_start':start,'row_end':end,'frame_index':frames},separators=(',',':'))+'\n');frames+=1
   pos+=length;d.close();print('PREPARED',pos,n,flush=True)
 assert pos==n
 for a in arrays.values():a.flush()
 for name in ['phrase_embeds.pt','flat_alphas.pt','train-counts.json','shared-frames.json']:
  link=data/name
  if not link.exists():link.symlink_to(PARENT/'data'/name)
 output={'passed':True,'rows':n,'frames':frames,'fit_videos':sorted(fit),'selection_videos':sorted(select),'split_source_sha256':sha(PARENT/'preparation.json'),'export_sha256':sha(src/'export.json'),'time':time.time(),'files':{p.name:sha(p) for p in data.glob('gate-*npy')}}
 tmp=ROOT/'preparation.tmp';tmp.write_text(json.dumps(output,indent=2));tmp.replace(ROOT/'preparation.json');print('COMPLETE',n,frames,flush=True)
if __name__=='__main__':main()

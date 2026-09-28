"""Run on NCShare CPU allocation. Export verified original paired-cache rows only."""
import argparse, hashlib, json, pickle, shutil, time
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    root=a.output;root.mkdir(parents=True,exist_ok=True);data=root/'data';data.mkdir(exist_ok=True)
    old=Path('/work/bbyrd1/stage56-full-20260914');paired=Path('/work/bbyrd1/adapted-road-20260911/cache');source=Path('/work/bbyrd1/proposal-study-20260910')
    splits=json.loads(Path('/work/bbyrd1/class-gate-study-20260914/results/preparation.json').read_text())['videos']
    official=json.loads((source/'code/shared-frames.json').read_text());annpath=Path('/work/bbyrd1/adapted-road-20260911/road_waymo_trainval_v1.1.json');ann=json.loads(annpath.read_text())
    assert len(splits['expert'])==420 and len(splits['development'])==90 and len(splits['gate'])==90
    assert not (set(splits['expert']) & set(splits['development']) or set(splits['expert']) & set(splits['gate']) or set(splits['development']) & set(splits['gate']))
    assert not set(v for vs in splits.values() for v in vs)&{k.rsplit('_',1)[0] for k in official['frames']}
    report={'split':splits,'source_manifest_sha256':{},'shards':{},'splits':{}}
    allrows={}; splitrows={}; keys_by_split={}
    for split in ['train','dev']:
        rows=[json.loads(l) for l in (old/f'data/{split}.jsonl').open()]
        expected=set(splits['expert' if split=='train' else 'development']);assert {r['video'] for r in rows}==expected
        splitrows[split]=rows;report['source_manifest_sha256'][split]=sha(old/f'data/{split}.jsonl')
    dets=pickle.load((source/'data/dets_v8x_best_val_fullcand.pkl').open('rb'))['records']
    keys=official['frames'];assert len(keys)==36717
    candidate_hash=hashlib.sha256(b''.join(k.encode()+np.asarray(dets[k]['boxes_xyxyn']).tobytes()+np.asarray(dets[k]['conf']).tobytes()+np.asarray(dets[k]['cls']).tobytes() for k in keys)).hexdigest()
    assert candidate_hash==official['candidate_sha256']
    val=[];hashes={};video_gt=pickle.load((source/'data/dets_i3d_val_fullcand.pkl').open('rb'));assert video_gt['labels']==official['labels'];gts={}
    for k in keys:
        v,f=k.rsplit('_',1);fid=int(f);nf=ann['db'][v]['numf'];fids=[min(max(fid-3+j,1),nf) for j in range(8)]
        hh=[]
        for fj in fids:
            rel=f'{v}/{fj:05d}.jpg'
            if rel not in hashes:hashes[rel]=sha(old/'frames'/rel)
            hh.append(hashes[rel])
        val.append({'video':v,'fid':fid,'fids':fids,'key_t':fids.index(fid),'boxes':dets[k]['boxes_xyxyn'].tolist(),'frame_sha256':hh})
        gts[k]=video_gt['records'][v+'/'+str(fid)]['gt']
    splitrows['val']=val
    # Exact old manifests for training and development. No resampling or label remapping.
    arrays={}
    for split,rows in splitrows.items():
        n=sum(len(r['boxes']) for r in rows);k=len(rows)
        arrays[split]={name:np.lib.format.open_memmap(data/f'{split}-{name}.npy',mode='w+',dtype=dtype,shape=shape) for name,dtype,shape in [('crop','float16',(n,1024)),('boxes','float32',(n,4)),('targets','uint8',(n,184)),('frame','int32',(n,)),('q','float32',(n,)),('agent','int16',(n,))]}
        keys_by_split[split]=[];pos=0
        with (data/f'{split}.jsonl').open('w') as out:
            for i,r in enumerate(rows):
                key=f"{r['video']}_{r['fid']:05d}";assert key not in allrows
                length=len(r['boxes']);end=pos+length
                allrows[key]=(split,pos,end);keys_by_split[split].append(key)
                arrays[split]['boxes'][pos:end]=np.asarray(r['boxes'],np.float32).reshape(-1,4)
                arrays[split]['frame'][pos:end]=i;arrays[split]['targets'][pos:end]=0
                if split!='val':
                    for j,ix in enumerate(r['positive_indices']):arrays[split]['targets'][pos+j,ix]=1
                else:
                    arrays[split]['q'][pos:end]=dets[key]['conf'];arrays[split]['agent'][pos:end]=dets[key]['cls']
                r.update(row_start=pos,row_end=end,frame_index=i);out.write(json.dumps(r,separators=(',',':'))+'\n');pos=end
        report['splits'][split]={'rows':n,'frames':k};print('ALLOCATED',split,n,k,flush=True)
    found=set()
    for split,nshard in [('train',8),('val',4)]:
        for j in range(nshard):
            path=paired/f'crop_feats_{split}.shard{j}of{nshard}.pkl';d=pickle.load(path.open('rb'));assert not d['meta']['partial'];h=hashlib.sha256()
            assert d['meta']['revision']=='1f9fca1389fd883defc652634d95a21121c85a8c' and d['meta']['pad']==2
            for key in sorted(d['boxes']):
                b=d['boxes'][key].astype(np.float32);h.update(key.encode());h.update(b.tobytes())
                if d['targets'] is not None:h.update(d['targets'][key].tobytes())
                if key not in allrows:continue
                assert key not in found;found.add(key);s,start,end=allrows[key]
                assert np.array_equal(b,arrays[s]['boxes'][start:end]),key
                if s!='val':assert np.array_equal(d['targets'][key],arrays[s]['targets'][start:end]),key
                x=np.asarray(d['feats'][key],np.float16);assert x.shape==(end-start,1024) and np.isfinite(x).all()
                arrays[s]['crop'][start:end]=x
            assert h.hexdigest()==d['meta']['row_sha256'];report['shards'][path.name]={'row_sha256':h.hexdigest(),'revision':d['meta']['revision']};del d
            print('JOINED',path.name,len(found),'/',len(allrows),flush=True)
    assert found==set(allrows)
    for aa in arrays.values():
        for a in aa.values():a.flush()
    # GT payload copied as benchmark source, never as training input.
    with (data/'evaluation-gt.pkl').open('wb') as f:pickle.dump(gts,f,pickle.HIGHEST_PROTOCOL)
    for filename in ['phrase_embeds.pt','flat_alphas.pt']:shutil.copyfile(source/'data'/filename,data/filename)
    for filename in ['shared-frames.json','train-counts.json']:shutil.copyfile(source/'code'/filename,data/filename)
    report['candidate_sha256']=candidate_hash;report['data_sha256']={p.name:sha(p) for p in sorted(data.iterdir()) if p.is_file()};report['completed_unix']=time.time();report['passed']=True
    (root/'preparation.json').write_text(json.dumps(report,indent=2));print('PREPARATION_COMPLETE',report['splits'],flush=True)
if __name__=='__main__':main()

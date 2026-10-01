"""Verify staged input hashes and reuse exact immutable parent row metadata."""
from pathlib import Path
import argparse,hashlib,json,time,shutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def atomic(d,p):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
    for _ in range(720):
        if (r/'results/inputs-transferred.json').exists():break
        print('WAITING_FOR_INPUT_TRANSFER',time.time(),flush=True);time.sleep(30)
    transfer=json.loads((r/'results/inputs-transferred.json').read_text());assert transfer['passed']
    assert sha(r/'inputs/frame-hashes.json')==transfer['frame_hashes_sha256']
    for f,h in transfer['gate_sha256'].items():assert sha(r/'inputs/gate'/f)==h,f
    hashes=json.loads((r/'inputs/frame-hashes.json').read_text())
    def check_frame(item):
        rel,h=item;assert sha(r/'frames'/rel)==h,rel
        return rel
    with ThreadPoolExecutor(max_workers=16) as pool:
        for i,_ in enumerate(pool.map(check_frame,hashes.items())):
            if i%5000==0:print('REMOTE_FRAMES_VERIFIED',i,len(hashes),flush=True)
    for file in ['assets.json','benchmark.json','text-benchmark.json']:assert json.loads((r/'results'/file).read_text())['passed']
    assert sha(r/'data/phrase_embeds.pt')==json.loads((r/'results/text-benchmark.json').read_text())['phrase_bank_sha256']
    parent=Path('/work/bbyrd1/contextual-roi-20260917');prep=json.loads((parent/'preparation.json').read_text());assert prep['passed']
    data=r/'data';meta={'splits':{},'source_sha256':{},'split':prep['split'],'passed':False}
    for split in ['train','dev','gate','val']:
        src=parent/'data' if split!='gate' else r/'inputs/gate'
        files=[f'{split}.jsonl']+[f'{split}-{k}.npy' for k in ['boxes','targets','frame']]
        files+= [f'{split}-{k}.npy' for k in (['partition'] if split=='gate' else ['q','agent'])]
        for name in files:
            h=sha(src/name)
            expected=transfer['gate_sha256'][name] if split=='gate' else prep['data_sha256'][name]
            assert h==expected,name
            dest=data/name
            if not dest.exists():dest.symlink_to(src/name)
            assert sha(dest)==h
            meta['source_sha256'][name]=h
        boxes=np.load(data/f'{split}-boxes.npy',mmap_mode='r');targets=np.load(data/f'{split}-targets.npy',mmap_mode='r');indices=np.load(data/f'{split}-frame.npy',mmap_mode='r')
        pos=0;keys=[];videos=set()
        for i,line in enumerate((data/f'{split}.jsonl').open()):
            row=json.loads(line);start,end=row['row_start'],row['row_end'];assert start==pos and row['frame_index']==i
            assert np.array_equal(np.asarray(row['boxes'],np.float32).reshape(-1,4),boxes[start:end])
            assert np.all(indices[start:end]==i);assert len(row['fids'])==8 and len(row['frame_sha256'])==8
            for fid,digest in zip(row['fids'],row['frame_sha256']):assert hashes[f'{row["video"]}/{fid:05d}.jpg']==digest
            videos.add(row['video']);keys.append(f'{row["video"]}_{row["fid"]:05d}');pos=end
        assert targets.shape==(pos,184) and len(boxes)==pos
        expected=prep['split'][{'train':'expert','dev':'development','gate':'gate'}[split]] if split!='val' else {k.rsplit('_',1)[0] for k in keys}
        assert videos==set(expected)
        meta['splits'][split]={'rows':pos,'frames':len(keys),'videos':sorted(videos)}
        if split=='val':
            official=json.loads((parent/'data/shared-frames.json').read_text());assert keys==official['frames'] and len(keys)==36717
        print('VERIFIED_SPLIT',split,pos,len(keys),flush=True)
    for name in ['flat_alphas.pt','shared-frames.json','train-counts.json','evaluation-gt.pkl']:
        assert sha(parent/'data'/name)==prep['data_sha256'][name]
        dest=data/name
        if not dest.exists():dest.symlink_to(parent/'data'/name)
        meta['source_sha256'][name]=sha(dest)
    gateprep=json.loads((r/'inputs/gate/preparation.json').read_text());partition=np.load(data/'gate-partition.npy',mmap_mode='r');select=set()
    for line in (data/'gate.jsonl').open():
        row=json.loads(line);v=partition[row['row_start']:row['row_end']];assert len(v) and np.all(v==v[0])
        if v[0]==1:select.add(row['video'])
    assert select==set(gateprep['selection_videos']) and len(select)==45
    meta.update(passed=True,time=time.time(),frames_sha256=transfer['frame_hashes_sha256'],candidate_sha256=official['candidate_sha256'],frame_sha256=official['frame_sha256'])
    atomic(meta,r/'preparation.json');print('PREPARATION_COMPLETE',flush=True)

if __name__=='__main__':main()

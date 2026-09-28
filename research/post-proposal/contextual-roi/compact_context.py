"""Join context caches with exact verified row ordering; emit mmap arrays."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from cache_context import CONTRACT,digest
from train_cached import file_sha,atomic_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();root=args.root
    prep=json.loads((root/'preparation.json').read_text());report={'passed':False,'files':{},'contract':CONTRACT,'preparation_sha256':file_sha(root/'preparation.json')}
    for split in ['train','dev','val']:
        meta=prep['splits'][split];n=meta['rows'];k=meta['frames'];box=np.load(root/f'data/{split}-boxes.npy',mmap_mode='r')
        context=np.lib.format.open_memmap(root/f'data/{split}-context.npy',mode='w+',dtype='float16',shape=(n,1024));scene=np.lib.format.open_memmap(root/f'data/{split}-scene.npy',mode='w+',dtype='float16',shape=(k,16,1024))
        for i,line in enumerate((root/f'data/{split}.jsonl').open()):
            row=json.loads(line);key=f"{row['video']}_{row['fid']:05d}";d=torch.load(root/f'context/{key}.pt',map_location='cpu',weights_only=True)
            assert d['fingerprint']==digest({'contract':CONTRACT,'row':row});start,end=row['row_start'],row['row_end'];assert i==row['frame_index']
            assert np.array_equal(d['boxes'].numpy(),box[start:end]);assert d['context_roi'].shape==(end-start,1024) and d['scene_tokens'].shape==(16,1024)
            assert torch.isfinite(d['context_roi']).all() and torch.isfinite(d['scene_tokens']).all()
            context[start:end]=d['context_roi'].numpy();scene[i]=d['scene_tokens'].numpy()
            if i%500==0:atomic_json({'phase':'compact','split':split,'frames':i,'time':time.time()},root/'pipeline-progress.json')
        context.flush();scene.flush()
    for p in sorted((root/'data').glob('*.npy')):report['files'][p.name]=file_sha(p)
    report.update(passed=True,time=time.time());atomic_json(report,root/'cache-ready.json')
if __name__=='__main__':main()

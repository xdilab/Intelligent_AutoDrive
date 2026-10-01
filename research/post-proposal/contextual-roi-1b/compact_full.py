"""Validate every cache record before exposing contiguous training arrays."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from prepare_full import sha,atomic
from cache_full import fingerprint

def main():
    torch.set_num_threads(4)
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root;cfg=json.loads((r/'protocol.json').read_text());prep=json.loads((r/'preparation.json').read_text());assert prep['passed']
    receipts=[json.loads((r/f'results/cache-{i}-complete.json').read_text()) for i in range(cfg['extraction_shards'])]
    assert all(v['passed'] and v['contract']==receipts[0]['contract'] for v in receipts)
    contract=receipts[0]['contract'];assert contract['protocol_sha256']==sha(r/'protocol.json') and contract['preparation_sha256']==sha(r/'preparation.json')
    for name,digest in contract['implementation_sha256'].items():assert sha(Path(__file__).parent/name)==digest
    assert sum(v['frames'] for v in receipts)==sum(x['frames'] for x in prep['splits'].values())
    report={'passed':False,'files':{},'contract':contract,'preparation_sha256':sha(r/'preparation.json')}
    for split,meta in prep['splits'].items():
        n,k=meta['rows'],meta['frames'];arrays={}
        for name,shape in [('crop',(n,1408)),('context',(n,1408)),('scene',(k,16,1408))]:
            arrays[name]=np.lib.format.open_memmap(r/f'data/{split}-{name}.npy',mode='w+',dtype='float16',shape=shape)
        boxes=np.load(r/f'data/{split}-boxes.npy',mmap_mode='r');pos=0
        for i,line in enumerate((r/f'data/{split}.jsonl').open()):
            row=json.loads(line);key=f'{row["video"]}_{row["fid"]:05d}';d=torch.load(r/f'cache/{split}/{key}.pt',weights_only=True,map_location='cpu')
            assert d['fingerprint']==fingerprint(row,contract);start,end=row['row_start'],row['row_end'];assert start==pos and i==row['frame_index']
            assert np.array_equal(d['boxes'].numpy(),boxes[start:end])
            assert d['crop'].shape==d['context_roi'].shape==(end-start,1408) and d['scene_tokens'].shape==(16,1408)
            assert all(torch.isfinite(d[x]).all() for x in ['crop','context_roi','scene_tokens'])
            arrays['crop'][start:end]=d['crop'].numpy();arrays['context'][start:end]=d['context_roi'].numpy();arrays['scene'][i]=d['scene_tokens'].numpy();pos=end
            if i%500==0:print('COMPACT',split,i,k,flush=True)
        assert pos==n and i+1==k
        for v in arrays.values():v.flush()
    for p in sorted((r/'data').iterdir()):
        report['files'][p.name]=sha(p);print('ARRAY_VERIFIED',p.name,flush=True)
    report.update(passed=True,time=time.time());atomic(report,r/'cache-ready.json')
    print('CACHE_COMPLETE',flush=True)

if __name__=='__main__':main()

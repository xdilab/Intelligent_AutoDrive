"""Resumable frozen1B crop/RoI/entire-frame caches, exact parent boxes/frames."""
from pathlib import Path
import argparse,hashlib,json,time,sys
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.ops import roi_align
from benchmark_vision import Encoder
from prepare_full import sha,atomic

def fingerprint(row,contract):
    return hashlib.sha256(json.dumps({'row':row,'contract':contract},sort_keys=True,separators=(',',':')).encode()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--shard',type=int,required=True);ap.add_argument('--shards',type=int,default=8);a=ap.parse_args();r=a.root
    assert 0<=a.shard<a.shards;prep=json.loads((r/'preparation.json').read_text());assert prep['passed'];cfg=json.loads((r/'protocol.json').read_text())
    contract={'protocol_sha256':sha(r/'protocol.json'),'preparation_sha256':sha(r/'preparation.json'),
              'implementation_sha256':{p:sha(Path(__file__).parent/p) for p in ['cache_full.py','benchmark_vision.py']}}
    sys.path.insert(0,'/work/bbyrd1/stage56-full-20260914/code');from eval_input import ClipLoader,normalize_clip,crop_batch
    loader=ClipLoader({'frames':str(r/'frames')});loader.root=r/'frames';torch.set_num_threads(8);model=None
    count=0;crops=0;start_time=time.monotonic();total=sum(s['frames'] for s in prep['splits'].values());global_index=0
    for split in ['train','dev','gate','val']:
        out=r/'cache'/split;out.mkdir(parents=True,exist_ok=True)
        for line in (r/f'data/{split}.jsonl').open():
            row=json.loads(line);assigned=global_index%a.shards==a.shard;global_index+=1
            if not assigned:continue
            key=f'{row["video"]}_{row["fid"]:05d}';assert '/' not in key and '..' not in key
            target=out/f'{key}.pt';digest=fingerprint(row,contract);n=len(row['boxes'])
            if target.exists():
                d=torch.load(target,weights_only=True,map_location='cpu');assert d['fingerprint']==digest
                assert d['crop'].shape==d['context_roi'].shape==(n,1408) and d['scene_tokens'].shape==(16,1408)
                assert all(torch.isfinite(d[k]).all() for k in ['crop','context_roi','scene_tokens'])
            else:
                if model is None:model=Encoder(r);print('CACHE_ENCODER_READY',model.audit,flush=True)
                imgs=loader.load(row);ft=normalize_clip(imgs);parts=[]
                # Same established float32 normalization/RoIAlign implementation
                # as the completed runtime benchmark; both new experts share it.
                for j in range(0,n,cfg['extraction_batch_size']):
                    x=crop_batch(ft,row['boxes'][j:j+cfg['extraction_batch_size']]);parts.append(model(x,row['key_t']).mean(1).half().cpu())
                crop=torch.cat(parts) if parts else torch.empty((0,1408),dtype=torch.float16)
                fm=model(F.interpolate(ft,size=(224,224),mode='bilinear',align_corners=False)[None],row['key_t']).reshape(1,16,16,1408).permute(0,3,1,2)
                boxes=torch.tensor(row['boxes'],device='cuda',dtype=torch.float32).reshape(-1,4)
                rois=torch.cat([boxes.new_zeros(len(boxes),1),boxes*16],1)
                context=roi_align(fm,rois,output_size=7,aligned=True).mean((-1,-2)).half().cpu()
                scene=F.adaptive_avg_pool2d(fm,(4,4)).flatten(2).transpose(1,2)[0].half().cpu()
                d={'fingerprint':digest,'crop':crop,'context_roi':context,'scene_tokens':scene,'boxes':boxes.cpu()}
                assert all(torch.isfinite(d[k]).all() for k in ['crop','context_roi','scene_tokens'])
                tmp=target.with_suffix('.tmp');torch.save(d,tmp);tmp.replace(target)
            count+=1;crops+=n
            if count==1 or count%25==0:
                status={'shard':a.shard,'split':split,'frames':count,'crops':crops,'seconds':time.monotonic()-start_time,'time':time.time()}
                atomic(status,r/f'results/cache-{a.shard}-progress.json');print('CACHE_PROGRESS',json.dumps(status),flush=True)
    assert global_index==total
    atomic({'passed':True,'contract':contract,'frames':count,'crops':crops,'shard':a.shard,'shards':a.shards,'time':time.time()},r/f'results/cache-{a.shard}-complete.json')
    print('CACHE_SHARD_COMPLETE',a.shard,count,crops,flush=True)

if __name__=='__main__':main()

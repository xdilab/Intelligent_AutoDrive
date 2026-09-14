"""Verify source shards, then atomically merge into the isolated study."""
import hashlib,json,pickle,sys,time
from pathlib import Path
root=Path(sys.argv[1]);checks=json.loads((root/'code/shard-checksums.json').read_text())
for split,n in [('train',8),('val',4)]:
    merged={'feats':{},'targets':{},'n_boxes':{}};provenance={}
    for i in range(n):
        name=f'crop_feats_{split}.shard{i}of{n}.pkl'
        p=root/'shards'/name if split=='val' and i>=2 else Path('/work/bbyrd1/road_crop/out')/name
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
        assert p.stat().st_size==checks[name]['bytes'] and h.hexdigest()==checks[name]['sha256'],f'Shard checksum mismatch: {p}'
        print(f'VERIFIED {name}',flush=True);d=pickle.load(p.open('rb'))
        assert not d.get('meta',{}).get('partial',False)
        for field in merged:
            value=d.get(field) or {};assert not merged[field].keys() & value.keys(),f'Duplicate {field} keys';merged[field].update(value)
        provenance[name]=checks[name];del d
    if not merged['targets']:merged['targets']=None
    merged['meta']={'split':split,'dim':1024,'verified_shards':provenance}
    p=root/'data'/f'crop_feats_{split}.pkl';temp=p.with_suffix('.verified-partial')
    with temp.open('wb') as f:pickle.dump(merged,f,protocol=pickle.HIGHEST_PROTOCOL)
    temp.replace(p)
    print(f'MERGED {split}: {len(merged["feats"])} frames',flush=True);del merged
(root/'results/preparation.json').write_text(json.dumps({'status':'complete','shards':checks,'completed_unix':time.time()},indent=2)+'\n')

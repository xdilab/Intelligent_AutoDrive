"""Read-only asset audit, deterministic text prototypes, and pre-training freeze."""
import argparse, importlib.util, importlib.metadata, json, pickle, subprocess, sys, time, types
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from common import *
from descriptions import generate

def text_tower():
    # Load the exact publisher text implementation without constructing the unused vision tower.
    package=types.ModuleType('comp765_publisher');package.__path__=[str(SNAPSHOT)]
    sys.modules[package.__name__]=package
    spec=importlib.util.spec_from_file_location('comp765_publisher.mobile_clip',SNAPSHOT/'mobile_clip.py')
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    cfg=read(SNAPSHOT/'config.json')['model']['text_encoder']
    model=mod.TextTransformer(cfg['text_cfg'],cfg['embed_dim'])
    with safe_open(str(INPUTS['text_weights']),framework='pt') as f:
        state={k[len('text_encoder.'):]:f.get_tensor(k) for k in f.keys() if k.startswith('text_encoder.')}
    model.load_state_dict(state,strict=True)
    return model.half().cuda().eval().requires_grad_(False),mod.ClipTokenizer(cfg)

def prototypes(root):
    manifest=generate();save(root/'descriptions.json',manifest)
    model,tokenizer=text_tower();rows=manifest['rows']
    lengths=[];tokens=[]
    for r in rows:
        tok=tokenizer(r['descriptions']);seqs=[tuple(t.tolist()) for t in tok]
        assert len(set(seqs))==len(seqs),f'Token duplicate: {r["id"]}'
        for text,t in zip(r['descriptions'],tok):
            n=len(tokenizer.tokenizer.encode(text))+2
            assert n<=tokenizer.context_length,(r['id'],n)
            lengths.append(dict(id=r['id'],text=text,tokens_including_special=n,truncated=False))
        tokens.append(tok)
    canonical=torch.stack([t[0] for t in tokens])
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        c=model(canonical.cuda()).float().cpu()
        extras=[]
        for variant in range(1,5):
            batch=torch.stack([tokens[0][0]]+[t[variant] for t in tokens[1:]])
            extras.append(model(batch.cuda()).float().cpu())
    description_embeds=F.normalize(torch.stack([c]+extras,1),dim=-1)
    def aggregate(x): return F.normalize(F.normalize(x.float(),dim=-1).mean(1),dim=-1)
    single=aggregate(c[:,None]);multi=aggregate(torch.stack([c]+extras,1));multi[0]=single[0]
    old=torch.load(INPUTS['canonical_embeddings'],weights_only=False)
    order=[r['id'] for r in rows];assert old['order']==order
    delta=(single-old['embeds']).abs().max().item();cos=F.cosine_similarity(single,old['embeds']).min().item()
    tolerance=read(HERE/'protocol.json')['baseline_tolerance']
    assert delta<tolerance['max_abs'] and cos>tolerance['min_cosine'],(delta,cos)
    diagnostics={'canonical_max_abs_error':delta,'canonical_min_cosine':cos,'same_special_row':torch.equal(single[0],multi[0]),'per_class':[]}
    for condition,p in [('single',single),('multi',multi)]:
        assert p.shape==(184,512) and torch.isfinite(p).all()
        assert torch.allclose(p.norm(dim=-1),torch.ones(184),atol=2e-7)
        checkpoint(root/f'prototypes-{condition}.pt',dict(embeds=p,order=order,condition=condition))
        sims=(p@p.T).numpy();np.save(root/f'cosine-{condition}.npy',sims)
        diagnostics[condition]=dict(norm_min=float(p.norm(dim=-1).min()),norm_max=float(p.norm(dim=-1).max()),
            near_pairs=[dict(a=order[i],b=order[j],cosine=float(sims[i,j]),exact=bool(torch.equal(p[i],p[j]))) for i in range(184) for j in range(i) if sims[i,j]>=.999],
            mean_off_diagonal_cosine=float(sims[~np.eye(184,dtype=bool)].mean()))
    for i,r in enumerate(rows):
        diagnostics['per_class'].append(dict(id=r['id'],row=i,canonical_to_aggregate_cosine=float(single[i]@multi[i])))
    checkpoint(root/'description-embeddings.pt',dict(embeds=description_embeds,order=order,special_row_repeated_for_storage_only=True))
    save(root/'tokenization.json',dict(context_length=77,descriptions=lengths,all_distinct_within_class=True,truncation_count=0))
    save(root/'geometry.json',diagnostics)
    del model;torch.cuda.empty_cache()
    progress(root,'prototypes-validated',max_abs_canonical=delta,min_cosine_canonical=cos)

def audit(root):
    provenance={'created':time.time(),'inputs':{},'dependencies':{},'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()}
    for package in ['torch','numpy','scipy','transformers','safetensors','open_clip_torch','torchvision']:
        provenance['dependencies'][package]=importlib.metadata.version(package)
    provenance['python']=sys.version
    for k,p in INPUTS.items():
        s=p.stat();provenance['inputs'][k]=dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=sha(p))
        progress(root,'asset-hash',asset=k)
    source_paths=[OLD/'train-repeats.py',PARENT/'stage56-full/metric.py',PARENT/'stage56-full/baseline-core.py',OLD/'baseline-core.py',E12/'embed_phrases.py',E12/'crop_full/cache_crop_feats_h200.py']+list(SNAPSHOT.glob('*.py'))+[SNAPSHOT/'config.json']
    provenance['sources']={str(p):sha(p) for p in source_paths}
    save(root/'provenance.json',provenance)
    frames=read(INPUTS['frames']);labels=frames['labels'];splits=groups(labels['triplet'])
    counts=read(INPUTS['counts']);v=np.array([r['train_boxes'] for r in counts['rows']]);z=(np.log10(v)-np.log10(v).mean())/np.log10(v).std()
    assert np.allclose(z,[r['z'] for r in counts['rows']],rtol=0,atol=1e-12)
    save(root/'frequency-groups.json',{k:[labels['triplet'][i] for i in ix] for k,ix in splits.items()})
    assert len(frames['frames'])==36717
    assert hashlib.sha256(('\n'.join(frames['frames'])+'\n').encode()).hexdigest()==frames['frame_sha256']
    progress(root,'audit-training-cache')
    d=pickle.load(INPUTS['train'].open('rb'));keys=sorted(d['feats']);n=bad=0
    assert set(keys)==set(d['targets'])
    for k in keys:
        x=d['feats'][k];y=d['targets'][k];assert x.shape[1]==1024 and y.shape==(len(x),184)
        assert np.isfinite(y).all() and ((y==0)|(y==1)).all()
        n+=len(x);bad+=int((~np.isfinite(x)).sum())
    training=dict(rows=n,frames=len(keys),videos=sorted({k.rsplit('_',1)[0] for k in keys}),meta=d.get('meta'),nonfinite_elements_zeroed=bad,keys_sha256=hashlib.sha256(('\n'.join(keys)+'\n').encode()).hexdigest())
    save(root/'training-frames.json',keys);del d
    progress(root,'audit-splits')
    ann=read(INPUTS['annotation'])
    assert all('train' in ann['db'][v]['split_ids'] for v in training['videos'])
    valvideos={k.rsplit('_',1)[0] for k in frames['frames']}
    assert all('val' in ann['db'][v]['split_ids'] for v in valvideos)
    assert not set(training['videos']) & valvideos
    for h in HEADS[1:]: assert labels[h]==ann[h+'_labels']
    del ann
    progress(root,'audit-validation-cache')
    d=pickle.load(INPUTS['val'].open('rb'));yr=pickle.load(INPUTS['yolo'].open('rb'))['records'];gt=pickle.load(INPUTS['gt'].open('rb'))
    assert gt['labels']==labels
    h=hashlib.sha256();n=0;empty=[]
    for k in frames['frames']:
        r=yr[k];boxes=r['boxes_xyxyn'];v,f=k.rsplit('_',1);g=gt['records'][v+'/'+str(int(f))]['gt']
        for b in [k.encode(),np.asarray(boxes).tobytes(),np.asarray(r['conf']).tobytes(),np.asarray(r['cls']).tobytes()]:h.update(b)
        if k not in d['feats']:
            assert len(boxes)==0;empty.append(k)
        else:
            x=d['feats'][k];assert x.shape==(len(boxes),1024) and np.isfinite(x).all()
        if d.get('boxes') and k in d['boxes']: assert np.array_equal(d['boxes'][k],boxes)
        n+=len(boxes)
    assert h.hexdigest()==frames['candidate_sha256']
    save(root/'preflight-assets.json',dict(passed=True,training=training,validation=dict(rows=n,frames=len(frames['frames']),videos=len(valvideos),empty_feature_frames=empty,meta=d.get('meta'),candidate_sha256=h.hexdigest(),frame_sha256=frames['frame_sha256']),tail_counts={k:len(v) for k,v in splits.items()}))
    progress(root,'assets-validated',train_rows=training['rows'],val_rows=n)

def freeze(root):
    assert read(root/'validation.json')['passed']
    assert read(root/'analysis-validation.json')['passed']
    assert read(root/'preflight-assets.json')['passed']
    paths=list(HERE.glob('*.py'))+list(HERE.glob('*.json'))+[HERE/'PROTOCOL.md']
    paths += [root/n for n in ['descriptions.json','prototypes-single.pt','prototypes-multi.pt','description-embeddings.pt','provenance.json','preflight-assets.json','frequency-groups.json','training-frames.json','validation.json','analysis-validation.json','benchmark.json','tokenization.json','geometry.json','cosine-single.npy','cosine-multi.npy']]
    paths += [Path('/data/repos/PedestrianIntent++/ROAD_plus_plus_Baseline/modules')/n for n in ['evaluation.py','utils.py']]
    source=read(root/'provenance.json')['sources']
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    save(root/'freeze.json',dict(time=time.time(),code_revision=revision,files={**{str(p):sha(p) for p in paths},**source},policy='No modification after primary run; no validation-guided tuning'))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prototypes','audit','freeze']);ap.add_argument('--root',type=Path,default=DEFAULT_ROOT);a=ap.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    assert not (a.root/'freeze.json').exists(), 'Study already frozen; use a new output root, never overwrite the frozen run'
    (a.root/'logs').mkdir(exist_ok=True)
    if a.phase=='prototypes':setup();prototypes(a.root)
    elif a.phase=='audit':audit(a.root)
    else:freeze(a.root)

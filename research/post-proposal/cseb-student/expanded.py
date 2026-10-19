"""Expand data only: retain the pilot architecture, losses, and original holdout."""
import argparse,json,time,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import pilot as p


def load_sequences(root):
    m=json.loads((root/'manifest.json').read_text());groups={k:[] for k in m['groups']}
    assert len(m['sequences'])==sum(len(v) for v in m['groups'].values())
    for r in m['sequences']:
        assert p.sha(Path(r['path']))==r['sha256']
        seq=torch.load(r['path'],map_location='cpu',weights_only=True)
        groups[r['split']].append([{k:v.cuda().float() if k not in ['agents','fid'] else v.cuda() if k=='agents' else v for k,v in row.items()} for row in seq])
    return m,groups


def prepare(root,base):
    started=time.perf_counter();root.mkdir(parents=True,exist_ok=True);(root/'cache').mkdir(exist_ok=True)
    p.progress(root,'annotation-loading')
    ann=json.loads((p.ROOT/'road_waymo_trainval_v1.1.json').read_text());prep=json.loads(p.PREP.read_text())
    original=json.loads((base/'manifest.json').read_text());rank=lambda vs:sorted(vs,key=lambda v:p.hashlib.sha256(('cseb-student-pilot-0:'+v).encode()).hexdigest())
    train=rank(prep['videos']['train'])[:64];dev=rank(prep['videos']['dev'])[:24]
    groups={'train':train,'selection':dev[:3]+dev[6:15],'heldout':dev[3:6]+dev[15:24]}
    assert len(set(sum(groups.values(),[])))==88
    for k in groups:assert set(original['groups'][k])<=set(groups[k])
    labels=original['labels']
    for k in labels:
        if k in ['agent','action','loc','duplex','triplet']:assert labels[k]==ann[k+'_labels']
    starts={v:next(f for f in range(40,ann['db'][v]['numf']-6) if all(ann['db'][v]['frames'].get(str(j),{}).get('annotated')==1 for j in range(f,f+8))) for v in sum(groups.values(),[])}
    for v,start in starts.items():
        assert all((p.ROOT/'rgb-images'/v/f'{f:05d}.jpg').exists() for f in range(start-3,start+12)),v
    for path,key in [(p.TEACHER,'teacher_sha256'),(p.WEIGHTS,'backbone_sha256')]:assert p.sha(path)==original[key]
    manifest={k:v for k,v in original.items() if k not in ['sequences','prepare_seconds','groups','start_frames','source_sha256']}
    manifest.update(groups=groups,start_frames=starts,sequences=[],source_sha256={x.name:p.sha(x) for x in Path(__file__).parent.glob('*.py')},
                    expansion={'base_run':str(base),'base_manifest_sha256':p.sha(base/'manifest.json'),'pilot_holdout':original['groups']['heldout'],'change':'data expansion only; architecture/loss/epochs remain fixed'},
                    yolo_sha256=p.sha(p.YOLO),expected_frames=704)
    existing={r['video']:r for r in original['sequences']}
    if (root/'manifest.json').exists():
        old=json.loads((root/'manifest.json').read_text())
        for key in ['groups','start_frames','source_sha256','teacher_sha256','backbone_sha256','annotation_sha256','yolo_sha256']:assert old[key]==manifest[key],key
        existing.update({r['video']:r for r in old['sequences']})
    for split,videos in groups.items():
        for video in videos:
            if video in existing:
                r=existing[video];assert r['split']==split and p.sha(Path(r['path']))==r['sha256']
                seq=torch.load(r['path'],map_location='cpu',weights_only=True);assert [x['fid'] for x in seq]==list(range(starts[video],starts[video]+8))
                manifest['sequences'].append(r)
    p.write_json(root/'manifest.json',manifest)
    completed={r['video'] for r in manifest['sequences']};total=len(completed)*8;times=[]
    settings=root/'yolo-settings';settings.mkdir(exist_ok=True)
    teacher=p.CSEBInference(p.TEACHER,p.ENCODER,p.YOLO,settings);backbone=p.SharedBackbone(p.WEIGHTS).cuda()
    for split,videos in groups.items():
        for video in videos:
            if video in completed:continue
            seq=[];digests={}
            for fid in range(starts[video],starts[video]+8):
                tick=time.perf_counter();imgs=[]
                for f in range(fid-3,fid+5):
                    path=p.ROOT/'rgb-images'/video/f'{f:05d}.jpg'
                    with Image.open(path) as image:imgs.append(np.array(image.convert('RGB')))
                    digests[str(f)]=p.sha(path)
                out=teacher.predict_clip(np.stack(imgs),3);boxes=torch.from_numpy(out['boxes']);agents=torch.from_numpy(out['agents']);q=torch.from_numpy(out['confidence'])
                assert (q>0).all();raw=torch.from_numpy(out['scores']['blend'])/q[:,None];raw[:,0]=1
                assert torch.isfinite(raw).all() and (raw>=0).all() and (raw<=1.00001).all()
                roi,scene=backbone(imgs[3],boxes);target=p.mapped_targets(ann['db'][video]['frames'][str(fid)],ann,labels,out['boxes'],out['agents'])
                seq.append({'fid':fid,'roi':roi.half().cpu(),'scene':scene.half().cpu(),'boxes':boxes,'agents':agents,'q':q,'teacher':raw.clamp(0,1),'labels':target})
                total+=1;times.append(time.perf_counter()-tick)
                p.progress(root,'extracting',frames=total,total=704,video=video,fid=fid,candidates=len(boxes),seconds=times[-1],eta_seconds=(704-total)*float(np.mean(times)))
            path=root/'cache'/f'{video}.pt';p.atomic_save(seq,path)
            manifest['sequences'].append({'video':video,'split':split,'path':str(path),'sha256':p.sha(path),'frame_sha256':digests,'rows':sum(len(x['boxes']) for x in seq)})
            p.write_json(root/'manifest.json',manifest)
    # Preserve the fixed group ordering for held-out arrays and comparisons, including on resume.
    by_video={r['video']:r for r in manifest['sequences']}
    manifest['sequences']=[by_video[v] for vs in groups.values() for v in vs]
    manifest['prepare_seconds']=time.perf_counter()-started;p.write_json(root/'manifest.json',manifest)
    p.progress(root,'prepared',frames=total,seconds=manifest['prepare_seconds'])


def compare(root):
    m,groups=load_sequences(root);old=set(m['expansion']['pilot_holdout'])
    video_order=[r['video'] for r in m['sequences'] if r['split']=='heldout']
    selected=[s for v,s in zip(video_order,groups['heldout']) if v in old]
    assert len(selected)==3
    results={}
    for condition in ['supervised','distilled']:
        ck=torch.load(root/f'{condition}-best.pt',map_location='cuda',weights_only=False);model=p.Student().cuda().eval();model.load_state_dict(ck['model'])
        results[condition],arrays=p.evaluate(model,selected)
    results['teacher']=p.metrics(arrays['teacher'],arrays['labels'],arrays['teacher'])
    results['pilot_reference']=json.loads((Path(m['expansion']['base_run'])/'results.json').read_text())
    results['scope']='Same original3student-heldout videos/24frames,550proposals; no new checkpoint selection.'
    p.write_json(root/'original-holdout-comparison.json',results)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','train','benchmark']);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--base',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(8)
    try:
        if a.phase=='prepare':prepare(a.root,a.base)
        elif a.phase=='train':p.load_sequences=load_sequences;p.train(a.root);compare(a.root)
        else:p.benchmark(a.root)
    except Exception as exc:p.progress(a.root,'failed',operation=a.phase,error=str(exc),traceback=p.traceback.format_exc());raise
if __name__=='__main__':main()

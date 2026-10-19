"""Bounded student pilot. Official validation and teacher checkpoints are read-only."""
import argparse,hashlib,json,os,sys,time,random,traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_iou
from model import SharedBackbone,Student,objective
DEMO=Path(__file__).resolve().parents[1]/'demo'
sys.path.insert(0,str(DEMO))
from deployment import CSEBInference
from predict import detector_scores
from export_sample import sha
from infer_video import write_json

ROOT=Path('/data/datasets/road_waymo')
PREP=Path('/data/repos/wiki/artifacts/stage56-full/collected/preparation.json')
WEIGHTS=Path('/home/brandon/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth')
TEACHER=Path('/home/brandon/.local/share/road-reason/contextual-roi-1b-20260930/attention-global-dcb-seed0/best.pt')
YOLO=Path('/data/repos/ROAD_Reason/experiments/exp11_yolo/runs/v8x_agent_1280_eccv/weights/best.pt')
ENCODER=Path('/home/brandon/.local/share/road-reason/cseb-demo-runtime')

def average_precision_score(y,p):
    order=np.argsort(-p,kind='stable');y=np.asarray(y)[order];p=np.asarray(p)[order]
    ends=np.r_[np.flatnonzero(np.diff(p)),len(p)-1]
    tp=np.cumsum(y)[ends];precision=tp/(ends+1);recall=tp/y.sum()
    return float(np.sum(np.diff(np.r_[0.,recall])*precision))

def progress(root,phase,**data):
    value={'time':time.time(),'phase':phase,**data};write_json(root/'progress.json',value)
    print(json.dumps(value),flush=True)

def atomic_save(value,path):
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)

def mapped_targets(frame,annotation,labels,boxes,agents):
    offsets={'agent':1,'action':11,'loc':33,'duplex':49,'triplet':98}
    gt=[];ys=[];aids=[]
    for item in frame.get('annos',{}).values():
        y=np.zeros(184,np.float32);valid_agents=[]
        for kind,offset in offsets.items():
            for idx in item.get(kind+'_ids',[]):
                name=annotation['all_'+kind+'_labels'][idx]
                if name in labels[kind]:
                    k=labels[kind].index(name);y[offset+k]=1
                    if kind=='agent':valid_agents.append(k)
        if not valid_agents:continue
        y[0]=1;gt.append(item['box']);ys.append(y);aids.append(valid_agents)
    target=np.zeros((len(boxes),184),np.float32)
    if gt and len(boxes):
        iou=box_iou(torch.as_tensor(boxes),torch.tensor(gt,dtype=torch.float32)).numpy()
        for i,agent in enumerate(agents):
            allowed=np.array([int(agent) in x for x in aids]);v=np.where(allowed,iou[i],-1);j=int(v.argmax())
            if v[j]>=.5:target[i]=ys[j]
    return torch.from_numpy(target)

def prepare(root):
    started=time.perf_counter();progress(root,'annotation-loading')
    ann=json.loads((ROOT/'road_waymo_trainval_v1.1.json').read_text())
    prep=json.loads(PREP.read_text());rank=lambda vs:sorted(vs,key=lambda v:hashlib.sha256(('cseb-student-pilot-0:'+v).encode()).hexdigest())
    train=rank(prep['videos']['train'])[:8];dev=rank(prep['videos']['dev'])[:6]
    groups={'train':train,'selection':dev[:3],'heldout':dev[3:]}
    assert len(set(train+dev))==14 and not set(train)&set(prep['videos']['dev'])
    starts={v:next(f for f in range(40,ann['db'][v]['numf']-6) if all(ann['db'][v]['frames'].get(str(j),{}).get('annotated')==1 for j in range(f,f+8))) for v in train+dev}
    labels=json.loads((DEMO/'labels.json').read_text())
    for k in ['agent','action','loc','duplex','triplet']:assert labels[k]==ann[k+'_labels']
    manifest={'groups':groups,'start_frames':starts,'selection':'fixed SHA256 video ranking; earliest fully annotated8-frame window starting>=40; no predictions used',
      'teacher':str(TEACHER),'teacher_sha256':sha(TEACHER),'backbone':str(WEIGHTS),'backbone_sha256':sha(WEIGHTS),
      'partition_sha256':sha(PREP),'annotation_sha256':sha(ROOT/'road_waymo_trainval_v1.1.json'),
      'labels':labels,'detector':{'confidence':.25,'iou':.7,'imgsz':1280,'max_det':300},
      'student':'frozen ImageNet ResNet50 through layer3,448x672; ROI mean; scene GRU256;184-score head',
      'temporal':'student current/past only; teacher keyframe-3..+4; state resets at each8-frame segment',
      'training':{'epochs':30,'seed':0,'lr':.001,'weight_decay':.01,'kd_weight':1.,'selection':'minimum supervised selection BCE; one checkpoint per condition'},
      'metrics':'proposal/crop AP on positive-support classes, not official detector AP; heldout videos drawn from development only',
      'source_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},'sequences':[]}
    write_json(root/'manifest.json',manifest)
    settings=root/'yolo-settings';settings.mkdir();teacher=CSEBInference(TEACHER,ENCODER,YOLO,settings)
    backbone=SharedBackbone(WEIGHTS).cuda();total=0;times=[]
    for split,videos in groups.items():
        for video in videos:
            seq=[];digests={};v=ann['db'][video]
            for fid in range(starts[video],starts[video]+8):
                frame=v['frames'][str(fid)];assert frame.get('annotated')==1
                fids=list(range(fid-3,fid+5));imgs=[];start=time.perf_counter()
                for f in fids:
                    p=ROOT/'rgb-images'/video/f'{f:05d}.jpg'
                    with Image.open(p) as im:imgs.append(np.array(im.convert('RGB')))
                    digests[str(f)]=sha(p)
                out=teacher.predict_clip(np.stack(imgs),3)
                boxes=torch.from_numpy(out['boxes']);agents=torch.from_numpy(out['agents']);q=torch.from_numpy(out['confidence'])
                assert (q>0).all()
                raw=torch.from_numpy(out['scores']['blend'])/q[:,None];raw[:,0]=1
                assert torch.isfinite(raw).all() and (raw>=0).all() and (raw<=1.00001).all()
                roi,scene=backbone(imgs[3],boxes)
                labels_gt=mapped_targets(frame,ann,labels,out['boxes'],out['agents'])
                seq.append({'fid':fid,'roi':roi.half().cpu(),'scene':scene.half().cpu(),'boxes':boxes,'agents':agents,
                            'q':q,'teacher':raw.clamp(0,1),'labels':labels_gt})
                total+=1;times.append(time.perf_counter()-start)
                progress(root,'extracting',frames=total,total=112,video=video,fid=fid,candidates=len(boxes),seconds=times[-1],eta_seconds=(112-total)*float(np.mean(times)))
            path=root/'cache'/f'{video}.pt';atomic_save(seq,path)
            manifest['sequences'].append({'video':video,'split':split,'path':str(path),'sha256':sha(path),'frame_sha256':digests,'rows':sum(len(x['boxes']) for x in seq)})
            write_json(root/'manifest.json',manifest)
    manifest['prepare_seconds']=time.perf_counter()-started;write_json(root/'manifest.json',manifest)
    progress(root,'prepared',frames=total,seconds=manifest['prepare_seconds'])

def load_sequences(root):
    manifest=json.loads((root/'manifest.json').read_text());groups={k:[] for k in manifest['groups']}
    assert len(manifest['sequences'])==14
    for r in manifest['sequences']:
        assert sha(Path(r['path']))==r['sha256']
        seq=torch.load(r['path'],map_location='cpu',weights_only=True)
        groups[r['split']].append([{k:v.cuda().float() if k not in ['agents','fid'] else v.cuda() if k=='agents' else v for k,v in row.items()} for row in seq])
    return manifest,groups

def sequence(model,seq):
    state=None;outputs=[]
    for row in seq:
        logits,state=model(row['roi'],row['scene'],row['boxes'],row['agents'],state)
        outputs.append(logits)
    return torch.cat(outputs),torch.cat([r['labels'] for r in seq]),torch.cat([r['teacher'] for r in seq])

@torch.no_grad()
def evaluate(model,sequences):
    model.eval();pred=[];target=[];teacher=[];weighted=[];tweighted=[]
    for seq in sequences:
        logits,y,t=sequence(model,seq);p=logits.sigmoid();q=torch.cat([r['q'] for r in seq])
        pred.append(p);target.append(y);teacher.append(t);weighted.append(detector_scores(p,q));tweighted.append(detector_scores(t,q))
    p,y,t=[torch.cat(xs).cpu().numpy() for xs in [pred,target,teacher]]
    w,tw=[torch.cat(xs).cpu().numpy() for xs in [weighted,tweighted]]
    result=metrics(w,y,tw);result['bce']=float(-(y[:,1:]*np.log(p[:,1:].clip(1e-7,1-1e-7))+(1-y[:,1:])*np.log((1-p[:,1:]).clip(1e-7,1))).mean())
    return result,{'scores':w,'labels':y,'teacher':tw}

def metrics(p,y,t):
    report={'proposals':len(y),'matched':int(y[:,0].sum()),'teacher_mae':float(np.abs(p[:,1:]-t[:,1:]).mean()),'triplet_top1_agreement':float((p[:,98:].argmax(1)==t[:,98:].argmax(1)).mean())}
    for name,lo,hi in [('action',11,33),('loc',33,49),('duplex',49,98),('triplet',98,184)]:
        support=y[:,lo:hi].sum(0)>0
        report[name]={'supported_classes':int(support.sum()),'total_classes':hi-lo,
            'proposal_mAP':float(np.mean([average_precision_score(y[:,lo+j],p[:,lo+j]) for j in np.flatnonzero(support)])) if support.any() else None}
    return report

def train(root):
    started=time.perf_counter();manifest,groups=load_sequences(root);reports={}
    for condition,kd in [('supervised',0.),('distilled',1.)]:
        random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        model=Student().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);best=float('inf');history=[]
        for epoch in range(30):
            model.train();losses=[]
            for i in np.random.permutation(len(groups['train'])):
                seq=groups['train'][i];opt.zero_grad(set_to_none=True);logits,y,t=sequence(model,seq)
                if len(logits)==0:continue
                loss=objective(logits,y,t,kd);assert torch.isfinite(loss);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(norm)
                opt.step();losses.append(float(loss.detach()))
            selection,_=evaluate(model,groups['selection'])
            record={'epoch':epoch+1,'loss':float(np.mean(losses)),'selection_bce':selection['bce']};history.append(record)
            ck={'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'condition':condition,
                'manifest_sha256':sha(root/'manifest.json'),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate()}
            if selection['bce']<best:
                best=selection['bce'];atomic_save(ck,root/f'{condition}-best.pt')
            atomic_save(ck,root/f'{condition}-last.pt')
            progress(root,'training',condition=condition,**record,elapsed_seconds=time.perf_counter()-started)
        ck=torch.load(root/f'{condition}-best.pt',map_location='cuda',weights_only=False);model.load_state_dict(ck['model'])
        result,arrays=evaluate(model,groups['heldout']);result['selected_epoch']=ck['epoch'];result['trainable_parameters']=sum(p.numel() for p in model.parameters())
        reports[condition]=result;np.savez_compressed(root/f'{condition}-heldout.npz',**arrays)
        write_json(root/f'{condition}-history.json',history)
        # Fresh load verification and a compact checkpoint receipt, no automatic promotion.
        assert all(torch.isfinite(v).all() for v in ck['model'].values())
        write_json(root/f'{condition}-checkpoint-verified.json',{'time':time.time(),'sha256':sha(root/f'{condition}-best.pt'),'finite':True,'epoch':ck['epoch']})
    y=arrays['labels'];teacher=arrays['teacher'];reports['teacher']=metrics(teacher,y,teacher)
    # Constant per-class teacher mean uses training data only; weighted once by heldout detector confidence.
    mean=torch.cat([r['teacher'] for seq in groups['train'] for r in seq]).mean(0)
    q=torch.cat([r['q'] for seq in groups['heldout'] for r in seq]);prior=detector_scores(mean[None].expand(len(q),-1),q).cpu().numpy()
    reports['training_teacher_mean']=metrics(prior,y,teacher)
    reports['training_seconds']=time.perf_counter()-started
    write_json(root/'results.json',reports);progress(root,'trained',seconds=reports['training_seconds'])

@torch.inference_mode()
def benchmark(root):
    manifest=json.loads((root/'manifest.json').read_text());model=Student().cuda().eval()
    ck=torch.load(root/'distilled-best.pt',map_location='cuda',weights_only=False);model.load_state_dict(ck['model'])
    backbone=SharedBackbone(WEIGHTS).cuda();teacher=CSEBInference(TEACHER,ENCODER,YOLO,root/'yolo-settings')
    records=[]
    for video in manifest['groups']['heldout']:
        state=None
        first=manifest['start_frames'][video]
        for fid in range(first,first+4):
            images=np.stack([np.array(Image.open(ROOT/'rgb-images'/video/f'{f:05d}.jpg').convert('RGB')) for f in range(fid-3,fid+5)])
            torch.cuda.synchronize();start=time.perf_counter();ref=teacher.predict_clip(images,3);torch.cuda.synchronize();teacher_s=time.perf_counter()-start
            # YOLO repeated independently with identical settings, included in student timing.
            torch.cuda.synchronize();start=time.perf_counter()
            det=teacher.detector.predict(images[3][:,:,::-1].copy(),imgsz=1280,conf=.25,iou=.7,max_det=300,device=0,half=True,verbose=False)[0]
            boxes=det.boxes.xyxyn.float();agents=det.boxes.cls.long();q=det.boxes.conf.float()
            roi,scene=backbone(images[3],boxes);logits,state=model(roi,scene,boxes,agents,state)
            scores=detector_scores(logits.sigmoid(),q).cpu().numpy();torch.cuda.synchronize();student_s=time.perf_counter()-start
            assert np.array_equal(ref['boxes'],boxes.cpu().numpy());assert np.isfinite(scores).all()
            rec={'video':video,'fid':fid,'proposals':len(boxes),'teacher_seconds':teacher_s,'student_seconds':student_s,'warmup':fid==first}
            records.append(rec);progress(root,'latency',**rec)
    warm=[r for r in records if not r['warmup']]
    result={'records':records,'teacher_median_seconds':float(np.median([r['teacher_seconds'] for r in warm])),
        'student_median_seconds':float(np.median([r['student_seconds'] for r in warm])),
        'student_backbone_parameters':sum(p.numel() for p in backbone.parameters()),'student_head_parameters':sum(p.numel() for p in model.parameters()),
        'scope':'Paired GPU pipeline with YOLO and CPU score export; excludes disk decode/loading; first frame per video excluded as warmup;12 frames.'}
    result['median_speedup']=result['teacher_median_seconds']/result['student_median_seconds']
    write_json(root/'latency.json',result);progress(root,'complete',**{k:v for k,v in result.items() if k!='records'})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','train','benchmark']);ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    torch.set_num_threads(8)
    if a.phase=='prepare':a.root.mkdir(exist_ok=False,parents=True);(a.root/'cache').mkdir()
    try:globals()[a.phase](a.root)
    except Exception as e:
        progress(a.root,'failed',operation=a.phase,error=str(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()

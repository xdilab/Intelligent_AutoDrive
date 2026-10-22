"""Scientific preflight and representative throughput, without reporting benchmark AP."""
import argparse, ast, copy, multiprocessing as mp, resource, shutil, tempfile, time
import numpy as np
import torch
import torch.nn.functional as F
from common import *
import train
import evaluate as ev

def validation(root):
    matrices={c:torch.load(root/f'prototypes-{c}.pt',weights_only=False)['embeds'] for c in ['single','multi']}
    assert torch.equal(matrices['single'][0],matrices['multi'][0])
    results=[]
    for seed in [0,1,2]:
        m,_=train.initial(matrices['single'],seed);n,_=train.initial(matrices['multi'],seed)
        assert train.initial_hash(m)==train.initial_hash(n)
        assert sum(v.numel() for v in m.parameters())==524985
        results.append(dict(seed=seed,initial_hash=train.initial_hash(m),parameters=524985))
    # Resume exactness uses actual study training routine and actual focal alphas.
    x=torch.randn((32769,1024),generator=torch.Generator().manual_seed(803));y=(torch.rand((32769,184),generator=torch.Generator().manual_seed(804))>.95).float()
    # Fresh validation directories ensure stale smoke runs cannot count as passing.
    smoke=Path(tempfile.mkdtemp(prefix='validation-',dir=root))
    m,_=train.fit(root,'single',0,x,y,dest_override=smoke/'continuous')
    expected={k:v.detach().clone() for k,v in m.state_dict().items()}
    train.fit(root,'single',0,x,y,stop_after=2,dest_override=smoke/'resumed')
    n,_=train.fit(root,'single',0,x,y,dest_override=smoke/'resumed')
    assert all(torch.equal(v,n.state_dict()[k]) for k,v in expected.items()),'Resume diverged'
    with torch.no_grad():
        raw=n(x[:16].cuda()).sigmoid().cpu().numpy()
    assert raw.shape==(16,184) and np.isfinite(raw).all()
    multi,optimizer=train.initial(matrices['multi'],0)
    alpha=torch.load(INPUTS['alphas'],weights_only=True).float().cuda()
    loss=focal(multi(x[:512].cuda()),y[:512].cuda(),alpha)
    assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();optimizer.step()
    assert torch.isfinite(multi(x[:16].cuda())).all()
    q=np.linspace(.01,.99,16,dtype=np.float32);scores=ev.assemble(raw,q)
    assert np.array_equal(scores[:,0],q) and np.array_equal(scores[:,11:],raw[:,11:]*q[:,None])
    # Exact AST verification against the published baseline evaluator used in prior studies.
    original=Path('/data/repos/PedestrianIntent++/ROAD_plus_plus_Baseline/modules')
    all_functions={}
    for p in [original/'evaluation.py',original/'utils.py']:
        all_functions.update({node.name:ast.dump(node,include_attributes=False) for node in ast.parse(p.read_text()).body if isinstance(node,ast.FunctionDef)})
    checked=[]
    for node in ast.parse((PARENT/'stage56-full/baseline-core.py').read_text()).body:
        if isinstance(node,ast.FunctionDef):
            assert ast.dump(node,include_attributes=False)==all_functions[node.name],f'AP source differs: {node.name}'
            checked.append(node.name)
    # Missed GT, duplicate detections, class filtering, and empty-frame behavior.
    metric=ev.metric
    box=np.array([[.1,.1,.5,.5],[.12,.12,.52,.52],[.6,.6,.9,.9]],np.float32)
    metric.BOXES=[box,np.empty((0,4),np.float32)];metric.AGENTS=[np.array([0,0,1]),np.array([],dtype=int)]
    metric.SCORES=[np.tile(np.array([.9,.8,.7],np.float32)[:,None],(1,184)),np.empty((0,184),np.float32)]
    gt=np.c_[box[:1],np.zeros(1,np.float32)];missed=np.array([[.1,.1,.4,.4,0]],np.float32)
    metric.GTS=[[gt,missed] for _ in HEADS]
    g,c,ap,_=metric.one_class((5,0,'test'));assert abs(ap-50)<1e-8
    # Same exact reference score row construction, independently evaluated.
    reference=metric.core['evaluate_detections']([gt,missed],[[np.c_[box,np.array([.9,.8,.7],np.float32)],np.empty((0,5),np.float32)]],['test'],.5)[1][0]
    assert ap==reference
    metric.SCORES=metric.BOXES=metric.AGENTS=metric.GTS=None
    save(root/'validation.json',dict(passed=True,time=time.time(),paired_initializations=results,resume_exact=True,resume_mid_epoch=True,both_conditions_finite_loss_predictions=True,score_assembly_exact=True,ap_source_ast_equal=checked,synthetic_missed_gt_duplicate_ap=ap,shapes=[184,512],smoke_path=str(smoke)))
    analysis_validation(root)
    progress(root,'validation-passed')

def analysis_validation(root):
    import analyze, contextlib, io
    fixture=Path(tempfile.mkdtemp(prefix='comp765-analysis-test-'))
    save(fixture/'freeze.json',{});manifest=read(INPUTS['frames'])
    for condition in ['single','multi']:
        for seed in [0,1,2]:
            value=10+(seed if condition=='multi' else 0)
            save(fixture/'runs'/f'{condition}-seed{seed}'/'detector-results.json',dict(frame_sha256=manifest['frame_sha256'],candidate_sha256=manifest['candidate_sha256'],n_frames=36717,selected_epoch=10,signature={'freeze':sha(fixture/'freeze.json')},summary={h:value for h in HEADS},tail={h:{'mAP':value} for h in ['tail47','deep28','common39']},per_class_AP={'triplet':[value]*86}))
    with contextlib.redirect_stdout(io.StringIO()):analyze.main(fixture)
    r=read(fixture/'analysis/comparison.json');d=r['metrics']['tail47']['paired_delta']
    assert d['values']==[0,1,2] and d['mean']==1 and d['sample_sd']==1
    assert abs(d['ci95'][0]-(-1.4841377117))<1e-8
    assert r['class_counts']['tail47']=={'improved':47,'declined':0,'tied':0}
    assert len((fixture/'analysis/per-class-triplet.csv').read_text().splitlines())==87
    save(root/'analysis-validation.json',dict(passed=True,fixture=str(fixture),known_paired_deltas=[0,1,2],expected_ci95=d['ci95'],all86_classes_exported=True))

def benchmark(root):
    x,y=train.load_training(root);p=torch.load(root/'prototypes-single.pt',weights_only=False)['embeds'];m,opt=train.initial(p,0)
    alpha=torch.load(INPUTS['alphas'],weights_only=True).float().cuda();order=torch.randperm(len(x),generator=torch.Generator().manual_seed(0));times=[]
    torch.cuda.reset_peak_memory_stats()
    for step in range(34):
        torch.cuda.synchronize();start=time.monotonic();idx=order[step*16384:(step+1)*16384]
        loss=focal(m(x[idx].cuda()),y[idx].cuda(),alpha);assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();opt.step();torch.cuda.synchronize()
        if step>=4:times.append(time.monotonic()-start)
    training=dict(measured_steps=len(times),batch_size=16384,mean_step_seconds=float(np.mean(times)),median_step_seconds=float(np.median(times)),estimated_one_run_seconds=float(np.mean(times)*math_ceil(len(x)/16384)*10),peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    del m,opt,x,y;torch.cuda.empty_cache()
    x,q,offsets,official=ev.load_validation(root);m,_=train.initial(p,0);m.eval()
    scores,inference_seconds=ev.infer(root,m,x,q,offsets,'benchmark-untrained-no-AP-reported');del m;torch.cuda.empty_cache()
    # Representative full-population classes: common/tail/deep triplet, action, loc, duplex.
    labels=official['labels'];gg=groups(labels['triplet'])
    tasks=[(5,gg[n][0],labels['triplet'][gg[n][0]]) for n in ['common39','tail47','deep28']]
    tasks += [(2,3,labels['action'][3]),(3,0,labels['loc'][0]),(4,0,labels['duplex'][0])]
    start=time.monotonic()
    with mp.get_context('fork').Pool(6) as pool:results=list(pool.imap_unordered(ev.one_class,tasks))
    elapsed=time.monotonic()-start
    durations=[dict(group=HEADS[g],class_id=c,seconds=sec) for g,c,_ap,_text,sec in results]
    # Discard all benchmark AP; these are initialized weights, not study results.
    estimated_metric=sum(r['seconds'] for r in durations)/len(durations)*184/8
    out=dict(time=time.time(),training=training,validation_rows=len(x),inference_seconds=inference_seconds,detector_class_timings=durations,benchmark_wall_seconds=elapsed,estimated_evaluation_seconds=estimated_metric+inference_seconds,peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Untrained runtime calibration only. No AP values retained. Six representative full-population classes;8 workers planned; class variability makes ETA provisional.')
    save(root/'benchmark.json',out);progress(root,'benchmark-complete',training_seconds=training['estimated_one_run_seconds'],evaluation_seconds=out['estimated_evaluation_seconds'])

def math_ceil(x):
    import math
    return math.ceil(x)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['validation','benchmark']);ap.add_argument('--root',type=Path,default=DEFAULT_ROOT);a=ap.parse_args();setup()
    if a.phase=='validation':validation(a.root)
    else:benchmark(a.root)

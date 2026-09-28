"""Reserved-video preparation, two-lane extraction, frozen expert routing and evaluation."""
from pathlib import Path
import concurrent.futures,json,os,shutil,subprocess,sys,time,threading,statistics,math
from common import ROOT,BASE,CODE,atomic_json,file_sha
STOP=threading.Event()
def status(phase,**kw):atomic_json({'phase':phase,'time':time.time(),'pid':os.getpid(),**kw},ROOT/'pipeline-progress.json')
def command(args,name,gpu=None):
 if STOP.is_set():raise RuntimeError('Sibling failed; preserved state')
 env=os.environ.copy();env.update(OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1')
 if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
 with (ROOT/'logs'/f'{name}.log').open('a') as out:
  p=subprocess.Popen(args,stdout=out,stderr=subprocess.STDOUT,env=env)
  with (ROOT/'processes.jsonl').open('a') as f:f.write(json.dumps({'pid':p.pid,'command':args,'gpu':gpu,'log':str(ROOT/'logs'/f'{name}.log'),'time':time.time()})+'\n')
  while p.poll() is None:
   if STOP.is_set():p.terminate();p.wait(timeout=1800);raise RuntimeError('Stopped sibling safely')
   (ROOT/'supervisor-heartbeat').touch();time.sleep(5)
  if p.returncode:STOP.set();raise RuntimeError(f'{name} exited {p.returncode}')
def parallel(tasks):
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  jobs=[pool.submit(f,*args) for f,args in tasks]
  for f in concurrent.futures.as_completed(jobs):
   try:f.result()
   except BaseException:STOP.set();raise

def compact():
 import numpy as np,torch
 sys.path.insert(0,str(CODE.parent/'contextual-roi'));from cache_context import CONTRACT,digest
 prep=json.loads((ROOT/'preparation.json').read_text());n=prep['rows'];nf=prep['frames'];context=np.lib.format.open_memmap(ROOT/'data/gate-context.npy',mode='w+',dtype='float16',shape=(n,1024));scene=np.lib.format.open_memmap(ROOT/'data/gate-scene.npy',mode='w+',dtype='float16',shape=(nf,16,1024));boxes=np.load(ROOT/'data/gate-boxes.npy',mmap_mode='r')
 for i,line in enumerate((ROOT/'data/gate.jsonl').open()):
  row=json.loads(line);d=torch.load(ROOT/f"context/{row['video']}_{row['fid']:05d}.pt",map_location='cpu',weights_only=True);assert d['fingerprint']==digest({'contract':CONTRACT,'row':row});start,end=row['row_start'],row['row_end'];assert np.array_equal(d['boxes'].numpy(),boxes[start:end]);assert torch.isfinite(d['context_roi']).all() and torch.isfinite(d['scene_tokens']).all();context[start:end]=d['context_roi'];scene[i]=d['scene_tokens']
  if i%500==0:status('compact-gate-context',frames=i,total=nf)
 context.flush();scene.flush();atomic_json({'passed':True,'preparation_sha256':file_sha(ROOT/'preparation.json'),'files':{p.name:file_sha(p) for p in (ROOT/'data').glob('gate-*npy')},'time':time.time()},ROOT/'cache-ready.json')
def compare():
 from common import parents
 metrics=['triplet','tail47','deep28','common39','action','loc','duplex'];raw=[];cfg=json.loads((ROOT/'protocol.json').read_text())
 def val(d,k):return d['summary'][k] if k in d['summary'] else d['tail'][k]['mAP']
 for seed in cfg['seeds']:
  runs={}
  for kind in ['global','class','generic','factorized']:runs[kind]=json.loads((ROOT/'runs'/f'seed{seed}'/f'{kind}-detector-results.json').read_text())
  for name,p in zip(['classification','contrastive'],parents(seed)):runs[name]=json.loads((p/'detector-results.json').read_text())
  assert len({(r['frame_sha256'],r['candidate_sha256']) for r in runs.values()})==1
  raw.append({'seed':seed,'metrics':{name:{k:val(d,k) for k in metrics} for name,d in runs.items()}})
 comps={}
 for ref in ['generic','global','class','classification','contrastive']:
  comps[ref]={}
  for k in metrics:
   values=[r['metrics']['factorized'][k]-r['metrics'][ref][k] for r in raw];mean=statistics.mean(values);sd=statistics.stdev(values);delta=4.30265273*sd/math.sqrt(3);comps[ref][k]={'per_seed':values,'mean':mean,'sample_sd':sd,'ci95_unadjusted':[mean-delta,mean+delta]}
 atomic_json({'passed':True,'primary':'factorized minus generic, tail47','raw':raw,'factorized_minus':comps,'caveats':['n=3, validation-informed exploratory follow-up','All learned variants fit45 reserved videos and select on separate45; experts frozen','Pairwise ranking is an AP surrogate, not exact detector AP optimization','Both experts use language attention; no isolated language attribution']},ROOT/'comparison.json')
def main():
 ROOT.mkdir(exist_ok=True,parents=True);(ROOT/'logs').mkdir(exist_ok=True)
 try:
  assert shutil.disk_usage(ROOT).free>12*1024**3
  if not (ROOT/'export/export.json').exists():
   status('waiting-gate-export',job='735058')
   while True:
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','ncshare','test -f /work/bbyrd1/contextual-gate-20260919/export.json'])
    if p.returncode==0:break
    state=subprocess.check_output(['ssh','-o','BatchMode=yes','ncshare','sacct -X -n -P -j 735058 --format=State'],text=True)
    if any(k in state for k in ['FAILED','CANCELLED','OUT_OF_MEMORY','TIMEOUT']):raise RuntimeError('Gate export failed: '+state)
    status('waiting-gate-export',job='735058');time.sleep(30)
   status('download-gate-export');(ROOT/'export').mkdir(exist_ok=True)
   command(['rsync','-rt','--partial','--timeout=180','-e','ssh -o BatchMode=yes','--include=gate-*.npz','--include=gate-*.json','--include=export.json','--exclude=*','ncshare:/work/bbyrd1/contextual-gate-20260919/',str(ROOT/'export')+'/'],'download')
  if not (ROOT/'preparation.json').exists():status('prepare-heldout-gate');command([sys.executable,str(CODE/'prepare_local.py')],'prepare')
  if not (ROOT/'cache-ready.json').exists():
   extractor=CODE.parent/'contextual-roi/cache_context.py';baseargs=[sys.executable,str(extractor),'--manifest',str(ROOT/'data/gate.jsonl'),'--frames','/data/datasets/ROAD_plusplus/rgb-images','--output',str(ROOT/'context'),'--shards','2']
   if not (ROOT/'extraction-benchmark.json').exists():
    status('benchmark-gate-context');start=time.time();parallel([(command,(baseargs+['--shard',str(g),'--limit','20'],f'benchmark-gpu{g}',g)) for g in [0,1]]);elapsed=time.time()-start;prep=json.loads((ROOT/'preparation.json').read_text());estimate=prep['frames']*elapsed/40
    atomic_json({'time':time.time(),'frames':40,'seconds':elapsed,'remaining_extraction_estimate_seconds':estimate,'note':'Includes startup; not full gate study ETA'},ROOT/'extraction-benchmark.json')
   status('extract-gate-context');parallel([(command,(baseargs+['--shard',str(g)],f'context-gpu{g}',g)) for g in [0,1]]);compact()
  # Production preflight must pass before any gate fitting/evaluation.
  command([sys.executable,str(CODE/'preflight.py')],'preflight',0)
  while not (ROOT/'review-clearance.json').exists():status('waiting-code-review');time.sleep(30)
  clearance=json.loads((ROOT/'review-clearance.json').read_text());assert clearance['passed']
  for name,digest in clearance['code_sha256'].items():assert file_sha(CODE/name)==digest
  status('fit-and-evaluate-gates')
  def worker(gpu,seeds):
   for seed in seeds:
    command([sys.executable,str(CODE/'predict_gate.py'),'--seed',str(seed)],f'predict-seed{seed}',gpu)
    command([sys.executable,str(CODE/'train_gate.py'),'--seed',str(seed)],f'train-seed{seed}',gpu)
    for kind in ['global','class','generic','factorized']:command([sys.executable,str(CODE/'evaluate_gate.py'),'--seed',str(seed),'--kind',kind],f'eval-{kind}-seed{seed}',gpu)
  parallel([(worker,(0,[0,2])),(worker,(1,[1]))]);compare();results=[json.loads(p.read_text()) for p in (ROOT/'runs').glob('seed*/*-detector-results.json')];assert len(results)==12;atomic_json({'passed':True,'runs':results,'time':time.time()},ROOT/'all-results.json');status('complete');subprocess.run(['notify-send','Contextual gate study complete','All12 gate detector evaluations finished.'])
 except BaseException as e:status('failed',error=repr(e));subprocess.run(['notify-send','Contextual gate study needs attention',str(e)]);raise
if __name__=='__main__':main()

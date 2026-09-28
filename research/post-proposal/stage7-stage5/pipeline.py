"""Three matched residual refinements, with fresh Stage5 baseline detector evaluation."""
from pathlib import Path
import concurrent.futures,json,os,subprocess,sys,time,threading
from train_cached import atomic_json
ROOT=Path('/data/repos/wiki/artifacts/stage7-stage5-20260918');CODE=Path(__file__).parent;STOP=threading.Event()
def status(phase,**kw):atomic_json({'phase':phase,'time':time.time(),'pid':os.getpid(),**kw},ROOT/'pipeline-progress.json')
def command(args,name,gpu):
 if STOP.is_set():raise RuntimeError('Sibling failed; checkpoints preserved')
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1');log=ROOT/'logs'/f'{name}.log'
 with log.open('a') as f:
  p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,env=env)
  with (ROOT/'processes.jsonl').open('a') as out:out.write(json.dumps({'pid':p.pid,'command':args,'gpu':gpu,'log':str(log),'time':time.time()})+'\n')
  while p.poll() is None:
   if STOP.is_set():p.terminate();p.wait(timeout=1800);raise RuntimeError('Sibling failure; saved progress')
   (ROOT/'supervisor-heartbeat').touch();time.sleep(10)
  if p.returncode:STOP.set();raise RuntimeError(f'{name} exited{p.returncode}; see {log}')
def main():
 (ROOT/'logs').mkdir(exist_ok=True)
 try:
  assert json.loads((ROOT/'preflight.json').read_text())['passed'];status('train-and-evaluate')
  def worker(gpu,seeds):
   for seed in seeds:
    command([sys.executable,str(CODE/'train_cached.py'),'--root',str(ROOT),'--seed',str(seed)],f'train-stage7-seed{seed}',gpu)
    for name in [f'stage7-seed{seed}',f'stage5-baseline-seed{seed}']:
     command([sys.executable,str(CODE/'evaluate_cached.py'),'--root',str(ROOT),'--run',name],'eval-'+name,gpu)
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   fs=[pool.submit(worker,0,[0,2]),pool.submit(worker,1,[1])]
   for f in concurrent.futures.as_completed(fs):f.result()
  results=[json.loads(p.read_text()) for p in sorted((ROOT/'runs').glob('*/detector-results.json'))];assert len(results)==6
  atomic_json({'passed':True,'runs':results,'time':time.time()},ROOT/'all-results.json');status('complete');subprocess.run(['notify-send','Stage7 integration complete','All3 seeds and3 baseline detector evaluations finished.'])
 except Exception as exc:
  STOP.set();status('failed',error=repr(exc));subprocess.run(['notify-send','Stage7 needs attention',str(exc)]);raise
if __name__=='__main__':main()

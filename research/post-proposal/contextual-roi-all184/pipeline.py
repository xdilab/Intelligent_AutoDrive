"""Six all184 runs; wait for original12-run study, reuse immutable frozen caches."""
from pathlib import Path
import concurrent.futures,json,os,subprocess,sys,time,threading
from train_cached import atomic_json
ROOT=Path('/data/repos/wiki/artifacts/contextual-roi-all184-20260918')
BASE=Path('/data/repos/wiki/artifacts/contextual-roi-20260917')
CODE=Path(__file__).parent
STOP=threading.Event()

def status(phase,**kw):atomic_json({'phase':phase,'time':time.time(),'pid':os.getpid(),**kw},ROOT/'pipeline-progress.json')
def command(args,name,gpu):
    if STOP.is_set():raise RuntimeError('Sibling failed; checkpoints preserved')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1')
    log=ROOT/'logs'/f'{name}.log'
    with log.open('a') as f:
        p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,env=env)
        with (ROOT/'processes.jsonl').open('a') as fp:fp.write(json.dumps({'pid':p.pid,'command':args,'gpu':gpu,'log':str(log),'time':time.time()})+'\n')
        while p.poll() is None:
            if STOP.is_set():p.terminate();p.wait(timeout=1800);raise RuntimeError('Sibling failed; stopped safely')
            (ROOT/'supervisor-heartbeat').touch();time.sleep(10)
        if p.returncode:STOP.set();raise RuntimeError(f'{name} exited{p.returncode}; inspect {log}')

def main():
    (ROOT/'logs').mkdir(exist_ok=True,parents=True)
    try:
        while not (BASE/'all-results.json').exists():
            old=json.loads((BASE/'pipeline-progress.json').read_text())
            if old.get('phase')=='failed':raise RuntimeError('Original study failed; all184 launch held for recovery')
            status('waiting-for-original-study',dependency=str(BASE/'all-results.json'));(ROOT/'supervisor-heartbeat').touch();time.sleep(30)
        old=json.loads((BASE/'all-results.json').read_text());assert old['passed'] and len(old['runs'])==12
        marker=json.loads((ROOT/'cache-ready.json').read_text());assert marker['passed']
        cfg=json.loads((ROOT/'protocol.json').read_text());status('train-and-evaluate')
        def worker(gpu,fusion):
            for seed in cfg['seeds']:
                name=f'{fusion}-contrastive-all184-seed{seed}'
                command([sys.executable,str(CODE/'train_cached.py'),'--root',str(ROOT),'--fusion',fusion,'--contrastive',str(cfg['contrastive_weight']),'--seed',str(seed)],'train-'+name,gpu)
                command([sys.executable,str(CODE/'evaluate_cached.py'),'--root',str(ROOT),'--run',name],'eval-'+name,gpu)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(worker,0,'mlp'),pool.submit(worker,1,'attention')]
            for f in concurrent.futures.as_completed(futures):f.result()
        results=[json.loads(p.read_text()) for p in sorted((ROOT/'runs').glob('*/detector-results.json'))];assert len(results)==6
        atomic_json({'passed':True,'contrastive_scope':'all184','runs':results,'time':time.time()},ROOT/'all-results.json');status('complete')
        subprocess.run(['notify-send','All184 contextual study complete','All6 detector evaluations finished.'])
    except Exception as exc:
        status('failed',error=repr(exc));subprocess.run(['notify-send','All184 contextual study needs attention',str(exc)]);raise
if __name__=='__main__':main()

"""Six DCB runs on immutable frozen caches."""
from pathlib import Path
import concurrent.futures,json,os,subprocess,sys,time,threading,shutil
from train_cached import atomic_json
ROOT=Path('/data/repos/wiki/artifacts/contextual-roi-dcb-20260921')
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
        assert shutil.disk_usage(ROOT/'runs').free>5*1024**3
        assert json.loads((ROOT/'preflight.json').read_text())['passed']
        clearance=json.loads((ROOT/'review-clearance.json').read_text());assert clearance['passed']
        from train_cached import file_sha
        for name,digest in clearance['code_sha256'].items():assert file_sha(CODE/name)==digest
        for parent in [BASE]:
            assert json.loads((parent/'all-results.json').read_text())['passed']
        marker=json.loads((ROOT/'cache-ready.json').read_text());assert marker['passed']
        cfg=json.loads((ROOT/'protocol.json').read_text());status('train-and-evaluate')
        def worker(gpu,weight):
            fusion="attention"
            for seed in cfg['seeds']:
                name=f'attention-'+('contrastive-all184' if weight else 'classification')+f'-dcb-seed{seed}'
                command([sys.executable,str(CODE/'train_cached.py'),'--root',str(ROOT),'--fusion',fusion,'--loss','dcb','--contrastive',str(weight),'--seed',str(seed)],'train-'+name,gpu)
                command([sys.executable,str(CODE/'evaluate_cached.py'),'--root',str(ROOT),'--run',name],'eval-'+name,gpu)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(worker,0,0.0),pool.submit(worker,1,0.001)]
            for f in concurrent.futures.as_completed(futures):
                try:f.result()
                except BaseException:STOP.set();raise
        status('blend-selection-and-evaluation')
        for seed in cfg['seeds']:
            name=f'attention-global-dcb-seed{seed}'
            command([sys.executable,str(CODE/'prepare_blend.py'),'--root',str(ROOT),'--seed',str(seed)],'select-'+name,0)
            command([sys.executable,str(CODE/'evaluate_cached.py'),'--root',str(ROOT),'--run',name],'eval-'+name,0)
        results=[json.loads(p.read_text()) for p in sorted((ROOT/'runs').glob('*/detector-results.json'))];assert len(results)==9
        subprocess.run([sys.executable,str(CODE/'compare.py')],check=True)
        atomic_json({'passed':True,'contrastive_scope':'all184','runs':results,'time':time.time()},ROOT/'all-results.json');status('complete')
        subprocess.run(['notify-send','Contextual DCB study complete','All9 detector evaluations finished.'])
    except Exception as exc:
        status('failed',error=repr(exc));subprocess.run(['notify-send','Contextual DCB study needs attention',str(exc)]);raise
if __name__=='__main__':main()

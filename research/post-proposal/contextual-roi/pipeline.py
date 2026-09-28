"""Local persistent pipeline; no cluster GPU allocation or changes to other studies."""
import argparse,concurrent.futures,hashlib,json,os,subprocess,sys,time,threading
from pathlib import Path
from train_cached import atomic_json,file_sha

ABORT=threading.Event()
CODE=Path(__file__).parent
REMOTE='/work/bbyrd1/contextual-roi-20260917'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root;root.mkdir(exist_ok=True,parents=True);logs=root/'logs';logs.mkdir(exist_ok=True)
    def status(phase,**kw):atomic_json({'phase':phase,'time':time.time(),'pid':os.getpid(),**kw},root/'pipeline-progress.json')
    def command(cmd,name,gpu=None):
        if ABORT.is_set():raise RuntimeError('Sibling task failed; preserving checkpoints')
        env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4')
        if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
        with (logs/f'{name}.log').open('a') as f:
            proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
            with (root/'processes.jsonl').open('a') as fp:fp.write(json.dumps({'pid':proc.pid,'command':cmd,'gpu':gpu,'log':str(logs/f'{name}.log'),'time':time.time()})+'\n')
            while proc.poll() is None:
                if ABORT.is_set():
                    proc.terminate()
                    try:proc.wait(timeout=1800)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                    raise RuntimeError('Stopped sibling after pipeline failure; resume checkpoints preserved')
                # Child-specific logs/progress are watched separately for actual stalls.
                (root/'supervisor-heartbeat').touch();time.sleep(10)
            if proc.returncode:ABORT.set();raise RuntimeError(f'{name} exited{proc.returncode}; see {logs/name}.log')
    try:
        if not (root/'download-verified.json').exists():
            status('waiting-for-verified-preparation',remote=REMOTE,job='733853')
            while True:
                probe=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','ncshare',f'test -f {REMOTE}/preparation.json'],capture_output=True)
                if probe.returncode==0:break
                state=subprocess.check_output(['ssh','-o','BatchMode=yes','ncshare','sacct -X -n -P -j 733853 --format=State'],text=True).strip()
                if any(x in state for x in ['FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED']):raise RuntimeError('CPU preparation failed: '+state)
                status('waiting-for-verified-preparation',remote=REMOTE,job='733853');time.sleep(60)
            status('download')
            command(['rsync','-rt','--partial','--timeout=180','-e','ssh -o BatchMode=yes -o ConnectTimeout=15',f'ncshare:{REMOTE}/data',f'ncshare:{REMOTE}/preparation.json',str(root)+'/'],'download')
            status('verify-download');prep=json.loads((root/'preparation.json').read_text());assert prep['passed']
            for name,digest in prep['data_sha256'].items():
                if file_sha(root/'data'/name)!=digest:raise RuntimeError('Transferred data hash mismatch: '+name)
            atomic_json({'passed':True,'preparation_sha256':file_sha(root/'preparation.json'),'time':time.time()},root/'download-verified.json')
        if not (root/'cache-ready.json').exists():
            for split in ['train','dev','val']:
                status('context-extraction',split=split)
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    futures=[pool.submit(command,[sys.executable,str(CODE/'cache_context.py'),'--manifest',str(root/f'data/{split}.jsonl'),'--frames','/data/datasets/ROAD_plusplus/rgb-images','--output',str(root/'context'),'--shards','2','--shard',str(gpu)],f'cache-{split}-gpu{gpu}',gpu) for gpu in [0,1]]
                    for f in concurrent.futures.as_completed(futures):f.result()
            status('compact-context');command([sys.executable,str(CODE/'compact_context.py'),'--root',str(root)],'compact-context')
        status('train-and-evaluate')
        cfg=json.loads((root/'protocol.json').read_text())
        def worker(gpu,fusion):
            for seed in cfg['seeds']:
                for weight in [0.,cfg['contrastive_weight']]:
                    name=f'{fusion}-'+('contrastive' if weight else 'classification')+f'-seed{seed}'
                    command([sys.executable,str(CODE/'train_cached.py'),'--root',str(root),'--fusion',fusion,'--contrastive',str(weight),'--seed',str(seed)],'train-'+name,gpu)
                    command([sys.executable,str(CODE/'evaluate_cached.py'),'--root',str(root),'--run',name],'eval-'+name,gpu)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(worker,0,'mlp'),pool.submit(worker,1,'attention')]
            for f in concurrent.futures.as_completed(futures):f.result()
        results=[json.loads(p.read_text()) for p in sorted((root/'runs').glob('*/detector-results.json'))];assert len(results)==12
        atomic_json({'passed':True,'runs':results,'time':time.time()},root/'all-results.json');status('complete')
        subprocess.run(['notify-send','Contextual RoI study complete','All12 head runs and detector evaluations finished.'])
    except Exception as exc:
        status('failed',error=repr(exc));subprocess.run(['notify-send','Contextual RoI pipeline needs attention',str(exc)[:300]]);raise
if __name__=='__main__':main()

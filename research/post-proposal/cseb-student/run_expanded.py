"""Bounded local expansion pipeline; separate candidate, no default promotion."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--base',type=Path,required=True);a=ap.parse_args()
    a.root.mkdir(parents=True,exist_ok=True);(a.root/'logs').mkdir(exist_ok=True)
    job={'root':str(a.root),'base':str(a.base),'pid':os.getpid(),'started':time.time(),'status':'running','phase':'initializing'}
    save(a.root/'job.json',job)
    try:
        for phase,limit in [('prepare',5400),('train',900),('benchmark',600),('verify',120)]:
            cmd=[sys.executable,str(Path(__file__).with_name('expanded.py')),phase,'--root',str(a.root),'--base',str(a.base)] if phase!='verify' else [sys.executable,str(Path(__file__).with_name('verify_runtime.py')),'--root',str(a.root)]
            job.update(phase=phase,time=time.time());save(a.root/'job.json',job)
            with (a.root/'logs'/f'{phase}.log').open('a') as log:
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                job['child_pid']=child.pid;save(a.root/'job.json',job)
                try:code=child.wait(timeout=limit)
                except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=30);raise RuntimeError(phase+' exceeded time limit')
                if code:raise RuntimeError(f'{phase} exited {code}; inspect logs')
        job.update(status='complete',time=time.time());save(a.root/'job.json',job)
    except BaseException as exc:
        job.update(status='failed',time=time.time(),error=str(exc));save(a.root/'job.json',job);raise
if __name__=='__main__':main()

"""Fail-closed sequential launcher; safe rerun resumes validated progress."""
import argparse, subprocess, sys, time
from common import *

def main(root):
    verify_freeze(root);assert read(root/'validation.json')['passed'];assert (root/'benchmark.json').exists()
    started=time.time();job=dict(status='running',started=started,time=started)
    try:
        for phase,script in [('training','train.py'),('evaluation','evaluate.py'),('analysis','analyze.py')]:
            job.update(phase=phase,time=time.time());save(root/'job.json',job);progress(root,phase)
            with (root/'logs'/f'{phase}.log').open('a') as log:
                subprocess.run([sys.executable,'-u',str(HERE/script),'--root',str(root)],stdout=log,stderr=subprocess.STDOUT,check=True)
        job.update(status='complete',phase='complete',time=time.time(),elapsed_seconds=time.time()-started);save(root/'job.json',job);progress(root,'complete')
    except BaseException as e:
        job.update(status='failed',error=repr(e),time=time.time());save(root/'job.json',job);raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=DEFAULT_ROOT);a=ap.parse_args();main(a.root)

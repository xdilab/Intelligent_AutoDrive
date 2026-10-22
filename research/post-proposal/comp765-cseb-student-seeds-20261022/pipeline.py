"""Bounded sequential phases with visible failures and resumable artifacts."""
from common import *
import subprocess,traceback
def main():
 phases=sys.argv[1:] or ['prepare','train','evaluate','runtime_audit','analyze'];verify();job=dict(root=str(ROOT),pid=os.getpid(),started=time.time(),time=time.time(),status='running',phases=phases);save(ROOT/'job.json',job)
 try:
  for phase in phases:
   job.update(phase=phase,time=time.time());save(ROOT/'job.json',job)
   with (ROOT/'logs'/f'{phase}.log').open('a') as log:
    child=subprocess.Popen([sys.executable,str(HERE/f'{phase}.py')],stdout=log,stderr=subprocess.STDOUT);job['child_pid']=child.pid;save(ROOT/'job.json',job)
    code=child.wait()
   if code:raise RuntimeError(f'{phase} exited{code}; inspect logs')
  job.update(status='complete',time=time.time());save(ROOT/'job.json',job)
 except BaseException as e:
  job.update(status='failed',error=repr(e),traceback=traceback.format_exc(),time=time.time());save(ROOT/'job.json',job);raise
if __name__=='__main__':main()

from assoc_common import *
import subprocess,traceback

def main():
 save(ROOT/'job.json',dict(status='running',time=time.time(),pid=os.getpid(),phases=sys.argv[1:]))
 try:
  for phase in sys.argv[1:]:
   command=[sys.executable,'-u',str(HERE/(phase+'.py'))]
   if phase=='official-benchmark':command=[sys.executable,'-u',str(HERE/'official.py'),'--benchmark-videos','2']
   progress('phase-start',name=phase,command=command)
   with (ROOT/'logs'/f'{phase}.log').open('a') as f:subprocess.run(command,check=True,stdout=f,stderr=subprocess.STDOUT)
   progress('phase-complete',name=phase)
  save(ROOT/'job.json',dict(status='complete',time=time.time(),phases=sys.argv[1:]));progress('pipeline-complete')
 except Exception as e:
  save(ROOT/'job.json',dict(status='failed',time=time.time(),error=repr(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()

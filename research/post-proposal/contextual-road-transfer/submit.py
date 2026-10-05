"""Review/benchmark-gated job DAG; all source progress preserved, no duplicate launches."""
from pathlib import Path
import argparse,json,subprocess,time,fcntl
from prepare_full import atomic,sha
ROOT=Path('/work/bbyrd1/contextual-road-transfer-20261005')
def main():
 lock=(ROOT/'results/submit.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['setup','full'],required=True);a=ap.parse_args();record=ROOT/'results/full-launch.json'
 launch=json.loads(record.read_text()) if record.exists() else {'root':str(ROOT),'jobs':{},'time':time.time()};jobs=launch['jobs'];old=Path('/work/bbyrd1/stage56-full-20260914/results')
 assert (ROOT/'results/code-review.json').exists(),'Prelaunch external review required'
 review=json.loads((ROOT/'results/code-review.json').read_text());assert review['passed']
 for name,digest in review['code_sha256'].items():assert sha(ROOT/'code'/name)==digest,name
 def submit(phase,arm='',gpu=False,dependency=None,array=None,hours=24):
  key=phase+('-'+arm if arm else '');assert key not in jobs
  label='full56-roctx-'+key;cmd=['sbatch','--parsable','--job-name='+label,'--cpus-per-task=8','--mem=96G',f'--time={hours}:00:00','--signal=B:USR1@180','--output='+str(ROOT/'results'/f'{label}-%A-%a.log')]
  # Shell USR1 would kill wrapper; trainer already saves frequently and handles TERM.
  cmd=[x for x in cmd if not x.startswith('--signal=')]
  cmd+=['--partition=gpu-hp','--qos=ncat_h200_hp','--gres=gpu:h200:1'] if gpu else ['--partition=common']
  if dependency:cmd+=['--dependency=afterok:'+':'.join(dependency)]
  if array:cmd+=['--array='+array]
  cmd+=[str(ROOT/'code/run_task.sh'),phase,arm]
  job=subprocess.check_output(cmd,text=True).strip().split(';')[0];assert job.isdigit();jobs[key]=job
  (old/f'roctx-{key}-job.txt').write_text(job+'\n')
  tasks=range(int(array.split('-')[1].split('%')[0])+1) if array else [4294967294]
  for task in tasks:(old/f'{label}-{job}-{task}.log').symlink_to(ROOT/'results'/f'{label}-{job}-{task}.log')
  atomic(launch,record);print('SUBMITTED',key,job,flush=True);return job
 if a.phase=='setup':
  assert not jobs;prepare=submit('prepare',hours=2);submit('benchmark',gpu=True,dependency=[prepare],hours=1)
  launch['eta']='Setup audit and eight-frame-per-encoder benchmark provisionally15–45minutes plus queue. Full-study ETA follows measured extraction throughput.'
 else:
  assert set(jobs)=={'prepare','benchmark'} and json.loads((ROOT/'results/benchmark.json').read_text())['passed']
  assert (ROOT/'results/eta.json').exists()
  cache=submit('cache',gpu=True,dependency=[jobs['benchmark']],array='0-23%8',hours=48)
  for arm in ['frozen','classification','contrastive']:
   compact=submit('compact',arm,dependency=[cache],hours=8)
   preflight=submit('preflight',arm,gpu=True,dependency=[compact],hours=2)
   train=submit('train',arm,gpu=True,dependency=[preflight],array='0-5%6',hours=48)
   blend=submit('blend',arm,gpu=True,dependency=[train],array='0-2%3',hours=8)
   ev=submit('evaluate',arm,gpu=True,dependency=[blend],array='0-8%4',hours=24)
   submit('summarize',arm,dependency=[ev],hours=1)
  launch['complete_submission']=True;launch['eta']=json.loads((ROOT/'results/eta.json').read_text())
 atomic(launch,record)
if __name__=='__main__':main()

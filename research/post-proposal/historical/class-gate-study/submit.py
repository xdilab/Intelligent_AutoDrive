import json,subprocess,time
from pathlib import Path
root=Path('/work/bbyrd1/class-gate-study-20260914');jobs={}
assert not (root/'results/jobs.json').exists(),'Refusing duplicate submission'
for name,dependency in [('prepare',None),('train','prepare'),('evaluate','train'),('summarize','evaluate')]:
 args=['sbatch','--parsable']
 if dependency:args+=['--dependency=afterok:'+jobs[dependency]]
 args+=[str(root/f'code/{name}.sbatch')];job=subprocess.check_output(args,text=True).strip().split(';')[0];assert job.isdigit();jobs[name]=job
 (root/'results/jobs.json').write_text(json.dumps({'jobs':jobs,'submitted_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2)+'\n');print(name,job,flush=True)

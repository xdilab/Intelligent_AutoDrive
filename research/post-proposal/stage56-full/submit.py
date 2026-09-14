from pathlib import Path
import subprocess,json,datetime,hashlib
root=Path('/work/bbyrd1/stage56-full-20260914');out=root/'results/launch.json';assert not out.exists(),'Already submitted'
subprocess.run(['python3',str(root/'code/guard.py')],check=True)
def submit(name,dependency=None):
 saved=root/f'results/{name}-job.txt'
 if saved.exists():return saved.read_text().strip()
 cmd=['sbatch','--parsable']
 if dependency:cmd+=['--dependency='+dependency]
 cmd+=[str(root/f'code/{name}.sbatch')];job=subprocess.check_output(cmd,text=True).strip().split(';')[0];(root/f'results/{name}-job.txt').write_text(job+'\n');return job
prep=submit('prepare');train=submit('train','afterok:'+prep);ev=submit('evaluate','aftercorr:'+train)
d={'prepared_job':prep,'training_array':train,'evaluation_array':ev,'submitted_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'code').iterdir() if p.is_file()},'stages':['stage5','stage6'],'conditions':['frozen','classification','contrastive']};out.write_text(json.dumps(d,indent=2));print(json.dumps(d))

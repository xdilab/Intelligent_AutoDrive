from pathlib import Path
import json,subprocess,datetime,hashlib
root=Path('/work/bbyrd1/stage56-full-20260914');out=root/'results/recovery-launch.json';assert not out.exists(),'Recovery already submitted'
def submit(name,dep=None):
 saved=root/f'results/recovery-{name}-job.txt'
 if saved.exists():return saved.read_text().strip()
 cmd=['sbatch','--parsable']
 if dep:cmd+=['--dependency='+dep]
 cmd+=[str(root/f'code/{name}.sbatch')];job=subprocess.check_output(cmd,text=True).strip().split(';')[0];saved.write_text(job+'\n');return job
verify=submit('verify-frames');train=submit('train','afterok:'+verify);ev=submit('evaluate','aftercorr:'+train)
for name,job in [('train',train),('evaluate',ev)]: (root/f'results/{name}-job.txt').write_text(job+'\n')
d={'frame_verification_job':verify,'training_array':train,'evaluation_array':ev,'submitted_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'supersedes_training':'731843','supersedes_evaluation':'731844','code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'code').iterdir() if p.is_file()}};out.write_text(json.dumps(d,indent=2));print(json.dumps({k:v for k,v in d.items() if k!='code_sha256'}))

from pathlib import Path
import subprocess,json,datetime
root=Path('/work/bbyrd1/stage56-full-20260914')
def submit(name,script,extra):
 p=root/f'results/epoch1-{name}-job.txt'
 if p.exists():return p.read_text().strip()
 job=subprocess.check_output(['sbatch','--parsable',*extra,str(root/'code'/script)],text=True).strip().split(';')[0]
 p.write_text(job+'\n');return job
first=submit('stage5-adapt','evaluate-epoch1.sbatch',['--array=1,2%2'])
controls=submit('controls','evaluate-epoch1.sbatch',['--array=0,3%2','--dependency=afterany:'+first])
ready=submit('ready','wait-epoch1.sbatch',[])
last=submit('stage6-adapt','evaluate-epoch1.sbatch',['--array=4,5%2','--dependency=afterany:'+controls+',afterok:'+ready])
d={'submitted_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage5_adaptation':first,'controls':controls,'stage6_checkpoint_gate':ready,'stage6_adaptation':last,'selection':'fixed epoch1 regardless of development improvement','scope':'reporting only; no tuning; final three-epoch evaluation unchanged','max_concurrent_early_evaluation_gpus':2}
(root/'results/epoch1-launch.json').write_text(json.dumps(d,indent=2));print(json.dumps(d,indent=2))

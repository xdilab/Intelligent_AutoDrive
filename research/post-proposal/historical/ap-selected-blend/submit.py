import datetime,json,subprocess
from pathlib import Path
root=Path('/work/bbyrd1/ap-selected-blend-20260914');out=root/'results/jobs.json';assert not out.exists(),'Duplicate submission refused';record={'submitted_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':{}}
for name in ['evaluate','summarize']:
 cmd=['sbatch','--parsable']
 if name=='summarize':cmd+=['--dependency=afterok:'+record['jobs']['evaluate']]
 job=subprocess.check_output(cmd+[str(root/f'code/{name}.sbatch')],text=True).strip().split(';')[0];record['jobs'][name]=job;out.write_text(json.dumps(record,indent=2));print(name,job,flush=True)

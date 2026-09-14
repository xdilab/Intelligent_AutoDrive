from pathlib import Path
import subprocess,json,datetime
wiki=Path('/data/repos/wiki');out=wiki/'artifacts/stage56-full/collected';out.mkdir(parents=True,exist_ok=True);remote='/work/bbyrd1/stage56-full-20260914'
subprocess.run(['rsync','-rt','ncshare:'+remote+'/results/',str(out)+'/'],check=True)
status=subprocess.check_output(['ssh','ncshare','sacct -j 731841,731843,731844 -X --format=JobID,State,Elapsed,ExitCode -n'],text=True);(out/'status.txt').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n'+status);print(status)
for name in ['preparation.json','launch.json']:
 p=out/name
 if p.exists():
  d=json.loads(p.read_text());print(name,d.get('counts',{k:v for k,v in d.items() if k!='code_sha256'}))

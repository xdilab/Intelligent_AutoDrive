"""Bounded local collection and synthesis for the paired encoder comparison."""
import csv,datetime,json,statistics as st,subprocess,time
from pathlib import Path
A=Path(__file__).resolve().parent;W=A.parent.parent;D=A/'collected';D.mkdir(exist_ok=True);P=W/'findings/adapted-road-stage5-stage6.md';manifest=json.loads((A/'shared-frames.json').read_text())
BEGIN='<!-- AUTO RESULTS START -->';END='<!-- AUTO RESULTS END -->'
def cmd(args):return subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
def update():
 results={}
 for condition in ['original','adapted']:
  for p in (D/'metrics'/condition).glob('*.json'):
   m=json.loads(p.read_text());assert m['n_frames']==36717 and m['frame_sha256']==manifest['frame_sha256'] and m['candidate_sha256']==manifest['candidate_sha256'];results[condition,p.stem]=m
 lines=[BEGIN,'',f'Last collected: {datetime.datetime.now(datetime.timezone.utc).isoformat()}. **{len(results)}/12 evaluations complete.**','','```text',(D/'status.txt').read_text().strip(),'```']
 if len(results)==12:
  metrics={'Triplet':lambda m:m['summary']['triplet'],'Tail 47':lambda m:m['tail']['tail47']['mAP'],'Deep 28':lambda m:m['tail']['deep28']['mAP'],'Common 39':lambda m:m['tail']['common39']['mAP']}
  lines+=['','Mean ± sample SD across seeds 0–2; percentages.','','| Encoder / stage | Triplet | Tail 47 | Deep 28 | Common 39 |','|---|---:|---:|---:|---:|']
  for condition in ['original','adapted']:
   for stage in [5,6]:
    vals=[[f(results[condition,f'seed{s}-stage{stage}']) for s in range(3)] for f in metrics.values()]
    lines.append(f'| {condition} / Stage {stage} | '+' | '.join(f'{st.mean(v):.4f} ± {st.stdev(v):.4f}' for v in vals)+' |')
  lines+=['','Interaction = (adapted Stage 6−5) − (original Stage 6−5), paired within seed. Positive means adaptation increases the measured contribution of phrase evidence. These are descriptive results, not significance claims.','','| Metric | Mean interaction (pp) | Seed 0 | Seed 1 | Seed 2 |','|---|---:|---:|---:|---:|']
  for name,f in metrics.items():
   ds=[(f(results['adapted',f'seed{s}-stage6'])-f(results['adapted',f'seed{s}-stage5']))-(f(results['original',f'seed{s}-stage6'])-f(results['original',f'seed{s}-stage5'])) for s in range(3)]
   lines.append(f'| {name} | {st.mean(ds):+.4f} | '+' | '.join(f'{d:+.4f}' for d in ds)+' |')
  with (D/'per-class.csv').open('w') as out:
   w=csv.writer(out,lineterminator='\n');w.writerow(['condition','seed','stage','head','class','ap_percent'])
   for (condition,name),m in sorted(results.items()):
    for head,values in m['ap_values'].items():
     labels=['agentness'] if head=='agentness' else manifest['labels'][head]
     for label,value in zip(labels,values):w.writerow([condition,name.split('-')[0],name.split('-')[1],head,label,value])
  lines+=['','[All per-class AP values](../artifacts/adapted-road-comparison/collected/per-class.csv).']
 lines+=['',END];text=P.read_text();a=text.index(BEGIN);b=text.index(END)+len(END);text=text[:a]+'\n'.join(lines)+text[b:]
 if len(results)==12:text=text.replace('status: draft','status: complete')
 temp=P.with_suffix('.tmp');temp.write_text(text);temp.replace(P)
 return len(results)==12
end=time.monotonic()+96*3600
while time.monotonic()<end:
 try:
  r=cmd(['rsync','-rt','-e','ssh -o BatchMode=yes -o ConnectTimeout=15','ncshare:/work/bbyrd1/adapted-road-20260911/results/',str(D)+'/'])
  if r.returncode:raise RuntimeError(r.stdout)
  r=cmd(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','ncshare','sacct -j 729475,729476,729477,729478,729479 -X --format=JobID,State,Elapsed,ExitCode -n'])
  if r.returncode:raise RuntimeError(r.stdout)
  (D/'status.txt').write_text(r.stdout);done=update();print(datetime.datetime.now().isoformat(),r.stdout,flush=True)
  if done:
   with (W/'log.md').open('a') as f:f.write('\n\n## Completed — Paired adapted ROAD-Waymo comparison\n\nCollected all 12 evaluations and per-class AP locally; updated [[findings/adapted-road-stage5-stage6]] with paired encoder/stage means and interactions. Requires scientific review; no automatic significance claim.\n')
   break
 except Exception as exc:print(type(exc).__name__,str(exc),flush=True)
 time.sleep(300)

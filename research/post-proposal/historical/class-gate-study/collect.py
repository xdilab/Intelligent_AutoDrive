"""One read-only remote collection; updates local wiki status, never submits jobs."""
import csv,datetime,json,re,subprocess
from pathlib import Path
A=Path(__file__).resolve().parent;W=A.parent.parent;D=A/'collected';D.mkdir(exist_ok=True);remote='/work/bbyrd1/class-gate-study-20260914'
def run(args):return subprocess.run(args,check=True,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180).stdout
run(['rsync','-rt','-e','ssh -o BatchMode=yes -o ConnectTimeout=15','ncshare:'+remote+'/results/',str(D)+'/'])
jobs=json.loads((D/'jobs.json').read_text())['jobs'];status=run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','ncshare','sacct -j '+','.join(jobs.values())+' -X --format=JobID,State,Elapsed,ExitCode -n']);now=datetime.datetime.now(datetime.timezone.utc);(D/'status.txt').write_text(now.isoformat()+'\n'+status);(D/('status-'+now.strftime('%Y-%m-%d-%H%M')+'.txt')).write_text(now.isoformat()+'\n'+status)
manifest=json.loads((A.parent/'post-proposal-experiments/shared-frames.json').read_text());metrics=[]
for p in (D/'metrics').glob('*.json'):
 m=json.loads(p.read_text());assert m['n_frames']==36717 and m['frame_sha256']==manifest['frame_sha256'] and m['candidate_sha256']==manifest['candidate_sha256'];metrics.append(m)
lines=['<!-- CLASS GATE STATUS START -->','',f'Last collected: **{now.isoformat()}**. **{len(metrics)}/24 final evaluations available.**','','```text',status.strip(),'```']
if (D/'preparation.json').exists():
 prep=json.loads((D/'preparation.json').read_text());lines+=['',f"Actual video split: {prep['video_counts']}. Crop rows (expert/gate/development): {prep['row_counts']}. Final-video overlap: {prep['final_video_overlap']}. Nonfinite training elements replaced using the original recipe: {prep['nonfinite_elements_zeroed_as_original_recipe']}."]
if len(metrics)==24 and (D/'summary.json').exists():
 s=json.loads((D/'summary.json').read_text());assert s['n_evaluations']==24;lines+=['','Matched 70%-video experts; mean ± sample SD across three head-training seeds. Percentages.','','| Variant | Triplet | Tail 47 | Deep 28 | Common 39 |','|---|---:|---:|---:|---:|']
 for r in s['means']:lines.append('| '+r['variant']+' | '+' | '.join(f"{r[k+'_mean']:.4f} ± {r[k+'_sd']:.4f}" for k in ['triplet','tail47','deep28','common39'])+' |')
 lines+=['','All metrics and paired differences are descriptive; gate settings were chosen on internal crop AP, not these detection results. Official validation previously informed the hypothesis; this is not a fresh blind test.']
weights=[];labels=manifest['labels']['duplex']+manifest['labels']['triplet']
for p in D.glob('train-seed*.json'):
 t=json.loads(p.read_text())
 for expert,g in t['selection'].items():
  for i,label in enumerate(labels):weights.append({'seed':t['seed'],'expert':expert,'class':label,'group':'duplex' if i<49 else 'triplet','global_weight':g['global']['weight'],'class_weight':g['class']['weights'][i],'selected_lambda':g['class']['lambda'],'global_fallback':i in g['no_positive_indices']})
if weights:
 with (D/'gate-weights.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(weights[0]));w.writeheader();w.writerows(weights)
lines+=['','[Collected run evidence](../artifacts/class-gate-study/collected/). This is a one-time collection; no continuous local monitor is implied.','','<!-- CLASS GATE STATUS END -->']
p=W/'directions/class-conditioned-language-fusion.md';s=p.read_text();a=s.index('<!-- CLASS GATE STATUS START -->');b=s.index('<!-- CLASS GATE STATUS END -->',a)+len('<!-- CLASS GATE STATUS END -->');s=s[:a]+'\n'.join(lines)+s[b:];s=re.sub(r'^updated: .*$',f'updated: {now.date().isoformat()}',s,count=1,flags=re.M);p.write_text(s);print('\n'.join(lines))

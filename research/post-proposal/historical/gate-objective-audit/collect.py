"""One-time audit collection, summary and plotting. Never launches jobs."""
import csv,datetime,json,subprocess
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;D=A/'collected';D.mkdir(exist_ok=True)
subprocess.run(['rsync','-rt','ncshare:/work/bbyrd1/gate-objective-audit-20260914/results/',str(D)+'/'],check=True)
status=subprocess.check_output(['ssh','ncshare','sacct -j 731554 -X --format=JobID,State,Elapsed,ExitCode -n'],text=True)
now=datetime.datetime.now(datetime.timezone.utc).isoformat();(D/'status.txt').write_text(now+'\n'+status);print(status)
counts=json.loads((A.parent/'post-proposal-experiments/train-counts.json').read_text());zs={r['label']:r['z'] for r in counts['rows']};manifest=json.loads((A.parent/'post-proposal-experiments/shared-frames.json').read_text());labels=manifest['labels']['duplex']+manifest['labels']['triplet'];rows=[];summaries=[]
for path in sorted(D.glob('audit-seed[0-2].json')):
 d=json.loads(path.read_text())
 for expert,e in d['experts'].items():
  cols=e['columns'];supported=[c for c in cols if c['supported']]
  summaries.append({'seed':d['seed'],'expert':expert,'global_optimum':e['global_optimum'],'trained_global':e['trained_global'],'supported_classes':len(supported),'supported_zero_optima':sum(c['optimum']==0 for c in supported),'max_supported_optimum':max(c['optimum'] for c in supported),'mean_BCE_trained':np.mean([c['trained_loss']['bce'] for c in cols]),'mean_BCE_direct':np.mean([c['optimum_loss']['bce'] for c in cols])})
  for j,g in enumerate(d['grid']):
   row={'seed':d['seed'],'expert':expert,'weight':g}
   for field in ['bce','positive_contribution','negative_contribution']:row[field]=float(np.mean([c['grid'][j][field] for c in cols]))
   for name,test in [('triplet',lambda z:True),('tail47',lambda z:z<0),('deep28',lambda z:z<-.5),('common39',lambda z:z>=0)]:
    ix=[i for i in range(49,135) if test(zs[labels[i]])];row[name+'_AP']=float(np.mean([cols[i]['grid'][j]['dev_AP'] for i in ix]))
   rows.append(row)
if rows:
 with (D/'curves.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (D/'summary.json').write_text(json.dumps({'collected_utc':now,'completed_seeds':len(list(D.glob('audit-seed[0-2].json'))),'results':summaries,'limits':'Internal cached-crop AP only; same fixed video split; endpoint clipping for convex numerical audit; no final detection tuning'},indent=2))
 print(json.dumps(summaries,indent=2))
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(14,4))
 for ax,metric,title in zip(axes,['bce','tail47_AP','common39_AP'],['Gate-fitting BCE','Development tail-47 crop AP','Development common-39 crop AP']):
  for expert in ['phrase','shuffled']:
   weights=sorted(set(r['weight'] for r in rows));values=[np.mean([r[metric] for r in rows if r['expert']==expert and r['weight']==g]) for g in weights];ax.plot(weights,values,marker='o',markersize=3,label=expert)
  ax.set_xscale('symlog',linthresh=.0015);ax.set_xlabel('Language mixture weight');ax.set_title(title);ax.grid(alpha=.2);ax.legend()
 fig.suptitle('Saved-prediction audit — internal partitions, not detection evaluation');fig.tight_layout();fig.savefig(D/'objective-curves.png',dpi=180);fig.savefig(D/'objective-curves.svg');plt.close(fig)

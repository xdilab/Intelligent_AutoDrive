"""All-class context for the selected detection traces; no new experiment results."""
import csv,json,statistics
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
A=Path(__file__).parent;S=A.parent/'post-proposal-experiments'
counts={r['label']:r for r in json.loads((S/'train-counts.json').read_text())['rows']}
records=defaultdict(dict)
for r in csv.DictReader((S/'per-class-all-runs.csv').open()):
 if r['head']=='triplet' and r['run'].startswith('seed'):
  seed,variant=r['run'].split('-',1);records[r['class']][int(seed[4:]),variant]=float(r['ap_percent'])
rows=[]
for c,vals in records.items():
 d={v:[vals[s,v] for s in range(3)] for v in ['head-flat','head-phrase','stage5','stage6']};pf=[d['head-phrase'][s]-d['head-flat'][s] for s in range(3)];ps=[d['head-phrase'][s]-d['stage5'][s] for s in range(3)];fs=[d['stage6'][s]-d['stage5'][s] for s in range(3)]
 rows.append({'class':c,'train_boxes':counts[c]['train_boxes'],'tail47':counts[c]['z']<0,'deep28':counts[c]['z']<-.5,**{v+'_mean':statistics.mean(x) for v,x in d.items()},'phrase_minus_flat':statistics.mean(pf),'phrase_minus_stage5':statistics.mean(ps),'stage6_minus_stage5':statistics.mean(fs),'phrase_over_flat_seeds':sum(x>0 for x in pf),'phrase_over_stage5_seeds':sum(x>0 for x in ps),'stage6_over_stage5_seeds':sum(x>0 for x in fs)})
with (A/'all-triplet-context.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={'classes':len(rows),'phrase_over_flat_mean':sum(r['phrase_minus_flat']>0 for r in rows),'phrase_over_flat_all_seeds':sum(r['phrase_over_flat_seeds']==3 for r in rows),'phrase_over_stage5_mean':sum(r['phrase_minus_stage5']>0 for r in rows),'phrase_over_stage5_all_seeds':sum(r['phrase_over_stage5_seeds']==3 for r in rows),'phrase_over_stage5_but_stage6_below_stage5_mean':[r['class'] for r in rows if r['phrase_minus_stage5']>0 and r['stage6_minus_stage5']<0]}
(A/'all-triplet-context.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
for ax,col,title in zip(axs,['phrase_minus_flat','phrase_minus_stage5'],['Phrase head versus flat head','Phrase head versus Stage 5']):
 for tail,color,label in [(True,'#087f8c','Tail 47'),(False,'#8064a2','Common 39')]:
  rr=[r for r in rows if r['tail47']==tail];ax.scatter([r[col] for r in rr],[r['stage6_minus_stage5'] for r in rr],c=color,label=label,alpha=.75,s=32)
 ax.axhline(0,color='.5',lw=.8);ax.axvline(0,color='.5',lw=.8);ax.set_xlabel('Phrase-head AP difference (percentage points)');ax.set_ylabel('Stage 6 − Stage 5 AP (percentage points)');ax.set_title(title);ax.legend(frameon=False)
 for c in ['Bus-Stop-VehLane','LarVeh-Stop-Jun','Ped-XingFmRht-xing']:
  r=next(r for r in rows if r['class']==c);ax.annotate(c,(r[col],r['stage6_minus_stage5']),xytext=(-5,5) if r[col]>5 else (5,5),ha='right' if r[col]>5 else 'left',textcoords='offset points',fontsize=7)
fig.suptitle('ROAD-Waymo: all 86 triplets, means across three seeds\nClass-level context; does not identify individual rescued detections',fontsize=12)
fig.savefig(A/'all-triplet-context.png',dpi=180);plt.close(fig)
print(json.dumps(summary,indent=2))

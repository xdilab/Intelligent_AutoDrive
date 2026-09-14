"""Summarize full-population traces without treating seeds as independent videos."""
import csv,json,statistics
from pathlib import Path
A=Path(__file__).parent;D=A/'collected';report=json.loads((D/'report.json').read_text());assert report['parity_pass'];rows=list(csv.DictReader((D/'budget-summary.csv').open()));out=[]
for c in report['classes']:
 rr=[r for r in rows if r['class']==c and float(r['budget_factor'])==1]
 row={'class':c,'gt_count':int(rr[0]['gt_count']),'budget':int(rr[0]['budget'])}
 for field in ['head-flat_tp','head-phrase_tp','stage5_tp','stage6_tp','phrase_only_opportunities','preserved_by_stage6','lost_by_stage6','stage5_hits_lost','stage6_rescues']:
  vals=[int(r[field]) for r in rr];row[field+'_mean']=statistics.mean(vals);row[field+'_by_seed']=vals
 out.append(row)
(A/'trace-summary.json').write_text(json.dumps(out,indent=2)+'\n')
lines=['# Prediction-level language/fusion diagnosis','','All six classes use all 36,717 frames and all YOLO candidates. These are selected case studies from the original completed study, not the BDD-X comparison.','','Counts below are means across three seeds at a prediction budget equal to each class’s evaluation GT count. They are diagnostic recall counts; no threshold was trained and no deployment operating point is claimed. A GT can be matched to different candidate boxes in different variants.','','| Class | GT / budget | Phrase TP | Stage 5 TP | Stage 6 TP | Phrase-only opportunities | Preserved | Lost |','|---|---:|---:|---:|---:|---:|---:|---:|']
for r in out:
 lines.append('| '+r['class']+' | '+str(r['gt_count'])+' | '+' | '.join(f'{r[f+"_mean"]:.1f}' for f in ['head-phrase_tp','stage5_tp','stage6_tp','phrase_only_opportunities','preserved_by_stage6','lost_by_stage6'])+' |')
lines+=['','A phrase-only opportunity is a GT instance captured by the phrase head but not Stage 5 at the same class-specific budget. Preserved/lost indicates whether Stage 6 captures that instance. Stage 6 can also recover other instances that the phrase head misses. Count changes at one budget do not equal AP changes. Sensitivity budgets 0.5× and 2× GT count are included in the CSV.','','[Full budget counts by seed](collected/budget-summary.csv) · [AP parity checks](collected/ap-parity.csv) · [Full provenance report](collected/report.json) · [Candidate examples](collected/examples.json).','',f'Maximum regenerated-versus-archived AP difference: {report["max_ap_error_pp"]:.8f} percentage points.']
(A/'results.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

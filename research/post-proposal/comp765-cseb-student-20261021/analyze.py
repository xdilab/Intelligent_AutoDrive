"""All paired results, per-class differences, plots, and reproducibility record."""
from common import *
import csv,math,subprocess,shutil,importlib.metadata
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def stats(x):return dict(mean=float(np.mean(x)),sample_sd=float(np.std(x,ddof=1)))
def main():
 verify();out=ROOT/'results';out.mkdir(exist_ok=True);data={c:[read(ROOT/'runs'/f'{c}-seed{s}'/'detector-results.json') for s in [0,1,2]] for c in ['supervised','distilled']};data_audit=read(ROOT/'data-audit.json');teacher=read(ROOT/'teacher-comparator.json')['result'];runtime=read(ROOT/'runtime-summary.json');labels=read(INPUTS['frames'])['labels'];ix=groups(labels['triplet']);tcrit=.95*math.sqrt(2/(1-.95**2))
 endpoints=HEADS+['tail47','deep28','common39'];summary={};rows=[]
 for key in endpoints:
  get=lambda r:r['summary'][key] if key in HEADS else r['tail'][key]['mAP']
  a=np.array([get(r) for r in data['supervised']]);b=np.array([get(r) for r in data['distilled']]);d=b-a;half=tcrit*np.std(d,ddof=1)/math.sqrt(3)
  summary[key]=dict(supervised=stats(a),distilled=stats(b),paired_delta=stats(d),paired_deltas=d.tolist(),paired_t95=[float(d.mean()-half),float(d.mean()+half)],teacher=get(teacher))
  for seed in [0,1,2]:rows.append(dict(endpoint=key,seed=seed,supervised=float(a[seed]),distilled=float(b[seed]),delta=float(d[seed]),teacher=get(teacher)))
 def csv_write(path,records):
  with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 csv_write(out/'per-seed.csv',rows);perclass=[]
 for group in HEADS:
  names=['agentness'] if group=='agentness' else labels[group]
  for j,label in enumerate(names):
   a=[r['per_class_AP'][group][j] for r in data['supervised']];b=[r['per_class_AP'][group][j] for r in data['distilled']];d=np.array(b)-a
   column={'agentness':0,'agent':1,'action':11,'loc':33,'duplex':49,'triplet':98}[group]+j
   row=dict(group=group,class_id=j,label=label,training_positive_proposals=data_audit['positive_class_counts']['train'][column],selection_positive_proposals=data_audit['positive_class_counts']['selection'][column],gt_count=data['supervised'][0]['gt_counts'][group][j],tail47=group=='triplet' and j in ix['tail47'],deep28=group=='triplet' and j in ix['deep28'],common39=group=='triplet' and j in ix['common39'],supervised_mean=float(np.mean(a)),distilled_mean=float(np.mean(b)),delta_mean=float(np.mean(d)),delta_sample_sd=float(np.std(d,ddof=1)),teacher=teacher['per_class_AP'][group][j])
   for s in [0,1,2]:row.update({f'supervised_seed{s}':a[s],f'distilled_seed{s}':b[s],f'delta_seed{s}':float(d[s])})
   perclass.append(row)
 csv_write(out/'per-class.csv',perclass);trip=[r for r in perclass if r['group']=='triplet'];counts={}
 for g,items in [('all86',trip)]+[(g,[r for r in trip if r[g]]) for g in ix]:
  counts[g]=dict(improve=sum(r['delta_mean']>0 for r in items),decline=sum(r['delta_mean']<0 for r in items),tie=sum(r['delta_mean']==0 for r in items),supported=sum(r['gt_count']>0 for r in items),classes=len(items))
 save(out/'summary.json',dict(endpoints=summary,class_changes=counts,seeds=[0,1,2],metric='Official detectorAP@0.5 (%)',primary='triplet',time=time.time()))
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
 fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True)
 for seed in [0,1,2]:
  for c,color in [('supervised','#414141'),('distilled','#18658b')]:
   hist=read(ROOT/'runs'/f'{c}-seed{seed}'/'history.json');epochs=[r['epoch'] for r in hist]
   axes[0,seed].plot(epochs,[r['loss'] for r in hist],label=c,color=color);axes[1,seed].plot(epochs,[r['selection_bce'] for r in hist],label=c,color=color)
  axes[0,seed].set_title(f'Seed {seed}');axes[1,seed].set_xlabel('Epoch');axes[0,seed].legend(frameon=False)
 axes[0,0].set_ylabel('Training objective (different losses)');axes[1,0].set_ylabel('Selection supervised BCE');fig.suptitle('Matched student training; KD objective adds a second BCE term');fig.tight_layout();fig.savefig(out/'training-curves.png');fig.savefig(out/'training-curves.pdf');plt.close(fig)
 student_ms=runtime['student']['summary']['total_seconds']['median']*1000;teacher_ms=runtime['teacher']['summary']['total_seconds']['median']*1000
 fig,ax=plt.subplots(figsize=(7,4))
 for c,color,marker in [('supervised','#414141','o'),('distilled','#18658b','s')]:
  r=summary['triplet'][c];ax.errorbar(student_ms,r['mean'],yerr=r['sample_sd'],fmt=marker,color=color,capsize=4,label=c+' (mean ± sample SD,3seeds)')
 ax.scatter([teacher_ms],[summary['triplet']['teacher']],marker='^',color='#a85325',label='Fixed CSEB teacher');ax.set_xscale('log');ax.set_xlabel('Median serial raw-video processing time (ms/frame; log scale)');ax.set_ylabel('Official detector triplet mAP (%)');ax.legend(frameon=False);ax.set_title('Accuracy and measured processing cost')
 fig.text(.05,.01,'Student cost measured on prespecified KD seed0; architecture identical for both arms.\nTiming includes YOLO/decode/export at conf0.25; AP uses fixed historical full candidates.\nTeacher additionally requires0.4s future lookahead; error bars do not show dataset uncertainty.',fontsize=7);fig.tight_layout(rect=[0,.14,1,1]);fig.savefig(out/'accuracy-versus-cost.png');fig.savefig(out/'accuracy-versus-cost.pdf');plt.close(fig)
 table=['| Endpoint | Supervised mean ± SD | Distilled mean ± SD | Paired Δ |95% paired-t interval| Fixed teacher |','|---|---:|---:|---:|---:|---:|']
 for key in ['triplet','tail47','deep28','common39','action','loc','duplex']:
  r=summary[key];table.append(f"|{key}|{r['supervised']['mean']:.3f} ± {r['supervised']['sample_sd']:.3f}|{r['distilled']['mean']:.3f} ± {r['distilled']['sample_sd']:.3f}|{r['paired_delta']['mean']:+.3f}|[{r['paired_t95'][0]:+.3f}, {r['paired_t95'][1]:+.3f}]|{r['teacher']:.3f}|")
 r=summary['triplet'];direction='higher' if r['paired_delta']['mean']>0 else 'lower';ci='The interval includes zero; the three-seed estimate is inconclusive about the sign of the effect.' if r['paired_t95'][0]<=0<=r['paired_t95'][1] else 'The interval excludes zero for this fixed data/protocol; only three seeds make uncertainty estimates fragile.'
 text=f'''# Teacher–Student Distillation for Efficient Road-Event Recognition

COMP 765, Brandon L. Byrd, NC A&T, Dr. Jung Hee Kim. Study started October 21, 2026 after the thesis defense. No thesis or defense slides were modified.

Distillation produced **{abs(r['paired_delta']['mean']):.3f} percentage points {direction} mean official detector tripletAP** than the identical label-only student. {ci} The mean distilled student remains {r['teacher']-r['distilled']['mean']:.3f} points below the fixed CSEB teacher. This gap is descriptive: the teacher has much larger training coverage and a future temporal window.

## Controlled comparison

Frozen shared ImageNet ResNet50 through layer 3, 3×3 RoIAlign, a 256-unit scene GRU, geometry and agent embeddings, and 184 outputs. There are 9,651,864 semantic parameters, including 1,108,568 trainable parameters; YOLO is additional and was previously trained on 600 videos. New semantic-head training uses 64 videos and selection uses 12, with 32 contiguous frames per video, 30 epochs, and paired seeds 0/1/2. AdamW uses learning rate 0.001, weight decay 0.01, and gradient clipping at 5. Only the distillation weight changes from 0 to 1. Both losses use independent Bernoulli BCE over 183 semantic outputs; slot 0 is excluded. Teacher targets are detached, confidence removed, and generated only for annotated training inputs. Unannotated frames update state without loss. The complete data audit counts {data_audit['counts']['train']['annotated']} annotated training frames and {data_audit['counts']['selection']['annotated']} annotated selection frames. Ground-truth-positive training proposals cover {sum(x>0 for x in data_audit['positive_class_counts']['train'][98:184])}/86 triplet classes; soft teacher probabilities are not counted as observed class coverage.

The speed reduction comes from the smaller student architecture. Distillation changes its training objective, not its inference cost; the matched comparison tests whether it improves accuracy at that fixed cost.

Final evaluation uses 36,717 scored frames from 198 official validation videos, 6,163,968 fixed detector candidates, and IoU 0.5. The student processes 38,663 current/past frames from video starts and resets only at video boundaries. Selection is minimum supervised development BCE. Official validation is disjoint from teacher/student semantic-head training and selection. However, all 198 videos and all 36,717 scored frames were used in the shared YOLO detector validation/checkpoint-selection pool, and the benchmark had prior thesis research use. The student selection pool belongs to teacher historical development. This is a reused validation benchmark, not a pristine independent test of the complete teacher/student systems. The pre-evaluation provenance addendum corrects the original protocol’s broader wording without changing the experiment.

## Official detector results

All values are AP percentage points; SD is sample SD over three seeds. The interval uses paired t(df = 2), not independent runs.

{chr(10).join(table)}

Individual seeds and every class are in [per-seed.csv](per-seed.csv) and [per-class.csv](per-class.csv). Tail47: {counts['tail47']['improve']} improve, {counts['tail47']['decline']} decline, {counts['tail47']['tie']} tie on mean AP across seeds. All 86: {counts['all86']['improve']} improve, {counts['all86']['decline']} decline, {counts['all86']['tie']} tie; {counts['all86']['supported']}/86 classes have final GT support. Fixed frequency membership comes from 600 training videos, never validation counts.

## Runtime and temporal behavior

Same RTX A6000, raw video, YOLO image size 1280, confidence threshold 0.25, NMS IoU 0.7, and maximum 300 detections; 4 warmup and 32 measured frames. Median serial processing: student {student_ms:.2f} ms, teacher {teacher_ms:.2f} ms. Student p95: {runtime['student']['summary']['total_seconds']['p95']*1000:.2f}ms; teacherp95{runtime['teacher']['summary']['total_seconds']['p95']*1000:.2f}ms. Processing throughput: student{runtime['student']['frames_per_processing_second']:.2f}frames/s, teacher{runtime['teacher']['frames_per_processing_second']:.2f}frames/s. Median processing ratio{runtime['teacher_to_student_median_processing_ratio']:.2f}×. Teacher also requires 0.4 s of future lookahead at 10 Hz; student requires none.

Stage-level load/decode/preprocess/detector/encoding/head/export times, GPU memory, resolutions and candidates are in `../runtime-summary.json`. The three complete-video isolated runner tests process {sum(x['frames'] for x in runtime['continuous'])} frames. Source-index checks forbid future reads; output scores are finite; checkpoint/evaluator/bundle replay is verified. Full GPU-allocation and CPU-RSS traces are retained, with post-warmup bounds checked. The longest tested source clip spans about {max(x['frames'] for x in runtime['continuous'])/10:.1f} seconds; longer streaming duration remains unmeasured. File-processing throughput is not end-to-end live sensor latency; capture, display, scheduling and deployment behavior are unmeasured. Runtime usesconf0.25 while detectorAP uses the unchanged historical full candidate set, so these are complementary measurements, not the same operating point.

## Historical context and limits

The original 24-frame pilot: 11.59% supervised versus 24.12% distilled proposal AP over 17 supported triplets. The completed 96-frame expansion: 17.67% versus 21.03% proposal AP over 45 classes. Neither is comparable numerically to this official detector endpoint, nor an independent teacher test.

The paired-seed interval describes training-seed variation on this fixed benchmark, not uncertainty over new videos. Only three seeds and 64 training videos; a frozen encoder; training 32-frame segments shorter than full streams; teacher provides richer future-context supervision. Absence of significance is not equivalence. No tuning on final AP, seed cherry-picking, architecture changes, or deployment-safety claims. No backbone-finetuning or other follow-up was mixed into the main comparison. More training coverage or fine-tuning would be separate future studies.

## Artifacts

- Protocol/code: `/data/repos/ROAD_Reason/research/post-proposal/comp765-cseb-student-20261021/`.
- Run root: `{ROOT}`; immutable split/freeze manifests, cache hashes, optimizer/RNG checkpoints and verified backups.
- Reproducibility: [reproducibility.json](reproducibility.json); training plots [PNG](training-curves.png)/[PDF](training-curves.pdf); accuracy/cost [PNG](accuracy-versus-cost.png)/[PDF](accuracy-versus-cost.pdf).
- Separate prespecified seed 0 bundle: `../bundle-seed0/`; existing realtime-v1 unchanged. Run `python run.py --bundle BUNDLE --video INPUT.mp4 --output NEW_OUTPUT --verify-isolation` with the documented environment.
'''
 (out/'FINDINGS.md').write_text(text)
 provenance=read(ROOT/'provenance.json');provenance.update(final_code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),all_code_sha256={str(x):sha(x) for x in HERE.glob('*.py')},checkpoints={f'{c}-seed{s}':read(ROOT/'runs'/f'{c}-seed{s}'/'training-complete.json') for c in data for s in [0,1,2]},bundle=read(ROOT/'bundle-seed0/bundle.json'),runtime=sha(ROOT/'runtime-summary.json'),results_sha256={x.name:sha(x) for x in out.iterdir() if x.is_file() and x.name!='reproducibility.json'},limitations=read(HERE/'protocol.json')['scope_limitations'])
 save(out/'reproducibility.json',provenance);save(ROOT/'study-complete.json',dict(time=time.time(),passed=True,results=str(out/'FINDINGS.md'),six_detector_results=True,runtime_complete=True));progress('study-complete',primary_delta=r['paired_delta']['mean'],results=str(out))
if __name__=='__main__':main()

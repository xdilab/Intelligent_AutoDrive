"""All seeds/classes, matched ResNet context, runtime and reproducibility."""
from common import *
import csv,math,statistics,subprocess,shutil
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def stats(x):return dict(mean=statistics.mean(x),sample_sd=statistics.stdev(x))
def paired(a,b):
 d=[y-x for x,y in zip(a,b)];half=4.302652729749*statistics.stdev(d)/math.sqrt(3);return dict(supervised=stats(a),distilled=stats(b),paired_delta=stats(d),paired_deltas=d,paired_t95=[statistics.mean(d)-half,statistics.mean(d)+half],df=2)
def csv_write(path,rows):
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def main():
 verify();out=ROOT/'results';out.mkdir(exist_ok=True);labels=read(INPUTS['frames'])['labels'];ix=groups(labels['triplet']);endpoints=HEADS+list(ix);data={};records={};runtime=read(ROOT/'runtime-summary.json');teacher=read(INPUTS['teacher_result']);audit=read(ROOT/'data-audit.json')
 for architecture,base in [('efficientnet',ROOT),('resnet',ORIGINAL_ROOT)]:
  for seed in [0,1,2]:
   pair=[]
   for condition in ['supervised','distilled']:
    folder=base/'runs'/f'{condition}-seed{seed}';d=read(folder/'detector-results.json');t=read(folder/'training-complete.json');assert sha(folder/'best.pt')==t['checkpoint_sha256']==d['signature']['checkpoint_sha256'];assert t['signature']['data_manifest_sha256']==sha(base/'data-manifest.json');assert d['n_frames']==36717 and d['n_candidates']==6163968;pair.append(t);data[architecture,condition,seed]=d;records[f'{architecture}-{condition}-seed{seed}']=dict(result_sha256=sha(folder/'detector-results.json'),training=t,path=str(folder))
   assert pair[0]['initialization_sha256']==pair[1]['initialization_sha256'];assert pair[0]['order_sha256']==pair[1]['order_sha256']==read(ORIGINAL_ROOT/'runs'/f'supervised-seed{seed}'/'training-complete.json')['order_sha256']
 for video in read(ROOT/'split-manifest.json')['groups']['final']:assert read(ROOT/'evaluation-inputs'/f'{video}.json')['images_sha256']==read(ORIGINAL_ROOT/'evaluation-inputs'/f'{video}.json')['images_sha256']
 for d in data.values():
  for h in ['agentness','agent']:assert d['per_class_AP'][h]==data['resnet','supervised',0]['per_class_AP'][h]
 rows=[];summary={};architecture_deltas={}
 for key in endpoints:
  get=lambda r:r['summary'][key] if key in HEADS else r['tail'][key]['mAP'];summary[key]={};architecture_deltas[key]={}
  for architecture in ['efficientnet','resnet']:
   a=[get(data[architecture,'supervised',s]) for s in [0,1,2]];b=[get(data[architecture,'distilled',s]) for s in [0,1,2]];summary[key][architecture]=paired(a,b)
   for seed in [0,1,2]:rows.append(dict(architecture=architecture,endpoint=key,seed=seed,supervised=a[seed],distilled=b[seed],delta=b[seed]-a[seed]))
  for condition in ['supervised','distilled']:
   d=[get(data['efficientnet',condition,s])-get(data['resnet',condition,s]) for s in [0,1,2]];architecture_deltas[key][condition]=dict(**stats(d),deltas=d,interpretation='Exploratory architecture contrast, including stride/channels/pretraining/projection changes; no identical initialization across encoders')
 csv_write(out/'per-seed.csv',rows);perclass=[]
 for group in HEADS:
  for j,label in enumerate(['agentness'] if group=='agentness' else labels[group]):
   row=dict(group=group,class_id=j,label=label,gt_count=data['efficientnet','supervised',0]['gt_counts'][group][j],tail47=group=='triplet' and j in ix['tail47'],deep28=group=='triplet' and j in ix['deep28'],common39=group=='triplet' and j in ix['common39'])
   for architecture in ['efficientnet','resnet']:
    for condition in ['supervised','distilled']:
     values=[data[architecture,condition,s]['per_class_AP'][group][j] for s in [0,1,2]]
     for s,v in enumerate(values):row[f'{architecture}_{condition}_seed{s}']=v
     row[f'{architecture}_{condition}_mean']=statistics.mean(values)
    row[f'{architecture}_kd_delta_mean']=row[f'{architecture}_distilled_mean']-row[f'{architecture}_supervised_mean']
   perclass.append(row)
 csv_write(out/'per-class.csv',perclass);changes={}
 for subset in ['all86']+list(ix):
  items=[r for r in perclass if r['group']=='triplet' and (subset=='all86' or r[subset])];changes[subset]=dict(improve=sum(r['efficientnet_kd_delta_mean']>0 for r in items),decline=sum(r['efficientnet_kd_delta_mean']<0 for r in items),tie=sum(r['efficientnet_kd_delta_mean']==0 for r in items))
 save(out/'summary.json',dict(time=time.time(),metric='Official detector AP@0.5 (%)',primary='triplet',seeds=[0,1,2],endpoints=summary,architecture_deltas=architecture_deltas,class_changes=changes,teacher_triplet=teacher['summary']['triplet'],input_parity_verified=True,all_12_run_records=records));shutil.copy2(ROOT/'runtime-summary.json',out/'runtime-summary.json')
 timing=[dict(architecture=a,stage=k,**v) for a in ['efficientnet','resnet'] for k,v in runtime[a]['summary'].items()];csv_write(out/'runtime-stages-seconds.csv',timing)
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
 fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True)
 for seed in [0,1,2]:
  for condition,color in [('supervised','#555555'),('distilled','#18658b')]:
   hist=read(ROOT/'runs'/f'{condition}-seed{seed}'/'history.json');axes[0,seed].plot([h['epoch'] for h in hist],[h['loss'] for h in hist],color=color,label=condition);axes[1,seed].plot([h['epoch'] for h in hist],[h['selection_bce'] for h in hist],color=color,label=condition)
  axes[0,seed].set_title(f'Seed {seed}');axes[1,seed].set_xlabel('Epoch')
 axes[0,0].legend(frameon=False);axes[0,0].set_ylabel('Training objective (different losses)');axes[1,0].set_ylabel('Development supervised BCE');fig.suptitle('EfficientNet-B0 matched supervised/distilled training');fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(out/'training-curves.png');fig.savefig(out/'training-curves.pdf');plt.close(fig)
 for cost,filename,xlabel in [('total_seconds','accuracy-versus-cost','Median serial raw-video processing (ms/frame)'),('parameters','accuracy-versus-parameters','Semantic encoder + head parameters (millions; YOLO additional)')]:
  fig,ax=plt.subplots(figsize=(8,4.5))
  for architecture,color in [('efficientnet','#18658b'),('resnet','#777777')]:
   x=(runtime[architecture]['parameters']['encoder']+runtime[architecture]['parameters']['heads'])/1e6 if cost=='parameters' else runtime[architecture]['summary'][cost]['median']*1000
   for condition,marker in [('supervised','o'),('distilled','s')]:
    d=summary['triplet'][architecture][condition];ax.errorbar(x,d['mean'],yerr=d['sample_sd'],fmt=marker,color=color,capsize=4,label=f'{architecture} {condition}')
  ax.set_xlabel(xlabel);ax.set_ylabel('Official triplet mAP (%)');ax.set_ylim(bottom=0);ax.legend(frameon=False);ax.set_title('Three seeds per architecture/condition; mean ± sample SD');fig.text(.03,.02,'Timing: prespecified distilled seed0, conf0.25, same GPU/input/candidates, 32 measured frames.\nOfficial AP uses historical full candidates. Error bars describe seeds, not unseen-video uncertainty.',fontsize=8);fig.tight_layout(rect=[0,.13,1,1]);fig.savefig(out/(filename+'.png'));fig.savefig(out/(filename+'.pdf'));plt.close(fig)
 table=['| Endpoint | B0 supervised mean ± SD | B0 distilled mean ± SD | Paired gain | 95% paired-t interval |','|---|---:|---:|---:|---:|']
 for key in ['triplet','tail47','deep28','common39','action','loc','duplex']:
  d=summary[key]['efficientnet'];table.append(f"| {key} | {d['supervised']['mean']:.3f} ± {d['supervised']['sample_sd']:.3f} | {d['distilled']['mean']:.3f} ± {d['distilled']['sample_sd']:.3f} | {d['paired_delta']['mean']:+.3f} | [{d['paired_t95'][0]:+.3f}, {d['paired_t95'][1]:+.3f}] |")
 d=summary['triplet']['efficientnet'];r=summary['triplet']['resnet'];er=runtime['efficientnet'];rr=runtime['resnet'];ci='includes zero; the sign is inconclusive under this interval' if d['paired_t95'][0]<=0<=d['paired_t95'][1] else 'excludes zero on this fixed benchmark, with only three seeds';params=read(ROOT/'preflight.json')['parameters'];tail=changes['tail47']
 report=f'''# EfficientNet-B0 student — COMP 765 architecture follow-up

Completed six runs, three matched seeds0–2, started October23,2026. This is a separate architecture follow-up after the ResNet study, not a replacement of historical results or a thesis experiment. No final-AP tuning, adaptive seed selection, backbone fine-tuning or cross-validation.

## Controlled experiment

Full frozen ImageNet EfficientNet-B0 features at448×672, stride32,1280channels;3×3 RoIAlign mean, scene mean,256-unit GRU, geometry8, agent embedding16, MLP256 to184. Frozen encoder {params['frozen_backbone']:,}, trainable head {params['trainable_head']:,}, semantic total {params['semantic_total']:,}; shared YOLO adds {er['parameters']['detector']:,}. ResNet used stride16/1024channels and9,651,864 semantic parameters. The architecture comparison therefore changes pretrained features, spatial resolution, channels and input projection count together. Only the distillation loss differs within each B0 pair; there is no identical trainable initialization across different encoder dimensions.

The exact64train/12selection videos, all2,432 image steps, boxes/agents/confidence/labels/teacher targets and30epoch budget are preserved. New visual features come only from the same current images. All76 caches passed field-by-field supervision checks. Paired initialization, video order, AdamW0.001/0.01, clipping5, full32frame BPTT and minimum supervised development BCE selection match. Teacher targets use independent Bernoulli probabilities; supervised BCE versus supervised BCE+weight1 teacher BCE, outputs1–183. Slot0 excluded from loss; detector confidence applied exactly once. Unannotated frames update state with no loss. Teacher has current−3..current+4 at10Hz as privileged training supervision; student has no future input.

## Official detector AP@0.5 (%)

{chr(10).join(table)}

The triplet interval {ci}. Intervals use paired t(df2), describe seed variation conditional on the fixed benchmark, and do not establish equivalence when zero is included. Tail47: {tail['improve']} mean-class improvements, {tail['decline']} declines, {tail['tie']} ties. All outcomes and all classes are in [per-seed.csv](per-seed.csv) and [per-class.csv](per-class.csv).

Matched historical ResNet seeds0–2: supervised {r['supervised']['mean']:.3f} ± {r['supervised']['sample_sd']:.3f}, distilled {r['distilled']['mean']:.3f} ± {r['distilled']['sample_sd']:.3f}. B0 minus ResNet distilled mean is {d['distilled']['mean']-r['distilled']['mean']:+.3f} points. These are the same three seeds and verified identical split/order/candidate/evaluator inputs; the completed ten-seed ResNet estimate is separate context, not substituted for this matched three-seed comparison. Architecture contrasts are exploratory. Historical fixed CSEB teacher triplet AP is {teacher['summary']['triplet']:.3f}; B0 distilled remains {teacher['summary']['triplet']-d['distilled']['mean']:.3f} points below it. Teacher has broader training coverage and future context; the original cross-environment teacher numerical-parity caveat remains.

All evaluations use36,717scored frames/198videos/6,163,968fixed candidates,38,663causal state steps from video starts,85/86supported triplets with all86 retained. Tail47/Deep28/Common39 retain fixed training-frequency membership. All current-image hashes match the original official evaluation inputs. This is detector AP, not historical pilot proposal AP.

## Measured efficiency and inference

Fresh processes on RTX A6000 GPU0, same raw MP4, same YOLO policy(imgsz1280/conf0.25/IoU0.7/max300), exact candidate parity;4warmup+32measured frames with CUDA synchronization at stage boundaries. No concurrent experiment on GPU0 during timing. Source resolution/candidate counts, load/decode/preprocessing/detector/encoder/head/CPU+JSONexport times, memory and parameters are in [runtime-summary.json](runtime-summary.json) and [stage CSV](runtime-stages-seconds.csv).

B0 median total {er['summary']['total_seconds']['median']*1000:.2f}ms, p95 {er['summary']['total_seconds']['p95']*1000:.2f}ms, throughput {er['frames_per_processing_second']:.2f}frames/s. ResNet fresh median {rr['summary']['total_seconds']['median']*1000:.2f}ms, p95 {rr['summary']['total_seconds']['p95']*1000:.2f}ms, throughput {rr['frames_per_processing_second']:.2f}frames/s. Encoder+RoI median: B0 {er['summary']['encoding_seconds']['median']*1000:.2f}ms versus ResNet {rr['summary']['encoding_seconds']['median']*1000:.2f}ms. These are small, instrumented serial file-processing samples with uncontrolled filesystem-cache state, not sensor-to-decision deployment latency or a guaranteed real-time rate. Official AP uses lower-threshold historical full candidates, so timing and AP are different operating points.

The prespecified B0 distilled seed0 bundle processes {sum(x['frames'] for x in runtime['continuous'])} frames across3 complete training videos with zero reset at boundaries, finite outputs, stable post-warmup CPU/GPU allocation and forbidden external model/cache reads. Packaged and evaluated checkpoint replay is exact within1e-7; teacher and teacher caches are absent. Raw video is the external input. Existing realtime bundle is unchanged. No model is promoted based on final AP.

## Exposure, limits and records

All198official videos were used in shared YOLO historical validation/checkpoint selection; semantic teacher/student train/selection excludes them, but the benchmark is reused and not an independent full-system test. The selection pool belongs to teacher historical development. Three seeds,64training videos, frozen features and32frame training windows limit conclusions. More seeds do not repair data exposure. No deployment-safety claim.

Protocol/code: `{HERE}`. Full checkpoints/cache/logs: `{ROOT}`. Bundle: `{ROOT}/bundle-seed0`. [Training curves](training-curves.png), [accuracy versus processing cost](accuracy-versus-cost.png), [accuracy versus parameter count](accuracy-versus-parameters.png), [reproducibility](reproducibility.json). Use the frozen protocol and existing environment. No thesis/slides changed or remote push performed.
'''
 (out/'FINDINGS.md').write_text(report)
 save(out/'reproducibility.json',dict(time=time.time(),revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),code_sha256={f.name:sha(f) for f in HERE.glob('*.py')},freeze=read(ROOT/'freeze.json'),environment=read(ROOT/'environment.json'),data_manifest_sha256=sha(ROOT/'data-manifest.json'),checkpoints=records,runtime_sha256=sha(ROOT/'runtime-summary.json'),bundle=read(ROOT/'bundle-seed0/bundle.json'),results_sha256={f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='reproducibility.json'}))
 save(ROOT/'study-complete.json',dict(time=time.time(),passed=True,pairs=3,detector_runs=6,input_parity=True,runtime_complete=True));progress('study-complete',triplet_delta=d['paired_delta']['mean'])
if __name__=='__main__':main()

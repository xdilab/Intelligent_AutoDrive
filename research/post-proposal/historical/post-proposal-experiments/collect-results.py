"""Unattended, bounded read-only cluster polling and local wiki result collection."""
import datetime,json,statistics,subprocess,time
from pathlib import Path
W=Path('/data/repos/wiki');A=W/'artifacts/post-proposal-experiments';D=A/'cluster-results';D.mkdir(exist_ok=True)
PAGE=W/'findings/post-proposal-controlled-study.md'
EXPECTED=['stage'+str(i) for i in range(4)]+[f'seed{s}-{v}' for s in range(3) for v in ['head-flat','head-phrase','head-shuffled','stage5','stage6','fusion-flat-evidence','fusion-shuffled']]
def command(args):
    return subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
def update(state):
    metrics={p.stem:json.loads(p.read_text()) for p in (D/'metrics').glob('*.json')}
    for name,m in metrics.items():
        assert name in EXPECTED and m['n_frames']==36717
        assert m['frame_sha256']==json.loads((A/'shared-frames.json').read_text())['frame_sha256']
    now=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds');complete=len(metrics)==25
    lines=['---','type: finding','title: "Post-proposal controlled study: shared evaluation and repeated seeds"','aliases: []','created: 2026-09-10',f'updated: {now[:10]}','sources: [wiki/artifacts/post-proposal-experiments/cluster-results, ROAD_Reason/experiments/exp12_phrase_head]','tags: [road-waymo, evaluation, repeated-seeds, controls]',f'status: {"complete" if complete else "draft"}','---','','# Post-proposal controlled study','',f'Last collected: {now}. **{len(metrics)}/25 evaluations completed.**','', '## Cluster status','', '```text',state.strip(),'```','','## Protocol and scope','', 'All runs use the same ordered 36,717-frame manifest and unchanged baseline AP implementation at IoU 0.5. Stage 0 retains its own detector boxes; Stages 1–6 use the same YOLO candidates. Source caches are checksum-verified before training.','', 'The omitted frame `train_00653_00042` has zero YOLO boxes and zero ground-truth boxes. The cache writer skipped empty detections. The evaluator now retains this frame explicitly and rejects missing features for any nonempty frame.','', 'Seeds 0, 1, and 2 repeat supervised focal-loss training with fixed features and video-level two-fold out-of-fold composition inputs. Stage 5 uses primitive scores and crop features. Stage 6 adds phrase-head composition scores. Controls supply equally wide flat-head scores or shuffled class-to-phrase assignments. These controls probe attribution; they do not by themselves prove a semantic mechanism.','', 'Shuffled phrases preserve the phrase-vector set but change the class assignment. Because the projection is trainable, this is not a pure removal of semantics. Seed variation measures optimization variability, not uncertainty over new datasets. Batch order uses a dedicated seeded generator, so these are new controlled runs rather than exact reproductions of historical training.','', 'See [[directions/thesis-proposal-fall-2026]] and [[findings/exp12-phrase-head-attribution]] for context. Historical results remain separate.','', '## Collected metrics','', 'Values are percentages; tail = 47 training-frequency triplets, deep tail = 28, common = 39.','', '| Run | Triplet f-mAP | Tail 47 | Deep 28 | Common 39 |','|---|---:|---:|---:|---:|']
    for name in EXPECTED:
        if name in metrics:
            m=metrics[name];t=m['tail'];lines.append(f'| {name} | {m["summary"]["triplet"]:.4f} | {t["tail47"]["mAP"]:.4f} | {t["deep28"]["mAP"]:.4f} | {t["common39"]["mAP"]:.4f} |')
    if complete:
        lines+=['','## Repeated-seed comparison','','Means ± sample standard deviations across three seeds.','','| Variant | Triplet f-mAP | Tail 47 |','|---|---:|---:|']
        for v in ['head-flat','head-phrase','head-shuffled','stage5','stage6','fusion-flat-evidence','fusion-shuffled']:
            ms=[metrics[f'seed{s}-{v}'] for s in range(3)];xs=[m['summary']['triplet'] for m in ms];ys=[m['tail']['tail47']['mAP'] for m in ms]
            lines.append(f'| {v} | {statistics.mean(xs):.4f} ± {statistics.stdev(xs):.4f} | {statistics.mean(ys):.4f} ± {statistics.stdev(ys):.4f} |')
        deltas=[metrics[f'seed{s}-stage6']['tail']['tail47']['mAP']-metrics[f'seed{s}-stage5']['tail']['tail47']['mAP'] for s in range(3)]
        lines+=['',f'Stage 6 minus Stage 5 tail difference: {statistics.mean(deltas):+.4f} percentage points on average; paired seed differences: '+', '.join(f'{d:+.4f}' for d in deltas)+'. No statistical-significance claim is made.','','## Per-class comparison','','Stage 6 minus Stage 5, averaged over seeds; full values are in the collected JSON artifacts.','', '| Triplet | Mean change (percentage points) |','|---|---:|']
        labels=json.loads((A/'shared-frames.json').read_text())['labels']['triplet'];rows=[]
        for i,label in enumerate(labels):
            delta=statistics.mean(metrics[f'seed{s}-stage6']['ap_values']['triplet'][i]-metrics[f'seed{s}-stage5']['ap_values']['triplet'][i] for s in range(3));rows.append((delta,label))
        for delta,label in sorted(rows,reverse=True):lines.append(f'| {label} | {delta:+.4f} |')
    temp=PAGE.with_suffix('.tmp');temp.write_text('\n'.join(lines)+'\n');temp.replace(PAGE)
    return complete
if __name__=='__main__':
    deadline=time.monotonic()+36*3600
    while time.monotonic()<deadline:
        try:
            sync=command(['rsync','-rt','-e','ssh -o BatchMode=yes -o ConnectTimeout=15','ncshare:/work/bbyrd1/proposal-study-20260910/results/',str(D)+'/'])
            status=command(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','ncshare','sacct -j 728738,728739,728740 --format=JobID,JobName,State,Elapsed,ExitCode -n; squeue -j 728738,728739,728740 -o "%.18i %.24j %.10T %.30R"'])
            (D/'status.txt').write_text(status.stdout)
            if sync.returncode:print(sync.stdout,flush=True)
            complete=update(status.stdout)
            print(datetime.datetime.now().isoformat(), 'complete' if complete else 'pending',status.stdout,flush=True)
            if complete:
                with (W/'log.md').open('a') as f:f.write('\n\n## 2026-09-11 — Controlled proposal study results collected\n\nCollected all 25 evaluations for jobs 728738–728740. Updated [[findings/post-proposal-controlled-study]] with three-seed summaries, matched controls, and all 86 triplet differences. Artifacts: `artifacts/post-proposal-experiments/cluster-results/`. Results require scientific review; no significance claim is made.\n')
                check=command(['python3',str(W/'.claude/scripts/wiki.py'),'lint']);(D/'wiki-lint.txt').write_text(check.stdout);break
        except Exception as exc:print(type(exc).__name__,str(exc),flush=True)
        time.sleep(120)

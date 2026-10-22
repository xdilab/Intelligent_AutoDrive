"""All-seed paired detector analysis; never substitutes partial results."""
import argparse, csv
import numpy as np
from scipy.stats import t
from common import *

def summarize(a):return dict(values=a.tolist(),mean=float(a.mean()),sample_sd=float(a.std(ddof=1)))

def main(root):
    runs={c:[read(root/'runs'/f'{c}-seed{s}'/'detector-results.json') for s in [0,1,2]] for c in ['single','multi']}
    manifest=read(INPUTS['frames']);labels=manifest['labels']['triplet'];ix=groups(labels)
    for cc in runs.values():
        for r in cc:
            assert r['frame_sha256']==manifest['frame_sha256'] and r['candidate_sha256']==manifest['candidate_sha256']
            assert r['n_frames']==36717 and r['selected_epoch']==10
            assert r['signature']['freeze']==sha(root/'freeze.json')
    report={'study':read(HERE/'protocol.json')['study'],'metric':'detector AP@0.5, percentage points','seeds':[0,1,2],'n':3,'metrics':{},'limitations':read(HERE/'protocol.json')['limitations']}
    table=[]
    for name in ['tail47','triplet','deep28','common39','agentness','agent','action','loc','duplex']:
        arm={c:np.array([r['tail'][name]['mAP'] if name in ix else r['summary'][name] for r in rr]) for c,rr in runs.items()}
        delta=arm['multi']-arm['single'];half=float(t.ppf(.975,2)*delta.std(ddof=1)/np.sqrt(3))
        report['metrics'][name]=dict(single=summarize(arm['single']),multi=summarize(arm['multi']),paired_delta=dict(**summarize(delta),ci95=[float(delta.mean()-half),float(delta.mean()+half)],df=2))
        for seed in [0,1,2]:table.append(dict(metric=name,seed=seed,single=arm['single'][seed],multi=arm['multi'][seed],paired_delta=delta[seed]))
    perclass=[]
    for i,label in enumerate(labels):
        a=np.array([r['per_class_AP']['triplet'][i] for r in runs['single']]);b=np.array([r['per_class_AP']['triplet'][i] for r in runs['multi']]);d=b-a
        row=dict(class_id=i,label=label,tail47=i in ix['tail47'],deep28=i in ix['deep28'],common39=i in ix['common39'],single_mean=a.mean(),single_sample_sd=a.std(ddof=1),multi_mean=b.mean(),multi_sample_sd=b.std(ddof=1),paired_delta_mean=d.mean(),paired_delta_sample_sd=d.std(ddof=1))
        for s in [0,1,2]:row.update({f'single_seed{s}':a[s],f'multi_seed{s}':b[s],f'delta_seed{s}':d[s]})
        perclass.append(row)
    report['class_counts']={group:dict(improved=int(sum(perclass[i]['paired_delta_mean']>0 for i in ids)),declined=int(sum(perclass[i]['paired_delta_mean']<0 for i in ids)),tied=int(sum(perclass[i]['paired_delta_mean']==0 for i in ids))) for group,ids in dict(all86=list(range(86)),**ix).items()}
    out=root/'analysis';out.mkdir(exist_ok=True);save(out/'comparison.json',report)
    for name,rows in [('per-seed.csv',table),('per-class-triplet.csv',perclass)]:
        with (out/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# COMP 765 detector results','',report['study'],'','All values are detector frame AP@0.5 percentages; deltas are percentage points. Fresh paired seeds0,1,2.','', '| Endpoint | Single mean ± SD | Multi mean ± SD | Paired Δ | 95% paired-t CI |','|---|---:|---:|---:|---:|']
    for name in ['tail47','triplet','deep28','common39']:
        v=report['metrics'][name];a=v['single'];b=v['multi'];d=v['paired_delta'];lo,hi=d['ci95']
        lines.append(f'| {name} | {a["mean"]:.4f} ± {a["sample_sd"]:.4f} | {b["mean"]:.4f} ± {b["sample_sd"]:.4f} | {d["mean"]:+.4f} | [{lo:+.4f}, {hi:+.4f}] |')
    lines += ['', '| Seed | Single Tail47 | Multi Tail47 | Δ Tail47 |','|---|---:|---:|---:|']
    v=report['metrics']['tail47']
    for s in [0,1,2]:lines.append(f'| {s} | {v["single"]["values"][s]:.4f} | {v["multi"]["values"][s]:.4f} | {v["paired_delta"]["values"][s]:+.4f} |')
    d=v['paired_delta'];lo,hi=d['ci95'];counts=report['class_counts']['tail47']
    if lo>0:interpretation='The primary paired interval favors multi-description prototypes under this fixed protocol.'
    elif hi<0:interpretation='The primary paired interval favors the single-description baseline under this fixed protocol.'
    else:interpretation='The primary paired interval crosses zero: this three-seed study is inconclusive about a reliable Tail47 improvement. This is not evidence of equivalence.'
    lines += ['',interpretation,'',f'Tail classes by mean paired AP change: {counts["improved"]} improve, {counts["declined"]} decline, {counts["tied"]} tie. All86 classes, all per-seed differences and all standard group metrics are exported; no success-only selection.','', 'With only three training seeds the paired-t interval is fragile and does not quantify uncertainty over new datasets. Secondary endpoints and per-class differences are descriptive, not multiplicity-corrected confirmatory tests.','', 'Averaging changes prototype geometry as well as the wording representation. This practical comparison does not isolate linguistic semantics or distinguish averaging from the quality of added descriptions. Canonical wording imprecisions are retained and disclosed in the description manifest.','', 'No thesis records were modified. See protocol, freeze, input hashes, dependency versions, exact source commit and checkpoint-selection records for reproducibility.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    save(out/'artifact-hashes.json',{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.suffix in ['.pt','.json','.csv','.md','.npy'] and 'backup' not in str(p) and p.name!='artifact-hashes.json'})
    print(json.dumps(report['metrics']['tail47']),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=DEFAULT_ROOT);a=ap.parse_args();main(a.root)

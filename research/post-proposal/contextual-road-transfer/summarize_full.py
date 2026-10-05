"""Aggregate only complete matched detector results; preserve per-seed evidence."""
from pathlib import Path
import argparse,json,statistics,time
from prepare_full import atomic,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);r=ap.parse_args().root
    groups={};evidence={}
    for arm in ['classification','contrastive-all184','global']:
        rows=[]
        for seed in range(3):
            p=r/'runs'/f'attention-{arm}-dcb-seed{seed}'/'detector-results.json';d=json.loads(p.read_text())
            assert d['n_frames']==36717
            rows.append({**d['summary'],**{k:v['mAP'] for k,v in d['tail'].items()}})
            evidence[str(p.relative_to(r))]=sha(p)
        groups[arm]={k:{'mean':statistics.mean(row[k] for row in rows),'sample_sd':statistics.stdev(row[k] for row in rows),'seeds':[row[k] for row in rows]} for k in rows[0]}
    atomic({'passed':True,'time':time.time(),'metric':'official validation detector AP@0.5 (%)','seeds':[0,1,2],'groups':groups,'sources':evidence,'protocol_sha256':sha(r/'protocol.json')},r/'results/full-summary.json')
    print('FULL_STUDY_COMPLETE',json.dumps(groups),flush=True)
if __name__=='__main__':main()

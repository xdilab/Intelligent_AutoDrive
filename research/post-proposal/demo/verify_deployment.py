"""Verify completed deployment output against the recorded research pipeline."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from export_sample import sha
from infer_video import write_json


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--sample',type=Path,required=True)
    ap.add_argument('--reference-predictions',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    report=json.loads((a.run/'report.json').read_text())
    progress=json.loads((a.run/'progress.json').read_text())
    manifest=json.loads((a.sample/'manifest.json').read_text())
    reference=json.loads((a.reference_predictions/'predictions.json').read_text())
    if progress['phase']!='complete':raise ValueError('Deployment run is incomplete')
    if report['inputs_sha256']['video']!=manifest['video_sha256']:
        raise ValueError('Source video differs')
    if report['inputs_sha256']['checkpoint']!=reference['checkpoint_sha256']:
        raise ValueError('Checkpoint differs')
    if report['inputs_sha256']['yolo']!=manifest['yolo_sha256'] or report['detector']!=manifest['detector']:
        raise ValueError('Detector configuration differs')
    if report['labels']!=manifest['labels']:
        raise ValueError('Label ordering differs')
    if sha(a.sample/'manifest.json')!=reference['manifest_sha256']:
        raise ValueError('Reference manifest hash differs')
    for key in ['q','agent']:
        if sha(a.sample/f'{key}.npy')!=manifest[f'{key}_sha256']:
            raise ValueError(f'Reference {key} checksum differs')
    q=np.load(a.sample/'q.npy',allow_pickle=False)
    agents=np.load(a.sample/'agent.npy',allow_pickle=False)
    scores=np.load(a.reference_predictions/'scores.npz',allow_pickle=False)
    rows=[json.loads(line) for line in (a.run/'predictions.jsonl').read_text().splitlines()]
    if len(rows)!=report['frames'] or len(rows)!=len(manifest['frames']):
        raise ValueError('Frame count differs')
    comparisons=[];passed=True;exact=True
    for row,record in zip(rows,manifest['frames']):
        if row['fid']!=record['fid'] or row['input_fids']!=record['input_fids'] or row['key_t']!=record['key_t']:
            raise ValueError('Temporal sampling differs')
        lo,hi=record['row_start'],record['row_end']
        path=a.sample/'cache'/(record['key']+'.pt')
        if sha(path)!=record['cache_sha256']:raise ValueError('Reference cache checksum differs')
        boxes=torch.load(path,map_location='cpu',weights_only=True)['boxes'].numpy()
        proposal_match=(np.array_equal(np.asarray(row['boxes'],np.float32).reshape(-1,4),boxes)
                        and np.array_equal(np.asarray(row['confidence'],np.float32),q[lo:hi])
                        and np.array_equal(np.asarray(row['agents'],np.int64),agents[lo:hi]))
        comparison={'fid':row['fid'],'candidates':hi-lo,'proposals_exact':proposal_match,'scores':{}}
        passed &= proposal_match;exact &= proposal_match
        for name in ['classification','contrastive','blend']:
            actual=np.asarray(row['scores'][name],np.float32).reshape(-1,184)
            expected=scores[name][lo:hi]
            if actual.shape!=expected.shape:raise ValueError('Score shape differs')
            error=np.abs(actual-expected)
            equal=np.array_equal(actual,expected)
            close=np.allclose(actual,expected,atol=1e-4,rtol=1e-3)
            top=np.array_equal(actual[:,98:].argmax(1),expected[:,98:].argmax(1))
            passed &= bool(close and top);exact &= equal
            comparison['scores'][name]={'exact':bool(equal),'allclose':bool(close),'triplet_top1_equal':bool(top),
                                        'max_abs_error':float(error.max()) if error.size else 0}
        comparisons.append(comparison)
    result={'passed':bool(passed),'exact':bool(exact),'frames':len(rows),'candidates':sum(c['candidates'] for c in comparisons),
            'score_values':sum(c['candidates'] for c in comparisons)*184*3,
            'max_abs_error':max(v['max_abs_error'] for c in comparisons for v in c['scores'].values()),
            'comparisons':comparisons,'deployment_predictions_sha256':sha(a.run/'predictions.jsonl'),
            'reference_scores_sha256':sha(a.reference_predictions/'scores.npz'),
            'reference_manifest_sha256':sha(a.sample/'manifest.json'),
            'scope':'Recorded120-frame qualitative sequence; numerical parity, not detector AP or domain-general validation.'}
    write_json(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='comparisons'}))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()

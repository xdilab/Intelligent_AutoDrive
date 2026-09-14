"""Trace parity on duplicate detections, tied scores, multilabel boxes and GT-only frames."""
import runpy
from pathlib import Path
import numpy as np
r=Path(__file__).parent
m=runpy.run_path(str(r/'trace.py'));core=runpy.run_path(str(r.parent/'post-proposal-experiments/baseline-core.py'))
rng=np.random.default_rng(16)
for trial in range(30):
    boxes=[];gts=[];scores=[]
    for f in range(5):
        n=int(rng.integers(0,8));g=int(rng.integers(0,4));pool=np.array([[0,0,.4,.4],[.5,.5,1,1],[.1,.1,.6,.6]],np.float32)
        b=pool[rng.integers(0,3,n)];gt=pool[rng.integers(0,3,g)];s=rng.choice(np.array([.1,.3,.8],np.float32),n)
        boxes.append(b);gts.append(gt);scores.append(s)
    offsets=np.r_[0,np.cumsum([len(b) for b in boxes])]
    if offsets[-1]==0:continue
    ious=[m['overlaps'](b,g) for b,g in zip(boxes,gts)];out=m['trace'](np.concatenate(scores),offsets,ious,[len(g) for g in gts],core['voc_ap'])
    gs=[np.c_[g,np.zeros(len(g))] for g in gts];ds=[np.c_[b,s] for b,s in zip(boxes,scores)]
    expected=core['evaluate_detections'](gs,[ds],['test'],.5)[1][0]
    assert abs(out['ap']-expected)<1e-4,(out['ap'],expected)
    assert (out['gt_det']>=0).sum()==out['tp'].sum()
print('PASS: 30 synthetic matching/ranking parity cases')

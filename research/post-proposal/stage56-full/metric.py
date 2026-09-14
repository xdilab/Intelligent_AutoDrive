import runpy
from pathlib import Path
import numpy as np
core=runpy.run_path(str(Path(__file__).parent/'baseline-core.py'))
HEADS=['agentness','agent','action','loc','duplex','triplet'];OFF=[0,1,11,33,49,98,184]
GTS=BOXES=SCORES=AGENTS=None;BASELINE=False
def one_class(task):
    group,c,name=task;gts=[];dets=[]
    for i,box in enumerate(BOXES):
        g=GTS[group][i];selected=g[g[:,-1]==c].copy();selected[:,-1]=0;gts.append(selected)
        if group==1 and not BASELINE:
            mask=AGENTS[i]==c;dets.append(np.concatenate([box[mask],SCORES[i][mask,0:1]],1))
        else:dets.append(np.concatenate([box,SCORES[i][:,OFF[group]+c:OFF[group]+c+1]],1))
    mean,ap,strings=core['evaluate_detections'](gts,[dets],[name],iou_thresh=.5)
    return group,c,float(ap[0]),strings[0]

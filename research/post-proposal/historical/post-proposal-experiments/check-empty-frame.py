"""Use the actual baseline functions to check the empty-frame handling."""
import ast,logging
from pathlib import Path
import numpy as np
p=Path('/data/repos/PedestrianIntent++/ROAD_plus_plus_Baseline/modules/evaluation.py')
names={'voc_ap','get_gt_of_cls','compute_iou','evaluate_detections','evaluate'}
tree=ast.parse(p.read_text());tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
ns={'np':np,'logger':logging.getLogger('test')};exec(compile(tree,str(p),'exec'),ns)
gt=[np.array([[0,0,1,1,0]],np.float32)]
det=[[np.array([[0,0,1,1,.9],[.1,.1,.2,.2,.8]],np.float32)]]
a=ns['evaluate_detections'](gt,det,['example'])
b=ns['evaluate_detections'](gt+[np.zeros((0,5),np.float32)], [det[0]+[np.zeros((0,5),np.float32)]],['example'])
assert a[0]==b[0] and np.array_equal(a[1],b[1]) and a[2]==b[2]
# A frame with GT but no predictions MUST still count toward recall.
c=ns['evaluate_detections'](gt+gt,[det[0]+[np.zeros((0,5),np.float32)]],['example'])
assert c[0] < a[0]
print('PASS: empty prediction/GT frame leaves AP unchanged; GT-only frame is retained and lowers recall/AP.')

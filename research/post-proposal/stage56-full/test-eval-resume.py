import tempfile,json,warnings
from pathlib import Path
import numpy as np
import torch
import train
class Model(torch.nn.Module):
 def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.));self.calls=0
 def forward(self,x,key):self.calls+=1;return x*self.weight,None,None
class Rows:
 def __init__(self,p):self.path=p;p.write_text('fixed-test-rows')
 def __len__(self):return 2
 def __getitem__(self,i):
  y=np.zeros((3,184),np.float32);y[i,98+i]=1
  return {'boxes':[[0,0,1,1]]*3,'targets':y,'key_t':0}
train.base.crops=lambda row,cfg:(torch.arange(len(row['targets'])*184,dtype=torch.float32).reshape(-1,184)/100,torch.tensor(row['targets']))
with tempfile.TemporaryDirectory() as tmp,warnings.catch_warnings():
 warnings.simplefilter('ignore');p=Path(tmp);rows=Rows(p/'rows');m=Model();first=train.evaluate(m,rows,{},p/'cache');assert m.calls==2
 second=train.evaluate(m,rows,{},p/'cache');assert m.calls==2 and first==second
 (p/'cache/000001.npz').unlink();third=train.evaluate(m,rows,{},p/'cache');assert m.calls==3 and third==first
 with torch.no_grad():m.weight.add_(1)
 try:train.evaluate(m,rows,{},p/'cache')
 except AssertionError:pass
 else:raise AssertionError('Stale model cache accepted')
print('PASS: identical cached AP, resume only missing frames, reject changed model')

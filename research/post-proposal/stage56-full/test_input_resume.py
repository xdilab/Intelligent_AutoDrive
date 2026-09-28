from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from contextlib import nullcontext
import numpy as np
import torch
import eval_input as e
with TemporaryDirectory() as td:
 p=Path(td);(p/'v_1.npy').write_bytes(b'existing cache preserved')
 keys=['v_1','v_2','v_3'];d={s:{'boxes_xyxyn':np.zeros((n,4))} for s,n in zip(keys,[2,0,65])};ann={'db':{'v':{'numf':3}}}
 todo=list(e.rows(keys,d,ann,p));assert [r[1] for r in todo]==['v_2','v_3'];assert todo[-1][2]['fids']==[1,1,2,3,3,3,3,3]
 class L:
  def __init__(self,*args):self.reads=0;self.hits=0
  def load(self,row):self.reads+=8;return np.zeros((8,2,2,3),np.uint8)
 calls=[]
 def crop(ft,boxes):calls.append(len(boxes));return torch.zeros(len(boxes),1)
 def model(x,key):return torch.zeros(len(x),184),None,None
 with patch.object(e,'ClipLoader',L),patch.object(e,'normalize_clip',lambda x:x),patch.object(e,'crop_batch',crop),patch.object(torch,'autocast',lambda *a,**k:nullcontext()):
  result=list(e.predictions(model,keys,d,ann,{},p))
 assert [x[1] for x in result]==['v_2','v_3'];assert result[0][2].shape==(0,184);assert result[1][2].shape==(65,184);assert calls==[64,1];assert (p/'v_1.npy').read_bytes()==b'existing cache preserved'
print('PASS: cache skip, temporal boundary clamping, empty frames, 64+1 batch ordering')

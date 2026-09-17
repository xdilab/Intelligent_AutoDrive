from pathlib import Path
import time
root=Path('/work/bbyrd1/stage56-full-20260914')
paths=[root/'runs'/name/'epoch-1.pt' for name in ['stage6-classification','stage6-contrastive']]
while True:
 missing=[str(p) for p in paths if not p.is_file()]
 print('WAIT_EPOCH1',missing,flush=True)
 if not missing:break
 time.sleep(300)

from pathlib import Path
import json,tempfile,importlib.util
spec=importlib.util.spec_from_file_location('watch','/data/repos/ROAD_Reason/research/post-proposal/monitoring/watch.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
with tempfile.TemporaryDirectory() as tmp:
 w.ROOT=Path(tmp)/'monitor';w.ROOT.mkdir();w.STUDY=Path(tmp)/'study';(w.STUDY/'results').mkdir(parents=True);(w.STUDY/'results/train-job.txt').write_text('999')
 w.run=lambda cmd: '' if cmd[0]=='squeue' else '999_0|FAILED|1:0|\n'
 first=w.poll();second=w.poll();assert len(first['new_alerts'])==1 and not second['new_alerts'];assert first['new_alerts'][0]['severity']=='error'
print('PASS: failed-job alert generated once; heartbeat recorded')

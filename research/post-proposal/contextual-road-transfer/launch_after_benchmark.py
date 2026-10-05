"""Submit authorized full DAG once runtime/input gates pass; record measured ETA basis."""
from pathlib import Path
import json,subprocess,time,sys,math
from prepare_full import atomic
ROOT=Path('/work/bbyrd1/contextual-road-transfer-20261005')
def main():
 launch=json.loads((ROOT/'results/full-launch.json').read_text())
 assert set(launch['jobs'])=={'prepare','benchmark'},'Existing/partial full submission: inspect rather than duplicate.'
 benchmark=json.loads((ROOT/'results/benchmark.json').read_text());audit=json.loads((ROOT/'results/input-audit.json').read_text());assert benchmark['passed'] and audit['passed']
 assert len(benchmark['arms'])==3
 work=[]
 for arm,bench in zip(['frozen','classification','contrastive'],benchmark['arms']):
  assert bench['passed'] and bench['audit']['condition']==arm
  meta=json.loads((ROOT/arm/'preparation.json').read_text());n=sum(v['rows'] for v in meta['splits'].values());rate=bench['crops_per_second_including_scene_and_io'];assert meta['passed'] and math.isfinite(rate) and rate>0
  work.append({'arm':arm,'rows':n,'sampled_rows_per_second':rate,'estimated_gpu_hours':n/rate/3600})
 floor=sum(x['estimated_gpu_hours'] for x in work)/8
 eta={'time':time.time(),'basis':work,'eight_gpu_extraction_floor_hours':floor,'remaining':'three complete encoder caches; compaction/preflight;18 heads;9 blends;27 detector evaluations','first_development_hours_after_extraction_start':[round(floor*1.5+2,1),round(floor*3+6,1)],'full_detector_hours_after_extraction_start':[round(floor*1.5+6,1),round(floor*3+18,1)],'queue':'Add unbounded scheduling/startup delay; eight simultaneous GPUs are a ceiling, not guaranteed. Audit/benchmark elapsed precedes these ranges.','uncertainty':'Provisional extrapolation from8frames per encoder; real crop density/cache writes/shared filesystem and downstream throughput unmeasured. Refresh after first extraction progress and each phase.','cache_hours_after_start':[round(floor*1.5,1),round(floor*3,1)]}
 atomic(eta,ROOT/'results/eta.json');print('MEASURED_PROVISIONAL_ETA',json.dumps(eta),flush=True)
 subprocess.run([sys.executable,str(ROOT/'code/submit.py'),'--phase','full'],check=True)
if __name__=='__main__':main()

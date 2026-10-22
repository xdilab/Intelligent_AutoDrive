"""Paired raw-video timing plus isolated, complete-video packaged inference."""
from common import *
import subprocess
def main():
 verify();(ROOT/'runtime').mkdir(exist_ok=True)
 for mode in ['student','teacher']:
  if not (ROOT/'runtime'/mode/'report.json').exists():subprocess.run([sys.executable,str(HERE/'runtime_worker.py'),mode],check=True)
 a=read(ROOT/'runtime/student/report.json');b=read(ROOT/'runtime/teacher/report.json');assert a['video_sha256']==b['video_sha256'];assert [r['candidate_sha256'] for r in a['records']]==[r['candidate_sha256'] for r in b['records']]
 reports=[]
 for video in read(ROOT/'split-manifest.json')['groups']['train'][:3]:
  dest=ROOT/'runtime'/f'continuous-{video}';bundle=ROOT/'bundle-seed0'
  if not (dest/'report.json').exists():
   assert not dest.exists(),'Incomplete standalone output: preserve and diagnose before rerun'
   subprocess.run([sys.executable,str(bundle/'run.py'),'--bundle',str(bundle),'--video',str(p.ROOT/'videos'/f'{video}.mp4'),'--output',str(dest),'--verify-isolation'],check=True)
  report=read(dest/'report.json');assert report['causal_source_indices_verified'] and report['external_assets_blocked'] and report['frames']>32
  allocated=[r['gpu_allocated'] for r in report['records'][32:]];reserved=[r['gpu_reserved'] for r in report['records'][32:]]
  assert max(allocated)-min(allocated)<128*1024**2,'Unexpected retained GPU-memory growth after warmup'
  reports.append(dict(memory_bound_passed=True,memory_bound_bytes=128*1024**2,video=video,frames=report['frames'],report=str(dest/'report.json'),post_warmup_allocated_range_bytes=[min(allocated),max(allocated)],post_warmup_reserved_range_bytes=[min(reserved),max(reserved)],tail_allocated_bytes=allocated[-1],processing_seconds=report['processing_seconds'],load_seconds=report['load_seconds']))
 # Production packaged output equals instrumented student scores on the same raw input.
 video=read(ROOT/'split-manifest.json')['groups']['train'][0];prod=(ROOT/'runtime'/f'continuous-{video}'/'predictions.jsonl').open();bench=(ROOT/'runtime/student/predictions.jsonl').open();diff=0.
 for x,y in zip(prod,bench):
  x,y=json.loads(x),json.loads(y);assert x['fid']==y['fid'];assert np.array_equal(x['boxes'],y['boxes']);diff=max(diff,float(np.max(np.abs(np.array(x['scores'])-np.array(y['scores'])))) if x['scores'] else 0.)
 prod.close();bench.close();assert diff<=1e-7
 save(ROOT/'runtime-summary.json',dict(student=a,teacher=b,continuous=reports,paired_candidates_exact=True,packaged_instrumented_max_abs=diff,teacher_to_student_median_processing_ratio=b['summary']['total_seconds']['median']/a['summary']['total_seconds']['median'],interpretation='Processing throughput is not sensor-to-decision live latency. Teacher additionally requires0.4seconds of future data at10Hz. Serial file processing excludes camera capture/display and is not a safety/deployment claim.',time=time.time()));progress('runtime-audit-complete',continuous_frames=sum(r['frames'] for r in reports),parity_max_abs=diff)
if __name__=='__main__':main()

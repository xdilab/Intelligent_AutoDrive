from assoc_common import *
import subprocess

def main():
 verify();strength=read(ROOT/'development-selection.json')['strengths']['association'];bundle=BASE_ROOT/'bundle-seed0';reports=[]
 ck=torch.load(checkpoint(0),map_location='cpu',weights_only=False)['model'];assert tensor_hash(ck)==tensor_hash(torch.load(bundle/'student.pt',map_location='cpu',weights_only=True))
 assert sha(bundle/'model.py')==sha(OLD/'model.py')
 for video in read(ROOT/'split-manifest.json')['groups']['train'][:3]:
  dest=ROOT/'runtime'/video
  if not (dest/'report.json').exists():subprocess.run([sys.executable,str(HERE/'run_video.py'),'--bundle',str(bundle),'--rules',str(ROOT/'rules.json'),'--strength',str(strength),'--video',str(p.ROOT/'videos'/f'{video}.mp4'),'--output',str(dest)],check=True)
  r=read(dest/'report.json');assert r['frames']>32 and r['causal_source_indices_verified'] and r['external_assets_blocked'];assert r['model_tensor_sha256']==tensor_hash(ck);assert r['correction_code_sha256']==sha(HERE/'correction.py');samples=r['records'][32:]
  assert max(x['gpu_allocated'] for x in samples)-min(x['gpu_allocated'] for x in samples)<128*1024**2
  assert max(x['cpu_rss'] for x in samples)-min(x['cpu_rss'] for x in samples)<256*1024**2
  reports.append(r);progress('runtime-video-complete',video=video,frames=r['frames'])
 allrows=[x for r in reports for x in r['records'][10:]];summary={k:dict(median_ms=1000*float(np.median([r[k] for r in allrows])),p95_ms=1000*float(np.percentile([r[k] for r in allrows],95))) for k in ['decode','detector','student','correction','export','total']}
 save(ROOT/'runtime-summary.json',dict(time=time.time(),videos=len(reports),frames=sum(r['frames'] for r in reports),timed_after_warmup=len(allrows),summary=summary,total_fps=sum(r['frames'] for r in reports)/sum(r['processing_seconds'] for r in reports),load_seconds=[r['load_seconds'] for r in reports],max_gpu_bytes=max(r['peak_gpu_bytes'] for r in reports),checkpoint_bundle_tensor_exact=True,shared_correction_code_exact=True,source_bundle=str(bundle),source_bundle_sha256=sha(bundle/'bundle.json'),no_teacher_or_feature_assets=True,reset_at_each_video=True,memory_bounds_passed=True,limitation='Runtime includes compressed prediction export. Official benchmark used all fixed candidates; raw video uses production confidence0.25/NMS0.7/max300. No end-to-end real-time claim.'))
if __name__=='__main__':main()

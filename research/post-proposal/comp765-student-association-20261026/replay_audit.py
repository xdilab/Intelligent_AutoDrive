"""Independent NumPy replay against previously verified raw-video baseline outputs.
Reference predictions are audit inputs only; the deployed adapter never reads them.
"""
from assoc_common import *
import gzip
from torchvision.ops import nms

def reference(base,rule,strength):
 box=np.asarray(base['boxes'],np.float32).reshape(-1,4);q=np.asarray(base['confidence'],np.float32);agents=np.asarray(base['agents'],int);scores=np.asarray(base['scores'],np.float32).reshape(-1,184)
 if not len(box):return scores
 prob=scores[:,49:]/q[:,None];events=np.asarray(rule['event_agents']);w=np.asarray(rule['weights'],np.float32);eligible=np.flatnonzero(q>=.25)
 if not len(eligible):return scores
 # Match the declared GPU NMS tie policy; all remaining arithmetic is independent NumPy.
 sel=eligible[nms(torch.from_numpy(box[eligible]).cuda(),torch.from_numpy(q[eligible]).cuda(),.5).cpu().numpy()[:30]]
 lo=np.maximum(box[:,None,:2],box[None,sel,:2]);hi=np.minimum(box[:,None,2:],box[None,sel,2:]);inter=np.maximum(hi-lo,0).prod(-1);areas=np.maximum(box[:,2:]-box[:,:2],0).prod(-1);iou=inter/np.maximum(areas[:,None]+areas[None,sel]-inter,1e-12);iom=inter/np.maximum(np.minimum(areas[:,None],areas[None,sel]),1e-12);centers=(box[:,:2]+box[:,2:])/2;distance=np.linalg.norm(centers[:,None,:]-centers[None,sel,:],axis=-1);adj=(distance<=.5)&(iou<.3)&(iom<.5);adj[np.arange(len(box))[:,None]==sel[None,:]]=False
 support=prob[sel]*q[sel,None]*(agents[sel,None]==events[None,:]);evidence=(adj[:,:,None]*support[None,:,:]).max(1);delta=(evidence@w)/(1+evidence@(w!=0).astype(np.float32));delta*=agents[:,None]==events[None,:]
 scale=np.exp(strength*delta);adjusted=prob*scale/((1-prob)+prob*scale);result=scores.copy();result[:,49:]=adjusted*q[:,None];return result

def main():
 setup();verify();rules=read(ROOT/'rules.json');strength=read(ROOT/'development-selection.json')['strengths']['association'];reports=[];maximum=0.
 for video in read(ROOT/'split-manifest.json')['groups']['train'][:3]:
  old=BASE_ROOT/'runtime'/f'continuous-{video}';new=ROOT/'runtime'/video;a=read(old/'report.json');b=read(new/'report.json');assert a['source_sha256']==b['source_sha256'] and a['bundle_sha256']==b['bundle_sha256'] and a['frames']==b['frames'];err=0.;count=0
  with (old/'predictions.jsonl').open() as first,gzip.open(new/'predictions.jsonl.gz','rt') as second:
   for l1,l2 in zip(first,second,strict=True):
    base=json.loads(l1);actual=json.loads(l2)
    for key in ['fid','source_index','boxes','agents','confidence']:assert base[key]==actual[key],(video,key,base['fid'])
    expected=reference(base,rules,strength);observed=np.asarray(actual['scores'],np.float32).reshape(-1,184);assert expected.shape==observed.shape
    if expected.size:err=max(err,float(np.max(np.abs(expected-observed))));assert np.array_equal(expected[:,:49],observed[:,:49])
    count+=1
  assert err<=1e-6,(video,err);assert count==a['frames'];maximum=max(maximum,err);reports.append(dict(video=video,frames=count,max_abs_score_error=err,reference_predictions_sha256=sha(old/'predictions.jsonl'),adapter_predictions_sha256=sha(new/'predictions.jsonl.gz')))
 save(ROOT/'replay-audit.json',dict(passed=True,time=time.time(),frames=sum(r['frames'] for r in reports),max_abs_score_error=maximum,tolerance=1e-6,exact_candidates_and_primitives=True,reports=reports,method='Independent NumPy evidence/association/odds-update replay from the original verified raw-video baseline scores; GPU NMS tie policy shared. Historical predictions are audit references only, never adapter dependencies.'))
 save(ROOT/'adapter-manifest.json',dict(time=time.time(),format=1,base_bundle=str(BASE_ROOT/'bundle-seed0'),base_bundle_sha256=sha(BASE_ROOT/'bundle-seed0/bundle.json'),student_checkpoint_sha256=sha(checkpoint(0)),rules=str(ROOT/'rules.json'),rules_sha256=sha(ROOT/'rules.json'),strength=strength,code_sha256={str(HERE/n):sha(HERE/n) for n in ['run_video.py','correction.py']},teacher_required=False,cached_features_required=False,future_frames_required=False,original_bundle_modified=False))
 progress('independent-runtime-replay-passed',frames=sum(r['frames'] for r in reports),max_abs=maximum)
if __name__=='__main__':main()

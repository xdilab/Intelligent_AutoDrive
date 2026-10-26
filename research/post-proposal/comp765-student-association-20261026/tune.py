"""One preregistered development selection; never accesses final predictions."""
from assoc_common import *
from correction import Correction

def ap_rows(y,s):
 return [100*p.average_precision_score(y[:,i],s[:,i]) if y[:,i].sum() else None for i in range(y.shape[1])]
def main():
 setup();assert read(ROOT/'preflight.json')['passed'];rules=read(ROOT/'rules.json');c=Correction(rules,'cuda');manifest=read(BASE_ROOT/'data-manifest.json');grid=read(HERE/'protocol.json')['selection']['grid'];results=[];started=time.perf_counter();hashes={}
 for seed in SEEDS:
  m=Student().cuda().eval();m.load_state_dict(torch.load(checkpoint(seed),map_location='cpu',weights_only=False)['model']);ys=[];zs=[];qs=[];ds=[];fs=[];counts=[]
  with torch.inference_mode():
   for rec in manifest['sequences']:
    if rec['split']!='selection':continue
    assert sha(rec['path'])==rec['sha256'];hashes[rec['path']]=rec['sha256'];seq=torch.load(rec['path'],map_location='cpu',weights_only=False);state=None
    for row0 in seq:
     row=to_gpu(row0);z,state=m(row['roi'],row['scene'],row['boxes'],row['agents'],state)
     if not row.get('annotated',True) or not len(z):continue
     d=c.delta(z,row['boxes'],row['agents'],row['q'],'association');f=c.delta(z,row['boxes'],row['agents'],row['q'],'frequency');ys.append(row['labels'][:,49:].cpu());zs.append(z[:,49:].cpu());qs.append(row['q'].cpu());ds.append(d.cpu());fs.append(f.cpu());counts.append(len(z))
    progress('development-predictions',seed=seed,video=rec['video'],elapsed=time.perf_counter()-started)
  y=torch.cat(ys).numpy();z=torch.cat(zs);q=torch.cat(qs);delta={'association':torch.cat(ds),'frequency':torch.cat(fs)}
  np.savez_compressed(ROOT/f'development-seed{seed}.npz',labels=y,logits=z.numpy(),q=q.numpy(),association=delta['association'].numpy(),frequency=delta['frequency'].numpy(),frame_candidate_counts=np.array(counts))
  for condition in ['association','frequency']:
   for strength in grid:
    scores=((z+strength*delta[condition]).sigmoid()*q[:,None]).numpy();aps=ap_rows(y,scores)
    results.append(dict(seed=seed,condition=condition,strength=strength,triplet_proposal_AP=float(np.mean([x for x in aps[49:] if x is not None])),duplex_proposal_AP=float(np.mean([x for x in aps[:49] if x is not None])),per_class_AP=aps,proposals=len(y),supported_triplets=sum(x is not None for x in aps[49:]),supported_duplexes=sum(x is not None for x in aps[:49])))
 strengths={};aggregate=[]
 for condition in ['association','frequency']:
  best=-float('inf');chosen=0
  for strength in grid:
   score=float(np.mean([r['triplet_proposal_AP'] for r in results if r['condition']==condition and r['strength']==strength]));aggregate.append(dict(condition=condition,strength=strength,mean_triplet_proposal_AP=score))
   if score>best+1e-10:chosen=strength;best=score
  strengths[condition]=chosen
 save(ROOT/'development-selection.json',dict(time=time.time(),metric='Development proposal AP (%), NOT official detector AP',results=results,aggregate=aggregate,strengths=strengths,seconds=time.perf_counter()-started,selection_cache_hashes=hashes))
 progress('development-selection-complete',strengths=strengths,seconds=time.perf_counter()-started)
if __name__=='__main__':main()

"""Frozen-checkpoint diagnostic; selected classes, full evaluation population.
Matches the baseline's per-frame greedy IoU matching, global ordering and VOC AP.
No score thresholds are fitted. Top-Ngt budgets are descriptive and not deployable.
"""
import argparse,csv,hashlib,json,pickle,runpy,time
from pathlib import Path
import numpy as np
import torch

CLASSES=['MedVeh-Stop-VehLane','Bus-Stop-VehLane','LarVeh-Stop-Jun','Ped-XingFmRht-xing','Ped-MovTow-LftPav','Bus-MovAway-OutgoLane']
VARIANTS=['head-flat','head-phrase','stage5','stage6']

def overlaps(a,b):
    if not len(a) or not len(b):return np.zeros((len(a),len(b)),np.float32)
    inter=np.maximum(np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]),0).prod(2)
    return inter/np.maximum((a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None,:]-inter,1e-15)

def trace(scores, offsets, ious, ngts, voc_ap):
    # Preserve the evaluator's frame-local order before its global np.argsort.
    ordered=[]; matches=[]; gtbase=0
    for f,mat in enumerate(ious):
        lo,hi=offsets[f:f+2]; order=np.argsort(-scores[lo:hi]);ids=lo+order
        match=np.full(len(order),-1,np.int64);remaining=list(range(ngts[f]))
        if remaining:
            for k,d in enumerate(order):
                if not remaining:break
                j=int(np.argmax(mat[d,remaining]));target=remaining[j]
                if mat[d,target]>=.5:match[k]=gtbase+target;remaining.pop(j)
        ordered.append(ids);matches.append(match);gtbase+=ngts[f]
    ordered=np.concatenate(ordered);matches=np.concatenate(matches)
    rankorder=np.argsort(-scores[ordered]);ids=ordered[rankorder];matched=matches[rankorder]
    tp=matched>=0; ctp=np.cumsum(tp);precision=ctp/np.arange(1,len(tp)+1);recall=ctp/max(gtbase,1)
    ap=float(voc_ap(recall,precision))
    gt_rank=np.full(gtbase,len(scores)+1,np.int64);gt_det=np.full(gtbase,-1,np.int64)
    hit=np.flatnonzero(tp);gt_rank[matched[hit]]=hit+1;gt_det[matched[hit]]=ids[hit]
    return {'ap':ap,'gt_rank':gt_rank,'gt_det':gt_det,'ids':ids,'matched':matched,'tp':tp,'ngt':gtbase}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    mod=runpy.run_path(str(a.study/'code/evaluate-study.py'));core=mod['core'];manifest=json.loads((a.study/'code/shared-frames.json').read_text());keys=manifest['frames'];labels=manifest['labels']['triplet'];cols=[98+labels.index(c) for c in CLASSES]
    yolo=pickle.load(open(a.study/'data/dets_v8x_best_val_fullcand.pkl','rb'))['records']
    actual=hashlib.sha256(b''.join(s.encode()+np.asarray(yolo[s]['boxes_xyxyn']).tobytes()+np.asarray(yolo[s]['conf']).tobytes()+np.asarray(yolo[s]['cls']).tobytes() for s in keys)).hexdigest();assert actual==manifest['candidate_sha256']
    assert len(keys)==36717 and hashlib.sha256(('\n'.join(keys)+'\n').encode()).hexdigest()==manifest['frame_sha256']
    boxes=[yolo[s]['boxes_xyxyn'].astype(np.float32) for s in keys];offsets=np.r_[0,np.cumsum([len(b) for b in boxes])];q=np.concatenate([yolo[s]['conf'].astype(np.float32) for s in keys]);N=len(q);del yolo
    baseline=pickle.load(open(a.study/'data/dets_i3d_val_fullcand.pkl','rb'))['records'];gt_byclass=[[] for _ in CLASSES];scale=np.array([840,600,840,600],np.float32)
    for s in keys:
        gt=baseline[mod['key'](s)]['gt'];gb=gt['boxes'].astype(np.float32)/scale if gt is not None else np.empty((0,4),np.float32)
        for ci,c in enumerate(CLASSES):gt_byclass[ci].append(gb[gt['triplet'][:,labels.index(c)]>0] if len(gb) else gb)
    del baseline
    feats=pickle.load(open(a.study/'data/crop_feats_val.pkl','rb'))['feats'];print('LOADED',N,'candidates',flush=True)
    paths={};checkpoint_hashes={}
    for seed in range(3):
        models={v:(mod['load_head'] if v.startswith('head') else mod['load_mlp'])(a.study/f'runs/seed-{seed}/{v}.pt').cuda() for v in VARIANTS}
        for v in VARIANTS:
            p=a.study/f'runs/seed-{seed}/{v}.pt';checkpoint_hashes[f'{seed}/{v}']=hashlib.sha256(p.read_bytes()).hexdigest()
        arrays={v:np.lib.format.open_memmap(a.out/f'seed{seed}-{v}-scores.npy',mode='w+',dtype=np.float32,shape=(N,len(CLASSES))) for v in VARIANTS}
        # Batch adjacent whole frames; preserve per-frame candidate row identity.
        with torch.no_grad():
            for start in range(0,len(keys),64):
                stop=min(start+64,len(keys));parts=[]
                for i in range(start,stop):
                    n=offsets[i+1]-offsets[i];f=feats.get(keys[i],np.empty((0,1024),np.float32));assert f.shape==(n,1024) and np.isfinite(f).all();parts.append(f)
                x=torch.from_numpy(np.concatenate(parts).astype(np.float32)).cuda();lo,hi=offsets[start],offsets[stop]
                if hi==lo:continue
                flat=models['head-flat'](x).sigmoid();phrase=models['head-phrase'](x).sigmoid();s5=models['stage5'](torch.cat([flat[:,:49],x],1)).sigmoid();s6=models['stage6'](torch.cat([flat[:,:49],phrase[:,49:],x],1)).sigmoid()
                raw={'head-flat':flat[:,cols],'head-phrase':phrase[:,cols],'stage5':s5[:,np.array(cols)-49],'stage6':s6[:,np.array(cols)-49]}
                for v in VARIANTS:arrays[v][lo:hi]=raw[v].cpu().numpy()*q[lo:hi,None]
                if start%4096==0:print('SCORING',seed,start,len(keys),flush=True)
        for v,arr in arrays.items():arr.flush();paths[seed,v]=a.out/f'seed{seed}-{v}-scores.npy'
        del arrays,models;torch.cuda.empty_cache()
    del feats
    summary=[];examples=[];budgets=[.5,1.,2.];parity=[]
    for ci,c in enumerate(CLASSES):
        gts=gt_byclass[ci];ious=[overlaps(b,g) for b,g in zip(boxes,gts)];ngts=[len(g) for g in gts];gtframe=np.repeat(np.arange(len(keys)),ngts);gtlocal=np.concatenate([np.arange(n) for n in ngts]);G=sum(ngts)
        for seed in range(3):
            traces={};scores={}
            for v in VARIANTS:
                scores[v]=np.load(paths[seed,v],mmap_mode='r')[:,ci];t=trace(scores[v],offsets,ious,ngts,core['voc_ap']);traces[v]=t
                expected=json.loads((a.study/f'results/metrics/seed{seed}-{v}.json').read_text())['ap_values']['triplet'][labels.index(c)];error=t['ap']-expected;parity.append({'seed':seed,'class':c,'variant':v,'ap':t['ap'],'expected_ap':expected,'delta_pp':error})
                print('AP',seed,c,v,t['ap'],'error',error,flush=True)
            for factor in budgets:
                K=min(N,max(1,int(G*factor)));hit={v:t['gt_rank']<=K for v,t in traces.items()};op=hit['head-phrase'] & ~hit['stage5'];preserved=op & hit['stage6'];lost=op & ~hit['stage6'];harm=hit['stage5'] & ~hit['stage6'];rescue=~hit['stage5'] & hit['stage6']
                summary.append({'seed':seed,'class':c,'gt_count':G,'candidate_count':N,'budget_factor':factor,'budget':K,**{v+'_tp':int(h.sum()) for v,h in hit.items()},'phrase_only_opportunities':int(op.sum()),'preserved_by_stage6':int(preserved.sum()),'lost_by_stage6':int(lost.sum()),'stage5_hits_lost':int(harm.sum()),'stage6_rescues':int(rescue.sum()),**{v+'_fp':int(K-h.sum()) for v,h in hit.items()}})
                if factor!=1.:continue
                for category,mask in [('phrase_help_preserved',preserved),('phrase_help_lost',lost),('stage5_hit_lost',harm),('stage6_rescue',rescue)]:
                    # One example per source video per class/seed/category; deterministic rank-gap ordering.
                    inds=np.flatnonzero(mask);gap=traces['stage5']['gt_rank']-traces['head-phrase']['gt_rank'];inds=inds[np.argsort(-gap[inds],kind='stable')];seen=set();picked=0
                    for g in inds:
                        fi=int(gtframe[g]);stem=keys[fi];video=stem.rsplit('_',1)[0]
                        if video in seen:continue
                        seen.add(video);local=int(gtlocal[g]);anchor=int(traces['head-phrase']['gt_det'][g]);anchor=anchor if anchor>=0 else int(traces['stage5']['gt_det'][g]);di=anchor-int(offsets[fi]) if anchor>=0 else -1
                        examples.append({'seed':seed,'class':c,'category':category,'frame':stem,'gt_local':local,'gt_box':gts[fi][local].tolist(),'candidate_index':di,'candidate_box':boxes[fi][di].tolist() if di>=0 else None,'candidate_iou':float(ious[fi][di,local]) if di>=0 else None,'budget':K,**{v+'_gt_rank':int(t['gt_rank'][g]) for v,t in traces.items()},**{v+'_score_same_candidate':float(sc[anchor]) if anchor>=0 else None for v,sc in scores.items()}});picked+=1
                        if picked==3:break
    def write_csv(name,rows):
        with (a.out/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_csv('ap-parity.csv',parity);write_csv('budget-summary.csv',summary)
    (a.out/'examples.json').write_text(json.dumps(examples,indent=2)+'\n')
    max_error=max(abs(r['delta_pp']) for r in parity)
    result={'classes':CLASSES,'seeds':[0,1,2],'n_frames':len(keys),'candidates':N,'candidate_sha256':actual,'frame_sha256':manifest['frame_sha256'],'checkpoint_sha256':checkpoint_hashes,'max_ap_error_pp':max_error,'parity_tolerance_pp':.02,'parity_pass':max_error<=.02,'scope':'Six disclosed post-hoc classes; all evaluation frames/candidates; descriptive not a gate-training set. Budget proportional to evaluation GT counts, not deployable threshold. Per-GT ranks permit different winning boxes; same-candidate scores separately recorded.','completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    (a.out/'report.json').write_text(json.dumps(result,indent=2)+'\n');assert result['parity_pass'],result
    print('COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()

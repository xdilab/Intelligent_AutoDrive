import json,pathlib
W=pathlib.Path('/data/repos/wiki/artifacts/defense-20260928');p=json.load(open(W/'raw-populated.json'))['structuredContent'];plan=json.load(open(W/'slide-plan.json'));ids=[x['page_id'] for x in plan];by={x['objectId']:x for x in p['slides']};assert all(i in by for i in ids)
plan[19]['caption']='DCB leads completed contextual triplet means; focal blend retains better tail AP.'
plan[19]['notes']+=' DCB’s primary endpoint, standalone tail47 improvement over focal, was not met.'
plan[20]['caption']='FRCB: selected result; ours: 3-seed mean. Protocols and proposals differ.'
plan[21]['caption']='As of Sep 28, 2026 · seed 0  |  Final selected Stage 6 detector evaluation pending'
plan[22]['notes']=plan[22]['notes'].replace('Full encoder adaptation is still pending and does not underpin the completed contextual result.','Final selected-checkpoint Stage 6 detector evaluation remains pending; it does not underpin the completed contextual result.')
plan[23]['notes']+=' Stage 6 versus contextual blend compares different architectures and training studies; it does not isolate context, language, or contrastive effects.'
req=[]
for n in [20,21,22]:
 oid=f'd{n:02d}_p15_i177';req.extend([{'deleteText':{'objectId':oid,'textRange':{'type':'ALL'}}},{'insertText':{'objectId':oid,'text':plan[n-1]['caption']}}])
for s in plan:
 oid=by[s['page_id']]['slideProperties']['notesPage']['notesProperties']['speakerNotesObjectId'];req.extend([{'deleteText':{'objectId':oid,'textRange':{'type':'ALL'}}},{'insertText':{'objectId':oid,'text':s['notes']}}])
# Keep all new slides, then remove only the 18 unchanged template exemplars.
req.append({'updateSlidesPosition':{'slideObjectIds':ids,'insertionIndex':0}})
for s in p['slides']:
 if s['objectId'] not in ids:req.append({'deleteObject':{'objectId':s['objectId']}})
(W/'slide-plan.json').write_text(json.dumps(plan,indent=2));(W/'finalize-requests.json').write_text(json.dumps(req))
print(len(req),'requests; verified',len(ids),'new slide IDs')

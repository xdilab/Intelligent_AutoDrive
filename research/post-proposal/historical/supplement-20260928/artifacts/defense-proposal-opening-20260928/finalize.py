import json,pathlib,re
W=pathlib.Path('/data/repos/wiki/artifacts/defense-proposal-opening-20260928');p=json.load(open(W/'proposal.json'))['structuredContent'];d=json.load(open(W/'populated.json'))['structuredContent'];b=json.load(open(W/'defense-before.json'))['structuredContent'];s={x['objectId']:x for x in d['slides']}
ids=[f'def_{i:02d}' for i in range(1,12)]+[f'rev_{i:02d}' for i in range(12,26)]
assert all(x in s for x in ids)
req=[{'deleteObject':{'objectId':f'def_{i:02d}'}} for i in range(12,26)]
req += [{'updateSlidesPosition':{'slideObjectIds':[x],'insertionIndex':0}} for x in reversed(ids)]
def notes(slide):
 es=slide['slideProperties']['notesPage']['pageElements'];return ''.join(t.get('textRun',{}).get('content','') for e in es for t in e.get('shape',{}).get('text',{}).get('textElements',[]))
opening=[
'Introduce the thesis: long-tail compositional road-event detection. A decoupled hybrid combines learned and semantic composition on ROAD-Waymo.',
'Follow the proposal opening: motivation, prior work, benchmark, method progression, and research direction. The later defense sections report the completed studies and remaining limitations. Visible outline wording is preserved verbatim at Brandon’s request.',
'The goal is safer decisions through understanding road users, behavior, and context. This is observed event recognition, not future intention prediction.',
'Three challenges: simultaneous agents, an ego vehicle whose viewpoint changes, and behavior that unfolds over time.',
'JAAD and PIE inform pedestrian understanding. Our task instead labels observed events across multiple kinds of road agents. Do not equate crossing outcome, observed action, and mental intent.',
'ROAD defines the event task; ROAD-R adds logical constraints; ROAD-Waymo expands geographic coverage. Long-tail performance remains a limitation.',
'ROAD-Waymo: 600 training and 198 validation videos. Ten agents, 22 actions, 16 locations, 49 duplexes and 86 triplets. Localize agents and score observed events.',
'Keep one example throughout: the gold box is a ground-truth Car-Stop-VehLane event in train_00407, frame00001. It is an annotated example, not a detector prediction.',
'Common parts can form rare combinations. Tail partitions come from training frequencies: 47 tail classes and 28 deep-tail classes nested within those 47. Ask whether visual and semantic evidence can improve recognition.',
'The proposed solution combines visual evidence and semantic relationships, measuring gains on rare events and effects on common events. This original proposal wording motivates the completed experiments that follow.',
'An eight-frame clip produces boxes and 184 scores per object: 1 agentness +10 agents +22 actions +16 locations +49 duplexes +86 triplets. This is multilabel; multiple labels can hold simultaneously. The image stack is illustrative, not an evaluated clip.'
]
choices=[9,9,10,11,12,13,15,16,18,19,20,21,23,24]
minutes=[25,20,45,45,55,55,50,50,55,45,60,80,80,80,90,75,90,125,85,105,85,115,85,65,30]
minutes[17]+=1800-sum(minutes)
assert sum(minutes)==1800
rows=[]
for i,id in enumerate(ids,1):
 if i<=11:
  text=opening[i-1]+'\n\nContent source: proposal '+p['slides'][i-1]['objectId']+'; https://docs.google.com/presentation/d/1_ywXj_hqlgdC0S68aH3kGAteT5UQ4oH3Q5-Pg6s_TeM/edit . Wording and meaning-bearing figures preserved; defense styling applied.'
 else:
  old=choices[i-12];text=notes(b['slides'][old-1])
  text=re.sub(r'^\s*(?:Opening, )?\d+ seconds\.?\s*','',text)
  if i==12:text='Stage 4: Semantic Crop Classifier. Frozen video and text encoders supply crop features and label phrases. Cosine similarity produces phrase scores. No Composition MLP. YOLO supplies candidate boxes at inference; head training uses GT-aligned supervision and negatives. Source: artifacts/defense-30min/alignment/stage4.drawio. This is the first architecture in the revised defense.'
  if i==18:text=notes(b['slides'][13])+'\n\n'+text+'\nDCB failed its primary standalone tail47-improvement endpoint; its blend improves overall triplet while the focal blend remains better on tail and deep tail.'
  if i==19:text+='\n\nProtocol: 420 expert-training videos, 90 development videos, 90 reserved gate videos (45 fit/45 selection). Selection uses training-derived held-out videos. Final YOLO detector evaluation uses 36,717 official validation frames at IoU0.5. Development crop AP is not detector AP.'
  if i==20:text+='\nOriginal Stage4/5/6 triplet means:9.07/11.23/11.28. Baselines use original frozen encoders. These contextual architectures are different studies; descriptive differences do not isolate a single causal change.'
  if i==24:text+='\n\n'+notes(b['slides'][21])
 text=f'Rehearsal cue: {minutes[i-1]} seconds.\n'+text
 oid=s[id]['slideProperties']['notesPage']['notesProperties']['speakerNotesObjectId']
 req.extend([{'deleteText':{'objectId':oid,'textRange':{'type':'ALL'}}},{'insertText':{'objectId':oid,'text':text}}]);rows.append({'number':i,'slide':id,'seconds':minutes[i-1],'notes':text})
(W/'finalize-requests.json').write_text(json.dumps(req));(W/'speaker-notes.json').write_text(json.dumps(rows,indent=2))
print('ready',len(req))

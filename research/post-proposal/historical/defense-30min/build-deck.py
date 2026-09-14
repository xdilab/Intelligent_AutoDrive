import json,copy,statistics,csv
from pathlib import Path
W=Path('artifacts/defense-30min'); P=json.loads((W/'raw-template.json').read_text())['structuredContent']; slides=P['slides']; R=[]; deleted=set(); uid=0
ORDER=[1,8,7,9,10,6,16,15,20,24,26,28,33,37,36,65,66,51,54,49,40,67,29,41,5,39,34,44,45,46]
TITLES=['Long-tail compositional road-event detection','One box, one road event','ROAD-Waymo defines the task','The triplet distribution is long-tailed','Hypothesis: separate detection and composition','Prior work establishes the task','One shared evaluation protocol','Stage 0: joint detection baseline','Stage 1: score transfer has limited coverage','Stage 2: RoIAlign reads each candidate','Stage 3: learn composition from visual evidence','Learned composition improves the baseline','Pretraining connects video and language','One crop sequence yields one feature vector','Stage 4: phrase-based scoring','Stage 5: crop features and composition','Stage 6: add phrase evidence','From similarities to ranked detections','Where the 184 output scores come from','Frame-mAP measures detection ranking','Controls separate useful evidence from attribution','Completed results across three seeds','Phrase scoring improves tail recognition','Fusion gains vary by class','Headline model per paper','Additional papers with partial task coverage','Driving-language adaptation is being tested','What the evidence supports','Selected references and evidence','Questions']
NOTES=[
'Introduce the task: recognize rare road events from driving video. The contribution is a controlled study of visual features, learned composition, and phrase evidence. Preview the question: when does language help the tail?',
'Point to the stopped car, then its agent, action, and location. Duplex joins agent and action; triplet adds location. One box can carry multiple valid labels. We recognize observed events, not future intent.',
'Use the verified vocabulary: 10 agents, 22 actions, 16 locations, 49 duplexes, 86 triplets. The study uses 600 training videos and 198 validation videos; our evaluated frame manifest is a subset described shortly.',
'Point to the common head and rare tail. Tail membership comes only from training counts: 47 triplets below the geometric mean; 28 form the deeper subset. A common agent or action can still belong to a rare combination.',
'State the hypothesis, then the test: separate box detection from event scoring, learn combinations, and evaluate whether phrase evidence helps rare events. The architecture still outputs the benchmark compositions; it does not eliminate class imbalance.',
'ROAD establishes compositional event detection. ROAD-R adds logical constraints. ROAD-Waymo expands the setting. Our question concerns visual and semantic evidence for rare compositions. Published numerical comparisons come after our controlled results.',
'Distinguish training from evaluation. Training uses GT positives plus random and detector-junk negatives. Evaluation uses predicted YOLO boxes, except Stage 0. All headline runs share 36,717 frames and the same frame-AP implementation.',
'Point left to right: the frozen, previously trained 3D-RetinaNet supplies both boxes and event scores. This is the baseline evaluated on our shared manifest. No encoder is trained during this evaluation.',
'YOLO supplies new boxes and agent identity. Stage 1 copies event scores only from overlapping baseline detections. Explain that unmatched boxes cannot gain event evidence; this motivates direct feature extraction.',
'Trace one candidate box onto the P3 feature map. RoIAlign samples fractional locations by interpolation, then pools a 256-value feature. It avoids rounding the box to coarse cells. The trained linear head predicts scores from that feature.',
'The composition MLP combines primitive predictions with visual features. Action and location outputs stay fixed; duplex and triplet scores are replaced. It learns a mapping from evidence to combinations, not a hand-coded product rule.',
'Compare Stage 0 with Stage 3, then Stage 2 with Stage 3. Triplet AP rises from 7.54 to 9.73; the composition addition improves over 6.65 with the same boxes. These are single-run stage results.',
'CLIP-style pretraining aligns video and text in a shared space. Distinguish the original global contrastive embedding from our contextual keyframe feature tap. Downstream ROAD heads use frozen encoders in the completed study.',
'One keyframe box defines a fixed padded region across eight frames. It is not tracking. The encoder contextualizes the sequence; pooled keyframe tokens yield one 1024-value crop feature. Future frames mean this is not causal anticipation.',
'The phrase matrix contains 184 fixed text embeddings. A learned projection aligns each crop feature with that matrix. Scale and class biases convert similarities into logits. Focal classification loss trains the head; this stage has no contrastive training.',
'Follow the two branches. The flat head supplies primitive evidence; the MLP combines those raw sigmoids with the crop feature to predict compositions. Keep the flat and composition scores separate until final assembly. Training proceeds in separate steps.',
'Keep Stage 5 and add phrase composition evidence to the MLP input. The added 135 values are raw phrase sigmoids, before detector confidence weighting. This tests whether phrase evidence adds information beyond the same crop feature.',
'The cosine is a learned logit ingredient, not a probability. Sigmoid produces a class score; multiplying by detector confidence produces the ranking score. Walk through the illustrative two-box example. Ranking is used for AP, not a separate learned ranking loss.',
'Count the blocks: one agentness, ten agents, twenty-two actions, sixteen locations, forty-nine duplexes, eighty-six triplets. That is 184 channels but 183 semantic classes. YOLO supplies agent outputs; the selected heads supply the other groups.',
'Explain AP through one ranked class, then average across classes. A correct label still needs a matched box at IoU 0.5. Frame-mAP evaluates boxes; video-mAP evaluates tubes. Our headline results are frame-mAP, not video-mAP.',
'Flat, phrase, shuffled-phrase, and fusion controls share features and evaluation. Two video folds generate out-of-fold training predictions for the MLP; these are not validation folds. Three seeds measure head-training variability, not new-dataset uncertainty.',
'Lead with the averages: Stage 5 is 11.23, Stage 6 is 11.28 triplet AP. Their tail means are also almost tied. Phrase-only scoring improves the tail over flat scoring. Do not claim a reliable Stage 6 improvement.',
'Compare identical crop features: phrase scoring improves tail and deeper-tail AP in all three seeds, but loses common-class performance. The trade-off matters. Shared phrase geometry may help rare classes; attribution controls constrain that explanation.',
'Point to gains and regressions, not just successes. These are Stage 6 minus Stage 5 class-AP differences averaged over seeds. A large gain on a rare class can coexist with a large loss elsewhere. Diagnose localization and visual ambiguity separately.',
'One headline model per paper: the ROAD-Waymo baseline, FRCB, and our Stage 5 composition model. Read the event column, then the primitive groups. FRCB and our model are numerically close, but frames, candidates, and run summaries differ. Do not claim superiority.',
'These additional papers report partial task coverage: HEG classifies road activity, Track 1 detects agent tubes, and Track 3 recognizes TACO atomic activities. Their event frame-AP is not reported. Include their published results without equating these metrics with ours.',
'The implemented pilot adapts selected blocks of both towers on BDD-X with contrastive loss, then freezes the towers for ROAD transfer. Compare original and adapted encoders across Stages 5 and 6. ROAD AP is still pending; larger encoders are a future ablation.',
'Conclude with what is established: learned composition helps the baseline, and phrase scoring improves rare-class recognition with a common-class trade-off. Fusion has no reliable average gain yet. Finish the adaptation comparison before testing encoder size and box-source ablations.',
'Point out that slide footers name the underlying sources. The working-deck companion contains the full comparison tables, protocol caveats, and local experiment evidence. These are available for questions; do not read the bibliography aloud.',
'Invite questions. Navigate back to the relevant figure or result when answering. Use the pointer to identify the exact component or class.']
TIMES=[35,65,50,60,60,50,65,45,45,65,65,50,50,65,65,85,80,70,60,80,55,75,65,65,80,70,80,45,20,10]
# Allocate exactly 30 minutes, preserving relative emphasis and 5-second checkpoints.
while sum(TIMES)>1800:
 for i in [24,25,26,15,16,19,21,23]:
  if sum(TIMES)<=1800:break
  TIMES[i]-=5
while sum(TIMES)<1800: TIMES[19]+=5
assert len(ORDER)==30 and sum(TIMES)==1800
E={e['objectId']:e for s in slides for e in s.get('pageElements',[])}
def tx(e):return ''.join(x.get('textRun',{}).get('content','') for x in e.get('shape',{}).get('text',{}).get('textElements',[]))
def add(k,**v):R.append({k:v})
def delete(id):
 if id not in deleted:add('deleteObject',objectId=id);deleted.add(id)
def pos(e):
 t=e.get('transform',{});return t.get('translateX',0)/12700,t.get('translateY',0)/12700

def replace(id,text,size=None):
 e=E[id];old=tx(e);els=e.get('shape',{}).get('text',{}).get('textElements',[])
 runs=[x for x in els if 'textRun'in x]
 # Record source runs; reconstruct paragraph styles after flexible copy edits.
 first=copy.deepcopy(runs[0]['textRun'].get('style',{})) if runs else {}
 allowed=['fontFamily','fontSize','bold','italic','foregroundColor','weightedFontFamily','underline','baselineOffset']
 first={k:v for k,v in first.items() if k in allowed}
 if size:first['fontSize']={'magnitude':size,'unit':'PT'}
 if old:add('deleteText',objectId=id,textRange={'type':'ALL'})
 add('insertText',objectId=id,insertionIndex=0,text=text)
 if first:add('updateTextStyle',objectId=id,textRange={'type':'ALL'},style=first,fields=','.join(first))
 # Preserve source paragraph hierarchy where the paragraph counts agree.
 oldpars=old.rstrip('\n').split('\n');newpars=text.split('\n')
 if len(oldpars)==len(newpars):
  at=0;oldat=0
  for o,n in zip(oldpars,newpars):
   candidates=[r for r in runs if r.get('startIndex',0)<=oldat<r.get('endIndex',0)]
   if candidates:
    st={k:v for k,v in candidates[0]['textRun'].get('style',{}).items() if k in allowed}
    if size:st['fontSize']={'magnitude':size,'unit':'PT'}
    if st and n:add('updateTextStyle',objectId=id,textRange={'type':'FIXED_RANGE','startIndex':at,'endIndex':at+len(n)},style=st,fields=','.join(st))
   at+=len(n)+1;oldat+=len(o)+1
 return id

def geom(id,x,y,w,h):
 e=E[id];sz=e['size'];unit=sz['width']['unit'];factor=12700 if unit=='EMU' else 1
 ow=sz['width']['magnitude']/factor;oh=sz['height']['magnitude']/factor
 add('updatePageElementTransform',objectId=id,applyMode='ABSOLUTE',transform={'scaleX':w/ow,'scaleY':h/oh,'translateX':x,'translateY':y,'unit':'PT'})

def rgb(h):h=h.lstrip('#');return {'red':int(h[0:2],16)/255,'green':int(h[2:4],16)/255,'blue':int(h[4:6],16)/255}
def box(sl,text,x,y,w,h,size=18,color='004684',fill=None,shape='TEXT_BOX',bold=False,stroke=None,dash=False):
 global uid;uid+=1;id=f'd30_{uid:04d}'
 add('createShape',objectId=id,shapeType=shape,elementProperties={'pageObjectId':sl,'size':{'width':{'magnitude':w,'unit':'PT'},'height':{'magnitude':h,'unit':'PT'}},'transform':{'scaleX':1,'scaleY':1,'translateX':x,'translateY':y,'unit':'PT'}})
 pr={'shapeBackgroundFill':{'solidFill':{'color':{'rgbColor':rgb(fill)},'alpha':1}} if fill else {'propertyState':'NOT_RENDERED'},'outline':{'propertyState':'NOT_RENDERED'},'contentAlignment':'MIDDLE'}
 if stroke:pr['outline']={'outlineFill':{'solidFill':{'color':{'rgbColor':rgb(stroke)}}},'weight':{'magnitude':2.4 if bold else 1,'unit':'PT'},'dashStyle':'DASH' if dash else 'SOLID'}
 add('updateShapeProperties',objectId=id,shapeProperties=pr,fields=','.join(pr))
 if text:
  add('insertText',objectId=id,insertionIndex=0,text=text)
  add('updateTextStyle',objectId=id,textRange={'type':'ALL'},style={'fontFamily':'Calibri','fontSize':{'magnitude':size,'unit':'PT'},'foregroundColor':{'opaqueColor':{'rgbColor':rgb(color)}},'bold':bold},fields='fontFamily,fontSize,foregroundColor,bold')
  add('updateParagraphStyle',objectId=id,textRange={'type':'ALL'},style={'spaceAbove':{'magnitude':0,'unit':'PT'},'spaceBelow':{'magnitude':0,'unit':'PT'},'lineSpacing':105},fields='spaceAbove,spaceBelow,lineSpacing')
 return id

def line(sl,x1,y1,x2,y2,arrow=True):
 global uid;uid+=1;id=f'd30_line_{uid:04d}'
 add('createLine',objectId=id,lineCategory='STRAIGHT',elementProperties={'pageObjectId':sl,'size':{'width':{'magnitude':max(abs(x2-x1),.1),'unit':'PT'},'height':{'magnitude':max(abs(y2-y1),.1),'unit':'PT'}},'transform':{'scaleX':1 if x2>=x1 else -1,'scaleY':1 if y2>=y1 else -1,'translateX':x1,'translateY':y1,'unit':'PT'}})
 add('updateLineProperties',objectId=id,lineProperties={'lineFill':{'solidFill':{'color':{'rgbColor':rgb('333333')}}},'weight':{'magnitude':1.5,'unit':'PT'},'endArrow':'FILL_ARROW' if arrow else 'NONE'},fields='lineFill,weight,endArrow')

def bodyclear(n):
 s=slides[n-1]
 for e in s['pageElements']:
  x,y=pos(e)
  if 165<=y<490:delete(e['objectId'])
 return s['objectId']

def title(n,text,sub=None,section=None):
 s=slides[n-1];ts=[e for e in s['pageElements'] if tx(e).strip()]
 for e in ts:
  x,y=pos(e)
  if 90<=y<140:replace(e['objectId'],text,28);geom(e['objectId'],75,99,820,42);break
 if sub is not None:
  ss=[e for e in ts if 140<=pos(e)[1]<163]
  if ss:replace(ss[0]['objectId'],sub,18);geom(ss[0]['objectId'],75,145,825,32)
  else:box(s['objectId'],sub,75,145,825,32,18)
 if section:
  for e in ts:
   if pos(e)[1]<20:replace(e['objectId'],section)

def simpletable(sl,rows,x=80,y=190,width=805,rowh=28,widths=None,fs=16):
 widths=widths or [width/len(rows[0])]*len(rows[0]);xx=x
 for i,row in enumerate(rows):
  xx=x
  for j,value in enumerate(row):
   bg='004684' if i==0 else ('FFF2CC' if str(row[0]).startswith('Our') else ('F1F6FB' if i%2 else 'FFFFFF'))
   box(sl,str(value),xx,y+i*rowh,widths[j],rowh,fs,'FFFFFF' if i==0 else '23445C',bg,bold=i==0)
   xx+=widths[j]

# Before mutations, map every source-local media object explicitly.
plan=[]; elapsed=0
for i,n in enumerate(ORDER):
 s=slides[n-1];media=[]
 for e in s['pageElements']:
  if 'image'in e or 'video'in e:media.append({'objectId':e['objectId'],'action':'replace' if n in [34,36,65,66] else 'delete' if n==5 else 'keep','reason':'updated native diagram' if n in [34,36,65,66] else 'source narrative media'})
 elapsed+=TIMES[i];plan.append({'slide':i+1,'sourceSlideNumber':n,'exemplarId':s['objectId'],'method':'edit copied exemplar in place','role':TITLES[i],'seconds':TIMES[i],'endTime':f'{elapsed//60:02}:{elapsed%60:02}','notes':NOTES[i],'media':media})
(W/'slide-plan.json').write_text(json.dumps(plan,indent=2));(W/'speaker-notes.md').write_text('# 30-minute speaking cues\n\n'+ '\n\n'.join(f"## {x['slide']}. {x['role']} ({x['seconds']}s; finish {x['endTime']})\n\n{x['notes']}" for x in plan))
# Remove empty source-local text placeholders and normalize section labels.
for n in ORDER:
 for e in slides[n-1]['pageElements']:
  if e.get('shape',{}).get('shapeType')=='TEXT_BOX' and not tx(e).strip():delete(e['objectId'])
  elif tx(e).strip() in ['Proposed Work','Appendix','Slide Subject']:replace(e['objectId'],'Methodology' if n not in [39,40,41,44] else 'Results & Next Steps')
# Cover: shorten and repair its crowded mixed-style title, retain branding/media.
replace('p3_i3','LONG-TAIL\nROAD-EVENT DETECTION\nLearned composition and language evidence',32);geom('p3_i3',55,56,830,200)
# Core framing.
replace('byrd_solution_el_0','RESEARCH HYPOTHESIS');replace('byrd_solution_el_3','Shared evidence may help rare compositions')
replace('byrd_solution_el_2','Separate localization from event scoring\nUse detector candidates with dedicated event heads.')
replace('byrd_solution_statement_panel_1','Learn combinations from shared evidence\nCompare crop features, primitive predictions, and phrase scores.')
replace('byrd_solution_statement_panel_2','Test gains and trade-offs\nMeasure tail, common classes, repeated seeds, and controls.')
replace('byrd_solution_el_5','Hypothesis: language evidence can help rare events; reliable fusion gains must be demonstrated.')
# Protocol cards retain native structures.
replace('story_16_8','Training crops');replace('story_16_9','GT-positive boxes\nRandom + YOLO-junk negatives',20)
replace('story_16_11','Evaluation candidates');replace('story_16_12','36,717 shared frames\nYOLO boxes for Stages 1–6',20)
replace('story_16_13','Frame-mAP at IoU 0.5; Stage 0 retains its own detector boxes',19)
replace('story_16_14','GT supervises training; predicted boxes are evaluated',19)
# Concise stage titles fix original wrapping.
title(24,'STAGE 2: RoIAlign FEATURES','Sample each candidate region without rounding its coordinates','Methodology')
title(26,'STAGE 3: LEARNED COMPOSITION','Primitive scores + RoI features → duplex and triplet scores','Methodology')
title(33,'PRETRAINING: VIDEO AND LANGUAGE','The global contrastive embedding differs from our crop-feature tap','Methodology')
title(37,'ONE CROP SEQUENCE, ONE VECTOR','Eight fixed-region crops → one contextual 1024-value feature','Methodology')
# Existing equations remain editable; update metadata.
title(49,'FRAME-mAP: RANK, MATCH, AVERAGE','Box AP at IoU 0.5; video-mAP instead evaluates linked tubes','Evaluation')
# Stage 4/5/6: replace obsolete diagram media with readable native modules.
for n,st in [(36,4),(65,5),(66,6)]:
 sl=bodyclear(n);title(n,f'STAGE {st}: '+{4:'PHRASE SCORING',5:'LEARNED COMPOSITION',6:'PHRASE-EVIDENCE FUSION'}[st],{4:'Frozen encoders; a supervised phrase-scoring head',5:'Flat scores and composition scores join at assembly',6:'Add phrase composition evidence to the Stage 5 input'}[st],'Methodology')
 for e in slides[n-1]['pageElements']:
  if 'image'in e:delete(e['objectId'])
 box(sl,'YOLOv8x\nFrozen detector',80,198,140,55,17,fill='DAE8FC',shape='TRAPEZOID',stroke='6C8EBF',dash=True)
 box(sl,'Candidate boxes + confidence q + agent class',258,198,390,48,17,fill='F5F5F5',shape='ROUND_RECTANGLE',stroke='909090')
 line(sl,220,225,258,225);line(sl,648,225,735,225)
 box(sl,'Score\nassembly',735,198,135,65,18,fill='F8CECC',shape='ROUND_RECTANGLE',stroke='B85450')
 box(sl,'InternVideo2\nFrozen video tower',80,310,145,68,17,fill='D5E8E4',shape='TRAPEZOID',stroke='52998F',dash=True)
 line(sl,300,246,173,310)
 box(sl,'Crop feature\nOne 1024-vector',260,317,145,54,17,fill='DAE8FC',shape='ROUND_RECTANGLE',stroke='6C8EBF');line(sl,225,345,260,345)
 if st==4:
  box(sl,'Phrase head\nProjection + cosine',450,302,185,68,17,fill='E1D5E7',shape='ROUND_RECTANGLE',stroke='9673A6',bold=True);line(sl,405,345,450,345)
  box(sl,'Frozen phrase matrix\n184 class descriptions',425,402,230,53,17,fill='E1D5E7',shape='ROUND_RECTANGLE',stroke='9673A6',dash=True);line(sl,535,402,535,370)
  box(sl,'Phrase scores',678,312,190,44,17,fill='E1D5E7',shape='ROUND_RECTANGLE',stroke='9673A6');line(sl,635,335,678,335);line(sl,802,312,802,263)
 else:
  box(sl,'Flat head\n184 logits',450,293,150,58,17,fill='DAE8FC',shape='ROUND_RECTANGLE',stroke='6C8EBF',bold=True);line(sl,405,337,450,322)
  box(sl,'Flat scores',655,292,135,38,17,fill='DAE8FC',shape='ROUND_RECTANGLE',stroke='6C8EBF');line(sl,600,322,655,312);line(sl,775,292,775,263)
  box(sl,'Composition MLP\n49 primitives + crop'+('\n+ 135 phrase scores' if st==6 else ''),470,388,230,72,17,fill='E1D5E7',shape='ROUND_RECTANGLE',stroke='9673A6',bold=True)
  line(sl,520,351,520,388);line(sl,365,371,470,415)
  box(sl,'Composition\nscores',740,390,130,61,17,fill='E1D5E7',shape='ROUND_RECTANGLE',stroke='9673A6');line(sl,700,424,740,424);line(sl,850,390,850,263)
  if st==6:line(sl,300,371,227,411);box(sl,'Phrase head evidence\nFixed text; learned projection',80,411,295,51,16,fill='E1D5E7',shape='ROUND_RECTANGLE',stroke='9673A6');line(sl,375,436,470,436)
 box(sl,'Dashed border: frozen weights    Thick border: trained in a separate step',80,469,800,25,14,'555555')
# Score example, native three-row equation family.
sl=bodyclear(51);title(51,'FROM SIMILARITY TO RANKING','Cosine similarity becomes a logit; detector confidence weights the score','Methodology')
for y,h,b in [(195,'1  Compare','z = normalize(Wx + a);   ℓ = exp(η) Pz + b'),(265,'2  Weight','s = q × sigmoid(ℓ)   for each event class'),(335,'3  Rank','Box A: 0.90 × 0.80 = 0.72;   Box B: 0.60 × 0.95 = 0.57')]:
 box(sl,h,80,y,170,52,19,bold=True);box(sl,b,260,y,625,52,20);line(sl,80,y+60,880,y+60,False)
box(sl,'Illustrative scores, not measured predictions. Higher s ranks first; AP also checks box overlap.',80,422,800,48,17,'555555')
# Replace assembly rows with compact group mapping including new stages.
sl=bodyclear(54);title(54,'183 CLASSES + AGENTNESS = 184','One box carries multiple scores; these are not 184 triplet classes','Methodology')
simpletable(sl,[['Output group','Channels','Stage 4','Stages 5 / 6'],['Agentness + agent','1 + 10','YOLO','YOLO'],['Action + location','22 + 16','Phrase head','Flat head'],['Duplex + triplet','49 + 86','Phrase head','Composition MLP']],y=210,rowh=55,widths=[225,135,200,245],fs=19)
box(sl,'Event scores use q × sigmoid(logit). No additional agent-class mask on compositions.',80,448,810,40,17,'555555')
# Controls retain three cards.
title(40,'CONTROLLED COMPARISONS','Same features, frames, targets, and evaluation; three training seeds','Results')
for id,v in {'story_38_42':'Head choice','story_38_43':'Flat vs phrase\nSame crop features','story_38_45':'Attribution','story_38_46':'Shuffled phrases\nFlat-evidence fusion','story_38_48':'Composition','story_38_49':'Two video folds\nOut-of-fold inputs','story_38_50':'25 completed evaluations; variability is across head-training seeds.'}.items():replace(id,v)
# Original native metrics table modified in place, preserving object type.
sl=slides[66]['objectId'];title(67,'COMPLETED THREE-SEED RESULTS',None,'Results')
tableid='g3fa051c7877_0_38';tab=E[tableid]['table'];oldrows=tab['rows'];oldcols=tab['columns']
# Reuse 8x9 table as a 6x5 table by deleting surplus rows/columns, then refill.
rows=[['Model','Triplet','Tail 47','Deep 28','Common 39'],['Flat head','8.42 ± 0.27','3.54 ± 0.50','2.78 ± 0.46','14.29 ± 0.12'],['Phrase head','9.07 ± 0.15','5.77 ± 0.13','4.66 ± 0.11','13.03 ± 0.19'],['Our Stage 5','11.23 ± 0.58','6.16 ± 0.79','4.77 ± 0.81','17.33 ± 0.35'],['Our Stage 6','11.28 ± 0.25','6.17 ± 0.28','4.57 ± 0.61','17.43 ± 0.27']]
# Keep table topology supported by native API, clear before reshape.
for rr in range(oldrows):
 for cc in range(oldcols):add('deleteText',objectId=tableid,cellLocation={'rowIndex':rr,'columnIndex':cc},textRange={'type':'ALL'})
for rr in range(oldrows-1,len(rows)-1,-1):add('deleteTableRow',tableObjectId=tableid,cellLocation={'rowIndex':rr,'columnIndex':0})
for cc in range(oldcols-1,4,-1):add('deleteTableColumn',tableObjectId=tableid,cellLocation={'rowIndex':0,'columnIndex':cc})
add('updateTableColumnProperties',objectId=tableid,columnIndices=list(range(5)),tableColumnProperties={'columnWidth':{'magnitude':160,'unit':'PT'}},fields='columnWidth')
add('updateTableRowProperties',objectId=tableid,rowIndices=list(range(5)),tableRowProperties={'minRowHeight':{'magnitude':48,'unit':'PT'}},fields='minRowHeight')
for rr,row in enumerate(rows):
 for cc,v in enumerate(row):
  loc={'rowIndex':rr,'columnIndex':cc};add('insertText',objectId=tableid,cellLocation=loc,insertionIndex=0,text=v)
  add('updateTextStyle',objectId=tableid,cellLocation=loc,textRange={'type':'ALL'},style={'fontFamily':'Calibri','fontSize':{'magnitude':17,'unit':'PT'},'bold':rr==0,'foregroundColor':{'opaqueColor':{'rgbColor':rgb('FFFFFF' if rr==0 else '23445C')}}},fields='fontFamily,fontSize,bold,foregroundColor')
  add('updateTableCellProperties',objectId=tableid,tableRange={'location':loc,'rowSpan':1,'columnSpan':1},tableCellProperties={'tableCellBackgroundFill':{'solidFill':{'color':{'rgbColor':rgb('004684' if rr==0 else 'FFF2CC' if rr>=3 else 'F1F6FB')}}}},fields='tableCellBackgroundFill')
add('updatePageElementTransform',objectId=tableid,applyMode='ABSOLUTE',transform={'scaleX':1,'scaleY':1,'translateX':80,'translateY':195,'unit':'PT'})
box(sl,'ROAD-Waymo frame-mAP@0.5 (%); mean ± sample SD over 3 seeds. Stage 5 and 6 are nearly tied.',80,457,805,38,17,'555555')
# Tail result layout retained, updated with completed crop-head study.
title(29,'PHRASE SCORING HELPS THE TAIL','Identical crop features: flat head → phrase head','Results')
replace('byrd_eight_results_panel_2','Common classes | 39 triplets\n14.29 → 13.03 f-mAP     −1.26 percentage points',20)
replace('byrd_eight_results_el_2','Tail classes | 47 triplets\n3.54 → 5.77 f-mAP     +2.23 percentage points',20)
replace('byrd_eight_results_panel_1','Deeper tail | 28 triplets\n2.78 → 4.66 f-mAP     +1.88 percentage points',20)
replace('byrd_eight_results_el_5','Three-seed means; phrase improves tail in every seed, with a common-class trade-off.',15)
# Class-level examples retain 3 native cards.
title(41,'FUSION HELPS SOME CLASSES, HURTS OTHERS','Stage 6 minus Stage 5: per-class AP differences, averaged over three seeds','Results')
for id,v in {'story_39_52':'Bus-Stop-VehLane','story_39_53':'+21.92 points\nLarge class-level gain','story_39_55':'Bus-MovAway-\nOutgoLane','story_39_56':'+3.65 points\nUseful tail example','story_39_58':'LarVeh-Stop-Jun','story_39_59':'−15.18 points\nLarge regression','story_39_60':'Descriptive differences; no statistical-significance or causal explanation claimed.'}.items():replace(id,v,17 if id in ['story_39_52','story_39_55','story_39_58'] else None)
# One headline model per paper, as requested by Brandon.
means={}
for stage in [5,6]:
 vals=[json.load(open(f'artifacts/post-proposal-experiments/cluster-results/metrics/seed{i}-stage{stage}.json'))['summary'] for i in range(3)]
 means[stage]={k:statistics.mean(d[k] for d in vals) for k in vals[0]}
sl=bodyclear(5);title(5,'PUBLISHED EVENT-DETECTION COMPARISON','One headline model per paper; ROAD-Waymo validation frame-mAP@0.5 (%)','Published Results')
rows=[['Paper / headline model','Agent','Action','Loc.','Duplex','Event'],['Khan et al. / SlowFast-32','16.0','13.0','11.9','10.7','6.8'],['Zhong et al. / FRCB','33.64','23.87','32.33','17.84','11.71'],['Our Stage 5 (3-seed mean)']+[f'{means[5][k]:.2f}' for k in ['agent','action','loc','duplex','triplet']]]
simpletable(sl,rows,y=212,rowh=52,widths=[290,103,103,103,103,103],fs=18)
box(sl,'Published results use each paper’s protocol. Our exact frames and candidates differ.\nFRCB 5-run event mean: 11.53 ± 0.35; our Stage 5: 11.23 ± 0.58.',80,431,810,57,17,'555555')
sl=bodyclear(39);title(39,'ADDITIONAL PAPERS: PARTIAL TASK COVERAGE','One headline method per paper; retain the metric each paper actually reports','Published Results')
simpletable(sl,[['Paper / headline model','Dataset / task','Reported result'],['HEG / R(2+1)D + graph','ROAD-Waymo classification','23.21% mAP'],['ROAD++ Track 1 winner','ROAD-Waymo agent tubes','30.82% average v-mAP'],['ROAD++ Track 3 winner','TACO atomic activities','69.0% mAP']],y=212,rowh=55,widths=[320,290,195],fs=18)
box(sl,'Event frame-mAP: not reported by these papers. Our model has no matched result for these tasks.\nSources: HEG Table 1; Track 1 Table 1; Track 3 Table 1. Full citations in the companion.',80,442,810,48,16,'555555')
# Adaptation update: three phases in the existing InternVideo template frame.
sl=bodyclear(34);title(34,'TESTING DRIVING-LANGUAGE ADAPTATION','Implemented pilot; downstream ROAD-Waymo accuracy is still pending','Next Steps')
for e in slides[33]['pageElements']:
 if 'image'in e:delete(e['objectId'])
for x,h,b in [(80,'1  Start pretrained','InternVideo2-CLIP-S\nVideo + text towers'),(350,'2  Adapt on BDD-X','Partial two-tower updates\nVideo ↔ text contrastive loss'),(620,'3  Transfer to ROAD','Freeze adapted towers\nRetrain Stage 5 / 6 heads')]:
 box(sl,h,x,220,250,43,20,bold=True);box(sl,b,x,277,245,93,19,fill='F1F6FB',shape='ROUND_RECTANGLE')
line(sl,325,315,350,315);line(sl,595,315,620,315)
box(sl,'Selected epoch 2: development loss 4.24 → 3.56; retrieval gains mixed.\nTest: (adapted Stage 6 − 5) − (original Stage 6 − 5), with three head-training seeds.',80,392,810,70,19)
# Conclusion preserve two-section layout.
title(44,'CONCLUSIONS & NEXT TESTS',None,'Conclusion')
replace('g3fa12704523_50_1179','Learned composition improves the baseline under a shared evaluation protocol.\nPhrase scoring improves tail recognition, with a common-class trade-off.',21)
replace('g3fa12704523_50_1188','Still to establish')
replace('g3fa12704523_50_1189','A reliable average gain from phrase-evidence fusion; adapted ROAD results.\nThen test encoder size and GT versus YOLO-positive training crops.',21)
# Reference slide concise primary sources, sources in notes retain full companion.
title(45,'SELECTED SOURCES & EVIDENCE','Full tables and protocol notes accompany this working deck','References')
replace('g3fa12704523_50_1194','Singh et al. ROAD. IEEE TPAMI, 2023.\nKhan et al. ROAD-Waymo. arXiv:2411.01683, v3, 2026.\nZhong et al. FRCB. Pattern Recognition 180:114489, 2026.\nHe et al. Mask R-CNN (RoIAlign). ICCV, 2017.\nRadford et al. CLIP. ICML, 2021; Wang et al. InternVideo2. ECCV, 2024.\nLin et al. Focal Loss. ICCV, 2017; Wolpert. Stacked Generalization, 1992.\nLocal evidence: 25 shared-frame evaluations; seeds 0–2; full per-class AP.',19)
replace('g3fa12704523_50_1203','QUESTIONS')
# Short notes and reset skipped flags; source slide master remains untouched.
for item in plan:
 n=item['sourceSlideNumber'];s=slides[n-1];note=s['slideProperties']['notesPage']['notesProperties']['speakerNotesObjectId']
 add('deleteText',objectId=note,textRange={'type':'ALL'});add('insertText',objectId=note,insertionIndex=0,text=f"{item['seconds']} seconds | Finish by {item['endTime']}\n\n{item['notes']}")
 add('updateSlideProperties',objectId=s['objectId'],slideProperties={'isSkipped':False},fields='isSkipped')
# Fit retained diagram media to intrinsic aspect ratio, within slide body.
for n in [15,20,24,26,33,37]:
 for e in slides[n-1]['pageElements']:
  if 'image' not in e or e['objectId'] in deleted:continue
  x,y=pos(e)
  if y<160:continue
  sz=e['size'];ratio=sz['width']['magnitude']/sz['height']['magnitude'];w=810;h=w/ratio
  if h>275:h=275;w=h*ratio
  geom(e['objectId'],80+(810-w)/2,188+(275-h)/2,w,h)
# Save requests before any remote content mutation.
(W/'content-requests.json').write_text(json.dumps(R))
(W/'finalize-requests.json').write_text(json.dumps([{'deleteObject':{'objectId':s['objectId']}} for i,s in enumerate(slides,1) if i not in ORDER]))
(W/'reorder-requests.json').write_text(json.dumps([{'updateSlidesPosition':{'slideObjectIds':[slides[n-1]['objectId'] for n in ORDER],'insertionIndex':0}}]))
(W/'comparison-means.json').write_text(json.dumps(means,indent=2))
print('Prepared',len(R),'requests; 30 slides; notes',sum(len(x.split()) for x in NOTES),'words; timing',sum(TIMES))

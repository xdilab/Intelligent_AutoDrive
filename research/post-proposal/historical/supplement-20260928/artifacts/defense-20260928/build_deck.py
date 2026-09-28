import json, pathlib, statistics, hashlib
from PIL import Image
W=pathlib.Path('/data/repos/wiki/artifacts/defense-20260928'); ROOT=pathlib.Path('/data/repos/wiki')
raw=json.load(open(W/'raw-template.json'))['structuredContent']; src={s['objectId']:s for s in raw['slides']}
models=json.load(open(ROOT/'artifacts/contextual-evolution-20260924/selection.json'))['models']
slides=[]
def add(title,body='',notes='',image=None,table=None,caption='',ex='p4',section='Methodology'):
 slides.append(dict(number=len(slides)+1,title=title,body=body,notes=notes,image=image,table=table,caption=caption,exemplar=ex,section=section))
def mr(k,label):
 m=models[str(k)]['mean'];return [label]+[f'{m[x]:.2f}' for x in ['triplet','tail47','deep28']]
def orig(stage,label):
 paths=[ROOT/f'artifacts/post-proposal-experiments/cluster-results/metrics/seed{i}-{stage}.json' for i in range(3)] if stage not in ['stage0','stage3'] else [ROOT/f'artifacts/post-proposal-experiments/cluster-results/metrics/{stage}.json']
 ds=[json.load(open(p)) for p in paths];return [label,f"{statistics.mean(d['summary']['triplet'] for d in ds):.2f}"]+[f"{statistics.mean(d['tail'][k]['mAP'] for d in ds):.2f}" for k in ['tail47','deep28']]
H=['Model','Triplet','Tail (47)','Deep tail (28)']
add('Long-Tail Compositional\nRoad-Event Detection','', 'Opening, 45 seconds. The goal is to recognize who is doing what and where, including rare combinations. Working defense title; formal thesis title can be substituted.',ex='p1',section='Title')
add('Outline','Motivation\nRelated Works & Limitations\nMethodology\nResults\nConclusion','30 seconds. Five sections, with architecture evolution followed by matched evidence.',ex='p3',section='Outline')
add('From objects to road events','Agent: car\nAction: stopped\nLocation: vehicle lane','60 seconds. This is an observed road event, not a prediction of mental intent. The gold annotation is the same Car–Stop–VehLane example used in the proposal. Source: artifacts/proposal-road-example/provenance.json.',image='artifacts/proposal-road-example/road-waymo-train00407-frame00001-annotated.png',section='Motivation')
add('Rare compositions are the challenge','', '60 seconds. Tail partitions use training counts only. z is the standardized log10 triplet frequency; tail47 z<0, deep28 z<−0.5, common39 z≥0. The deep subset is nested within the tail. Source: artifacts/proposal-tail/train-counts.json.',image='artifacts/proposal-tail/triplet-tail-counts-wide.png',caption='86 triplets  •  47 tail classes  •  28 deep-tail classes',section='Motivation')
add('Related works','ROAD / SlowFast: joint road-event detection\nHEG: higher-order agent–action relations\nFRCB: context and imbalance-aware learning','90 seconds. ROAD-Waymo SlowFast32 supplies a joint detector baseline. HEG reports a different agent–action task and is not mixed into the triplet detector table. FRCB motivates positional context and difficulty-calibrated BCE. Sources: https://arxiv.org/html/2411.01683v3 ; https://arxiv.org/html/2409.11206 ; https://doi.org/10.1016/j.patcog.2026.114489 . See artifacts/defense-30min/comparison-sources.md and artifacts/frcb-2026/tables.md.',section='Related Works & Limitations')
add('Research question','Can context improve rare-event recognition?\nWhen does phrase information help?\nCan complementary experts preserve both?','60 seconds. Isolated crops may omit road layout and interactions. Language may share information across labels, but content and prototype geometry need controls. We test these hypotheses, rather than assume every architectural addition helps.',section='Related Works & Limitations')
add('Data and evaluation','Train heads on ground-truth-aligned features\nSelect models and blends on held-out training videos\nEvaluate YOLO detections on 36,717 validation frames','90 seconds. ROAD-Waymo official validation is our final evaluation set; development crop AP is only for selection. Contextual split: 420 expert-training videos, 90 development videos, 90 reserved gate videos (45 gate-fit and 45 selection). Global scalar blend uses the 45 selection videos; neural gates additionally use gate-fit. Ground-truth positive supervision and background negatives must not be confused with detector evaluation. IoU=0.5. Three seeds for contextual comparisons. Source: artifacts/contextual-evolution-20260924/technical-evidence.md; contextual-gate protocol files.')
add('RoIAlign: boxes to features','', '90 seconds. YOLO proposes candidate boxes. RoIAlign maps each box to the feature map and uses bilinear interpolation instead of quantizing coordinates. The historical Stage 2 example uses ResNet50-I3D/FPN features; later contextual experiments use InternVideo2 spatial features with the same alignment principle. A crop re-encodes pixels; RoIAlign extracts features from an encoded full frame. Source: artifacts/proposal-expansion-eight/roi-align.drawio.',image='artifacts/proposal-expansion-eight/roi-align.png',caption='Candidate boxes select aligned regions from a spatial feature map.')
add('Stage 5: learned composition','', '90 seconds. Canonical name: Stage 5: Semantic Crop Classifier + Comp MLP. One crop sequence produces one feature vector. Flat primitive scores and crop features feed the composition MLP; no explicit phrase-score input. Stage 5 is a strong visual composition baseline. Source: artifacts/defense-30min/alignment/stage5.png; findings/post-proposal-controlled-study.md.',image='artifacts/defense-30min/alignment/stage5.png',caption='Semantic Crop Classifier + Comp MLP')
add('Stage 6: phrase fusion','', '90 seconds. Canonical name: Stage 6: Semantic Crop Classifier + Phrase Fusion + Comp MLP. Adds phrase composition scores to the evidence supplied to the composition MLP. Compare with Stage 5 without implying a reliable average improvement; their controlled means are close. Source: artifacts/defense-30min/alignment/stage6.png; findings/post-proposal-controlled-study.md.',image='artifacts/defense-30min/alignment/stage6.png',caption='Semantic Crop Classifier + Phrase Fusion + Comp MLP')
base='artifacts/contextual-evolution-20260924/simplified/'
add('Contextual MLP head','', '90 seconds. Inputs: crops, RoI features, and the entire frame. Box position is included after feature extraction. Frozen InternVideo2 and text encoders supply cached features. Mean pooling summarizes scene tokens; a residual text adapter and language fusion remain trainable. One 184-output classifier, no composition MLP in this standalone architecture. Source: contextual-roi/model.py and artifacts/contextual-evolution-20260924/technical-evidence.md.',image=base+'01-mlp.png')
add('Actor-query scene attention','', '75 seconds. Replace scene averaging with actor-conditioned attention. The actor representation selects relevant scene tokens. Other contextual inputs remain. This motivates rare-composition analysis; results do not improve every primitive metric. Source: contextual-roi/model.py; technical-evidence.md.',image=base+'02-attention.png')
add('All-label contrastive alignment','', '90 seconds. Illustrated MLP branch; also tested on the attention branch. Normalize the pre-language visual RoI and adapted phrase vectors. Multi-positive contrastive targets span all 184 labels, temperature 0.07, loss weight 0.001. Gradients reach visual and text adapters, not frozen encoders. This is not contrastive loss on output scores or on the downstream joint RoI. L=Lclassification+0.001 Lcontrastive. Supersedes a historical 86-only/frozen-target recipe; those two changes cannot be isolated by comparing old and corrected runs. Source: contextual-roi-all184/model.py.',image=base+'04-all184-contrastive.png')
add('Blend complementary experts','', '90 seconds. Return to the attention branch. Expert A: focal classification without contrastive loss. Expert B: focal classification plus all184 contrastive loss. Both use language. Freeze trained experts, preserve 49 primitive/agentness scores from A, blend 135 compositions: p=(1−w)pA+w pB. One global scalar per seed, chosen by development triplet crop AP. This is not a sample-dependent neural gate. Source: contextual-gate/prepare_blend.py, technical-evidence.md. Learned gates were also tested; best mean triplet 12.59 versus simple blend 12.58.',image=base+'05-expert-blend.png')
add('Difficulty-calibrated BCE','', '90 seconds. DCB retrains both attention experts with difficulty-calibrated BCE, retaining the blend topology. Per-entry detached multiplier: 0.5+y(1−sigmoid(z))+history. History tracks per-class mean positive residual. Expert B retains the 0.001 all184 contrastive auxiliary. FRCB-inspired implementation, not a reproduction of its full system. The figure’s balancing label refers to this difficulty-calibrated BCE. No new inference layer. Sources: contextual-roi-dcb/dcb.py and protocol.json.',image=base+'06-dcb-blend.png')
add('Scores, ranking, and frame-mAP','184 outputs: agentness + primitives + compositions\nRank each label’s detections by confidence\nMatch boxes at IoU ≥ 0.5; average class AP','90 seconds. Outputs: 1 agentness +10 agents +22 actions +16 locations +49 duplexes +86 triplets. Multi-label means simultaneous labels, not one softmax class. For contextual evaluation, classification probabilities are multiplied by YOLO confidence; agentness and agent metrics are YOLO-derived. AP integrates interpolated precision across recall changes; mAP averages AP over the evaluated class set. Tail and deep-tail averages are subsets of the 86 triplets. We report frame-mAP, not video/tube-mAP. Source: stage56-full/metric.py.',caption='AP(c) = Σₖ (Rₖ − Rₖ₋₁) P̂ₖ     •     mAP = (1 / |C|) Σc AP(c)')
add('Original architecture baselines',notes='90 seconds. Detector f-mAP (%) at IoU0.5. Stage 0 and 3 are single runs; Stage 4–6 are three-seed means. These are original frozen-encoder conditions. Stage 4 is Semantic Crop Classifier; Stage 3 is RoI Linear Scoring Head + Composition MLP. Stage 5 and 6 are nearly tied overall. Source: artifacts/post-proposal-experiments/cluster-results/metrics/*.json.',table=[H,orig('stage0','Stage 0 · RetinaNet'),orig('stage3','Stage 3 · RoI + Comp MLP'),orig('head-phrase','Stage 4 · Semantic Crop'),orig('stage5','Stage 5 · + Comp MLP'),orig('stage6','Stage 6 · + Phrase Fusion')],caption='Detector f-mAP (%)  •  Stage 0/3: single run; Stage 4–6: three-seed means',ex='p15',section='Results')
add('Context and contrastive learning',notes='90 seconds. Three-seed detector means. MLP and attention each have matched classification-only and corrected all184-contrastive conditions. Attention helps tail versus MLP without contrastive, while all184 contrastive has architecture-dependent tradeoffs. Do not interpret classification without contrastive as absence of language. Source: selection.json models88,89,92,93.',table=[H,mr(88,'MLP · no contrastive'),mr(89,'Attention · no contrastive'),mr(92,'MLP · + all184 contrastive'),mr(93,'Attention · + all184 contrastive')],caption='Attention and contrastive learning help different parts of the distribution.',ex='p15',section='Results')
add('Does phrase information help?',notes='90 seconds. Matched attention-head controls, three-seed means. Real-bank tail AP exceeds both controls in all three seeds. Real versus random tail gain 0.5344pp is an exploratory secondary result; its unadjusted paired-t interval is positive. Overall triplet equivalence is not established. Prototype geometry differs, so this does not isolate linguistic semantics. Source: directions/contextual-head-language-control.md completed v3 results; artifacts/contextual-roi-langctl-20260919/comparison.json.',table=[H,['Real phrase bank','12.02','6.19','4.48'],['Language bypass','11.88','5.76','4.27'],['Random prototypes','11.88','5.65','4.02']],caption='Exploratory tail benefit; semantic content and prototype geometry remain confounded.',ex='p15',section='Results')
add('Blending improves the overall result',notes='90 seconds. Three-seed means, same detector manifest. DCB blend improves triplet and common AP but focal blend retains better tail and deep-tail means. Both constituent experts contain language; this is complementarity evidence, not an isolated language test. The simple blend is chosen for clear ancestry; learned gate best triplet mean is 12.59. Source: selection.json models89,93,94,102,104.',table=[H,mr(89,'Focal · no contrastive'),mr(93,'Focal · + contrastive'),mr(94,'Focal · expert blend'),mr(102,'DCB · no contrastive'),mr(104,'DCB · expert blend')],caption='Best overall: DCB blend (13.43)  •  Better tail: focal blend (6.49)',ex='p15',section='Results')
m=models['104']['mean']
add('Published comparison',notes='90 seconds. Frame-mAP (%), IoU0.5. SlowFast32 from ROAD-Waymo benchmark; FRCB selected headline result from Table3 (11.71 triplet), separate five-run mean is 11.53±0.35. Our row is a three-seed mean on 36,717 locked frames. Different detector proposals and published protocols prevent a controlled SOTA-superiority claim. No published tail cells were fabricated. Sources: artifacts/frcb-2026/tables.md, artifacts/defense-30min/comparison-sources.md, selection.json.',table=[['Model','Agent','Action','Location','Duplex','Triplet'],['SlowFast32','16.00','13.00','11.90','10.70','6.80'],['FRCB','33.64','23.87','32.33','17.84','11.71'],['Our DCB blend']+[f'{m[k]:.2f}' for k in ['agent','action','loc','duplex','triplet']]],caption='Competitive published numbers; protocols and detector proposals differ.',ex='p15',section='Results')
add('ROAD encoder adaptation: interim',notes='60 seconds. These are seed0 detector evaluations, not three-seed means. Stage5 selected epoch3 is complete; Stage6 numbers are epoch1 only, and selected-checkpoint detector evaluation remains running as of September28. Stage5/6 adaptation updates the last visual block and heads. Its contrastive recipe differs from the contextual corrected184 loss; do not combine conclusions. Source: artifacts/stage56-full/status-20260928/remote-snapshot.json. Latest live check: inference10400 and10225 of36717 frames. Final Stage6 results pending.',table=[H,['Stage 5 · focal · epoch 3','10.61','4.75','2.28'],['Stage 5 · + contrastive · epoch 3','10.66','4.60','2.25'],['Stage 6 · focal · epoch 1','10.96','6.26','4.69'],['Stage 6 · + contrastive · epoch 1','10.89','6.08','4.39']],caption='Single seed  •  Stage 6 selected-checkpoint detector results are pending',ex='p15',section='Results')
add('Limitations','Three seeds on one benchmark\nOverall and rare-class gains can diverge\nLanguage content is not fully isolated','60 seconds. No universal contrastive benefit established. Real/random geometry is a confound. The original 86-only versus corrected184 comparison changes scope and text gradients. Published comparisons use different protocols. Full encoder adaptation is still pending and does not underpin the completed contextual result.',section='Conclusion')
add('Conclusion','Context strengthens compositional detection\nComplementary experts improve overall AP\nTail robustness remains the next target','60 seconds. DCB blend:13.43 triplet AP versus original Stage6 mean11.28, a descriptive +2.15pp within our detector evaluation. Focal blend retains higher tail than DCB blend. Phrase-bank controls support a limited, exploratory tail benefit. Next: more seeds, geometry-controlled phrase tests and targeted tail-preserving routing. Completed evidence and pending adaptation remain distinct.',ex='p18',section='Conclusion')
add('Thank you','Questions','Closing, 15 seconds. Return to the relevant architecture or results slide during Q&A; use the pointer to follow the data path.',ex='p18',section='Conclusion')
assert len(slides)==25
# Preserve source run snapshots; each new body deliberately consists of a single native bullet role.
for s in slides:
 s['build_method']='duplicate native exemplar; replace role-mapped text; retain master and footer'
 s['media_policy']='retain NCAT/lab branding; delete unrelated medical media; insert approved source image only' if s['image'] else 'retain native template branding; delete unrelated source content'
 if s['image']:
  p=ROOT/s['image'];s['image_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
(W/'slide-plan.json').write_text(json.dumps(slides,indent=2))
(W/'backup.json').write_text(json.dumps({'id':'1hC3JCpZBOOErHRcnDgq6doAOnE2DjdVfcEKH_7eiqZI','working':'1ma7UQGy-keAukf1w4L1EdSYgL4ehmwDBCsn62tccEL4'}))
clones=[];req=[];images=[];maps={}
def walk(es):
 for e in es:
  yield e
  yield from walk(e.get('elementGroup',{}).get('children',[]))
def put(oid,text,fs=None,bold=None,color=None):
 req.extend([{'deleteText':{'objectId':oid,'textRange':{'type':'ALL'}}},{'insertText':{'objectId':oid,'text':text,'insertionIndex':0}}])
 st={}
 if fs:st['fontSize']={'magnitude':fs,'unit':'PT'}
 if bold is not None:st['bold']=bold
 if color:st['foregroundColor']={'opaqueColor':{'rgbColor':color}}
 if st:req.append({'updateTextStyle':{'objectId':oid,'textRange':{'type':'ALL'},'style':st,'fields':','.join(st)}})
def pos(oid,x,y,w,h,source=None):
 # Shapes retain their base size; only transforms change.
 sz=source['size'] if source else {'width':{'magnitude':3000000,'unit':'EMU'},'height':{'magnitude':3000000,'unit':'EMU'}}
 def pt(v):return v['magnitude']/(12700 if v['unit']=='EMU' else 1)
 req.append({'updatePageElementTransform':{'objectId':oid,'applyMode':'ABSOLUTE','transform':{'scaleX':w/pt(sz['width']),'scaleY':h/pt(sz['height']),'translateX':x,'translateY':y,'unit':'PT'}}})
def plain(oid,fs=24):
 req.extend([{'deleteParagraphBullets':{'objectId':oid,'textRange':{'type':'ALL'}}},{'updateTextStyle':{'objectId':oid,'textRange':{'type':'ALL'},'style':{'fontFamily':'Arial','fontSize':{'magnitude':fs,'unit':'PT'},'bold':False,'italic':False,'foregroundColor':{'opaqueColor':{'rgbColor':{}}}},'fields':'fontFamily,fontSize,bold,italic,foregroundColor'}},{'updateParagraphStyle':{'objectId':oid,'textRange':{'type':'ALL'},'style':{'alignment':'START','indentStart':{'magnitude':0,'unit':'PT'},'indentEnd':{'magnitude':0,'unit':'PT'},'indentFirstLine':{'magnitude':0,'unit':'PT'},'spaceAbove':{'magnitude':0,'unit':'PT'},'spaceBelow':{'magnitude':18,'unit':'PT'},'lineSpacing':110},'fields':'alignment,indentStart,indentEnd,indentFirstLine,spaceAbove,spaceBelow,lineSpacing'}}])
def textbox(page,oid,text,x,y,w,h,fs=18):
 req.append({'createShape':{'objectId':oid,'shapeType':'TEXT_BOX','elementProperties':{'pageObjectId':page,'size':{'width':{'magnitude':w,'unit':'PT'},'height':{'magnitude':h,'unit':'PT'}},'transform':{'scaleX':1,'scaleY':1,'translateX':x,'translateY':y,'unit':'PT'}}}})
 req.append({'insertText':{'objectId':oid,'text':text}});plain(oid,fs)
for s in slides:
 n=s['number'];ex=s['exemplar'];page=f'def_{n:02d}';s['page_id']=page
 es=list(walk(src[ex]['pageElements']));mp={ex:page,**{e['objectId']:f'd{n:02d}_{e["objectId"]}' for e in es}};maps[n]=mp
 clones.append({'duplicateObject':{'objectId':ex,'objectIds':mp}})
 E={e['objectId']:e for e in es};delete=[]
 if ex=='p1':
  delete=['p1_i67','p1_i2','p1_i3','p1_i8','p1_i14','p1_i15']
  put(mp['p1_i64'],s['title'],36,True);put(mp['p1_i65'],'Brandon L. Byrd\nNorth Carolina A&T State University',22)
  pos(mp['p1_i65'],180,235,600,80,E['p1_i65'])
  put(mp['p1_i68'],'Advisor: Dr. Hamidreza Moradi\nMaster’s Thesis Defense · Fall 2026',20,False)
  pos(mp['p1_i68'],180,335,600,70,E['p1_i68'])
 else:
  title={'p3':'p3_i74','p4':'p4_i83','p15':'p15_i173','p18':'p18_i230'}[ex]
  put(mp[title],s['title'],34 if len(s['title'])>36 else 38)
  num={'p4':'p4_i85','p15':'p15_i174','p18':'p18_i231'}.get(ex)
  if num:put(mp[num],str(n),14)
  body={'p3':'p3_i75','p4':'p4_i84','p18':'p18_i233'}.get(ex)
  if ex=='p3':delete=['p3_i76','p3_i77']
  if ex=='p18':delete=['p18_i232']
  if body:
   if s['body']:
    put(mp[body],s['body']);plain(mp[body],26 if n!=16 else 24)
    pos(mp[body],90,130,780,330,E[body])
    if n!=25:req.append({'createParagraphBullets':{'objectId':mp[body],'textRange':{'type':'ALL'},'bulletPreset':'BULLET_DISC_CIRCLE_SQUARE'}})
    if n==3:pos(mp[body],75,150,305,250,E[body])
   else:delete.append(body)
  if ex=='p15':
   delete=['p15_i175','p15_i178','p15_i179','p15_i2','p15_i3','p15_i4','p15_i5']
   tab=mp['p15_i176']; data=s['table'];rows=len(data);cols=len(data[0])
   if cols>4:req.append({'insertTableColumns':{'tableObjectId':tab,'cellLocation':{'rowIndex':0,'columnIndex':3},'insertRight':True,'number':cols-4}})
   for row in range(5,rows-1,-1):req.append({'deleteTableRow':{'tableObjectId':tab,'cellLocation':{'rowIndex':row,'columnIndex':0}}})
   req.append({'updatePageElementTransform':{'objectId':tab,'applyMode':'ABSOLUTE','transform':{'scaleX':1,'scaleY':1,'translateX':80,'translateY':135,'unit':'PT'}}})
   widths=[330]+[(800-330)/(cols-1)]*(cols-1) if cols==4 else [240]+[112]*5
   for c,width in enumerate(widths):req.append({'updateTableColumnProperties':{'objectId':tab,'columnIndices':[c],'tableColumnProperties':{'columnWidth':{'magnitude':width,'unit':'PT'}},'fields':'columnWidth'}})
   req.append({'updateTableRowProperties':{'objectId':tab,'rowIndices':list(range(rows)),'tableRowProperties':{'minRowHeight':{'magnitude':42,'unit':'PT'}},'fields':'minRowHeight'}})
   for r,row in enumerate(data):
    for c,t in enumerate(row):
     cell={'rowIndex':r,'columnIndex':c}
     req.extend([{'deleteText':{'objectId':tab,'cellLocation':cell,'textRange':{'type':'ALL'}}},{'insertText':{'objectId':tab,'cellLocation':cell,'text':t}},{'updateTextStyle':{'objectId':tab,'cellLocation':cell,'textRange':{'type':'ALL'},'style':{'fontFamily':'Arial','fontSize':{'magnitude':17 if cols==4 else 16,'unit':'PT'},'bold':r==0,'foregroundColor':{'opaqueColor':{'rgbColor':{}}}},'fields':'fontFamily,fontSize,bold,foregroundColor'}},{'updateParagraphStyle':{'objectId':tab,'cellLocation':cell,'textRange':{'type':'ALL'},'style':{'alignment':'START' if c==0 else 'CENTER','spaceAbove':{'magnitude':0,'unit':'PT'},'spaceBelow':{'magnitude':0,'unit':'PT'}},'fields':'alignment,spaceAbove,spaceBelow'}}])
   put(mp['p15_i177'],s['caption']);plain(mp['p15_i177'],18);pos(mp['p15_i177'],80,415,800,65,E['p15_i177'])
  elif s['caption']:textbox(page,f'{page}_caption',s['caption'],80,446,800,43,18 if n!=16 else 20)
 if s['image']:
  path=str(ROOT/s['image']);iw,ih=Image.open(path).size
  x,y,w=40,100,880
  if n==3:x,y,w=400,110,500
  elif n==4:x,y,w=180,100,600
  elif n==8:x,y,w=55,150,850
  elif n in [9,10]:x,y,w=50,120,860
  h=w*ih/iw
  if n==4 and h>330:w=330*iw/ih;h=330;x=(960-w)/2
  images.append({'slide':n,'path':path,'request':{'createImage':{'objectId':f'{page}_image','url':path,'elementProperties':{'pageObjectId':page,'size':{'width':{'magnitude':w,'unit':'PT'},'height':{'magnitude':h,'unit':'PT'}},'transform':{'scaleX':1,'scaleY':1,'translateX':x,'translateY':y,'unit':'PT'}}}}})
 for d in delete:req.append({'deleteObject':{'objectId':mp[d]}})
(W/'clone-requests.json').write_text(json.dumps(clones));(W/'content-requests.json').write_text(json.dumps(req));(W/'image-requests.json').write_text(json.dumps(images));(W/'slide-plan.json').write_text(json.dumps(slides,indent=2))
(W/'source-style-snapshot.json').write_text(json.dumps({k:src[k] for k in ['p1','p3','p4','p15','p18']},indent=2))
print('Slides',len(slides),'Clones',len(clones),'Content',len(req),'Images',len(images))

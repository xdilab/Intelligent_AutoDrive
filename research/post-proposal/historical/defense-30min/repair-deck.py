from pathlib import Path
exec(Path('artifacts/defense-30min/build-deck.py').read_text().split('# Before mutations,')[0])
P=json.load(open(W/'raw-ordered-output.json'))['structuredContent'];slides=P['slides'];E={e['objectId']:e for s in slides for e in s.get('pageElements',[])};R=[];uid=1000
# Header repairs: retain one title and subtitle, remove stale model-symbol decorations.
for n,tid,sid,label,sub,trash in [
(13,'iv_original_el_0','iv_original_el_3','PRETRAINING: VIDEO AND LANGUAGE','Global contrastive embeddings differ from our crop-feature tap',['iv_original_el_5','iv_original_el_6','iv_original_el_10','iv_original_el_11','iv_original_el_12']),
(27,'g3fa051c7877_0_27','g3fa051c7877_0_30','DRIVING-LANGUAGE ADAPTATION','Implemented pilot; ROAD-Waymo accuracy is still pending',['iv_mirror_io','iv_mirror_role','iv_mirror_metric','model_symbol_internvideo_note'])]:
 replace(tid,label,28);geom(tid,75,99,820,42)
 replace(sid,sub,18);geom(sid,75,145,820,32)
 add('updateTextStyle',objectId=tid,textRange={'type':'ALL'},style={'italic':False,'bold':True},fields='italic,bold')
 add('updateTextStyle',objectId=sid,textRange={'type':'ALL'},style={'italic':True,'bold':False},fields='italic,bold')
 for id in trash:delete(id)
# Cover keeps image/logos and a clear white title with a smaller blue subtitle.
id='p3_i3';text='LONG-TAIL\nROAD-EVENT DETECTION\nLearned composition and language evidence';replace(id,text,42);geom(id,55,68,830,180)
add('updateTextStyle',objectId=id,textRange={'type':'ALL'},style={'foregroundColor':{'opaqueColor':{'rgbColor':rgb('FFFFFF')}},'bold':True,'italic':False},fields='foregroundColor,bold,italic')
start=text.index('Learned');add('updateTextStyle',objectId=id,textRange={'type':'FIXED_RANGE','startIndex':start,'endIndex':len(text)},style={'fontSize':{'magnitude':25,'unit':'PT'},'foregroundColor':{'opaqueColor':{'rgbColor':rgb('7DB8E5')}}},fields='fontSize,foregroundColor')
# Stop the slide 11 caption touching its diagram.
for e in slides[10]['pageElements']:
 if 'image'in e:
  sz=e['size'];ratio=sz['width']['magnitude']/sz['height']['magnitude'];h=235;w=h*ratio;geom(e['objectId'],80+(810-w)/2,188,w,h)
# Existing native baseline table: fix name, no changed metric values.
for e in slides[11]['pageElements']:
 if 'table'in e:
  for i,row in enumerate(e['table']['tableRows']):
   for j,c in enumerate(row['tableCells']):
    t=''.join(z.get('textRun',{}).get('content','') for z in c.get('text',{}).get('textElements',[]))
    if 'stacked composition' in t:
     loc={'rowIndex':i,'columnIndex':j};add('deleteText',objectId=e['objectId'],cellLocation=loc,textRange={'type':'ALL'});add('insertText',objectId=e['objectId'],cellLocation=loc,insertionIndex=0,text='Learned composition MLP')
# Footer and class-card fixes.
replace('story_39_60','Class-level gains and regressions; no significance claim.',19)
replace('story_39_55','Bus-MovAway-OutgoLane',15);geom('story_39_55',365,236,240,42)
replace('story_39_56','+3.65 points\nUseful tail example',19);geom('story_39_56',365,298,235,88)
for n,text in [(25,'Sources: Khan et al., arXiv:2411.01683, Table III; Zhong et al., PR 2026, Table 3; our study.'),(26,'Sources: Humnabadkar et al. (2024); Zhang et al. (2024, Track 1); Li et al. (2024, Track 3).'),(19,'Code: eval_comb.py; controlled study evaluation and score assembly.'),(27,'Source: implemented BDD-X pilot; selected epoch-2 checkpoint; adapted ROAD comparison protocol.')]:
 for e in slides[n-1]['pageElements']:
  if tx(e).strip() and pos(e)[1]>495:
   replace(e['objectId'],text,10);geom(e['objectId'],230,505,635,18)
# Name each distinct paper's headline method.
for e in slides[25]['pageElements']:
 t=tx(e).strip()
 if t=='HEG / R(2+1)D + graph':replace(e['objectId'],'Humnabadkar et al. / HEG',17)
 if t=='ROAD++ Track 1 winner':replace(e['objectId'],'Zhang et al. / YOLO ensemble',17)
 if t=='ROAD++ Track 3 winner':replace(e['objectId'],'Li et al. / Action-slot ensemble',17)
# All cited additional papers fit the reference page; full URLs in companion.
replace('g3fa12704523_50_1194','Singh et al. ROAD. TPAMI, 2023; Khan et al. ROAD-Waymo, arXiv:2411.01683.\nZhong et al. FRCB. Pattern Recognition 180:114489, 2026.\nHumnabadkar et al. HEG, arXiv:2409.11206, 2024.\nZhang et al. ROAD++ Track 1, arXiv:2410.23077, 2024.\nLi et al. ROAD++ Track 3, arXiv:2410.23092, 2024.\nHe et al. RoIAlign, 2017; Lin et al. Focal Loss, 2017; Wolpert. Stacking, 1992.\nRadford et al. CLIP, 2021; Wang et al. InternVideo2, 2024.\nLocal evidence: 25 shared-frame evaluations; three seeds; full per-class AP.',17)
geom('g3fa12704523_50_1194',75,190,825,280)
# Native row-vector and phrase-matrix grammar; model borders only mark weights.
icons={'ice':[],'fire':[]}
for n in [15,16,17]:
 sl=slides[n-1]['objectId']
 for e in slides[n-1]['pageElements']:
  t=tx(e).strip()
  if t.startswith('YOLOv8x'):
   replace(e['objectId'],'YOLOv8x',17);icons['ice'].append((sl,205,190))
  elif t.startswith('InternVideo2\nFrozen'):
   replace(e['objectId'],'InternVideo2\nVideo tower',16);icons['ice'].append((sl,205,301))
  elif t.startswith('Crop feature\n'):
   delete(e['objectId']);box(sl,'Crop feature',260,301,147,24,16)
   for j in range(8):box(sl,'',260+j*18,333,18,15,fill='A8C9E8' if j%2 else 'DAE8FC',shape='RECTANGLE',stroke='6C8EBF')
   box(sl,'One 1024-vector',260,352,155,23,14)
  elif t.startswith('Frozen phrase matrix'):
   delete(e['objectId']);box(sl,'Phrase matrix: 184 prompts',419,406,243,26,16)
   for i in range(3):
    for j in range(8):box(sl,'',463+j*17,436+i*6,17,6,fill='C5B0CF' if (i+j)%2 else 'E1D5E7',shape='RECTANGLE')
  elif t.startswith('Phrase head\n'):icons['fire'].append((sl,620,294))
  elif t.startswith('Flat head\n'):icons['fire'].append((sl,585,285))
  elif t.startswith('Composition MLP\n'):icons['fire'].append((sl,685,380))
  elif t.startswith('Phrase head evidence'):
   add('updateShapeProperties',objectId=e['objectId'],shapeProperties={'outline':{'weight':{'magnitude':2.4,'unit':'PT'}}},fields='outline.weight');icons['fire'].append((sl,358,403))
 box(sl,'boxes define crops',104,273,170,22,14,'555555')
 # Explicitly acknowledge video pixels, so coordinates cannot be mistaken for encoder features.
 box(sl,'Video pixels +',78,289,125,20,12,'555555')
 # Retain text legend, add its icon swatches in the free right margin.
 icons['ice'].append((sl,840,468));icons['fire'].append((sl,868,468))
# Baseline: use simpler native objects in the ample original diagram station.
sl=slides[7]['objectId']
for e in slides[7]['pageElements']:
 if 'image'in e:delete(e['objectId'])
box(sl,'8-frame clip',90,295,165,70,23,fill='D5E8D4',shape='ROUND_RECTANGLE',stroke='82B366')
box(sl,'3D-RetinaNet\nFrozen detector',345,270,230,115,23,fill='E1D5E7',shape='TRAPEZOID',stroke='9673A6',dash=True)
box(sl,'Predicted boxes\n+ 184 scores',670,295,210,70,23,fill='F5F5F5',shape='ROUND_RECTANGLE',stroke='909090')
line(sl,255,330,345,330);line(sl,575,330,670,330);icons['ice'].append((sl,558,260));box(sl,'Dashed border: frozen weights',80,430,500,30,16,'555555')
# Sequential numbers restore the source footer convention after removing blank placeholders.
for i,s in enumerate(slides,1):
 if i not in [1,30]:box(s['objectId'],str(i),915,504,25,22,11)
(W/'repair-requests.json').write_text(json.dumps(R));(W/'icon-placements.json').write_text(json.dumps(icons));print(len(R),'repair requests')

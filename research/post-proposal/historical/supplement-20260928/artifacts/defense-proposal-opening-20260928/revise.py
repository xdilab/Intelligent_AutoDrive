import json,copy,pathlib,hashlib
W=pathlib.Path('/data/repos/wiki/artifacts/defense-proposal-opening-20260928');R=W.parents[1]
p=json.load(open(W/'proposal.json'))['structuredContent'];d=json.load(open(W/'defense-before.json'))['structuredContent']
req=[];imgs=[];plan=[];maps={};blue={'red':0.0,'green':.22,'blue':.42};black={};white={'red':1,'green':1,'blue':1}
def walk(es):
 for e in es:
  yield e
  yield from walk(e.get('elementGroup',{}).get('children',[]))
def txt(e):return ''.join(t.get('textRun',{}).get('content','') for t in e.get('shape',{}).get('text',{}).get('textElements',[]))
def box(page,oid,text,x,y,w,h,fs=20,color=black,bold=False,font='Arial',italic=False):
 req.extend([{'createShape':{'objectId':oid,'shapeType':'TEXT_BOX','elementProperties':{'pageObjectId':page,'size':{'width':{'magnitude':w,'unit':'PT'},'height':{'magnitude':h,'unit':'PT'}},'transform':{'scaleX':1,'scaleY':1,'translateX':x,'translateY':y,'unit':'PT'}}}},{'insertText':{'objectId':oid,'text':text}},{'updateTextStyle':{'objectId':oid,'textRange':{'type':'ALL'},'style':{'fontFamily':font,'fontSize':{'magnitude':fs,'unit':'PT'},'bold':bold,'italic':italic,'foregroundColor':{'opaqueColor':{'rgbColor':color}}},'fields':'fontFamily,fontSize,bold,italic,foregroundColor'}},{'updateParagraphStyle':{'objectId':oid,'textRange':{'type':'ALL'},'style':{'spaceAbove':{'magnitude':0,'unit':'PT'},'spaceBelow':{'magnitude':5,'unit':'PT'},'lineSpacing':105},'fields':'spaceAbove,spaceBelow,lineSpacing'}}])
def replace(oid,text):req.extend([{'deleteText':{'objectId':oid,'textRange':{'type':'ALL'}}},{'insertText':{'objectId':oid,'text':text}}])
def geom(e):
 t=e['transform'];s=e['size'];return [t.get('translateX',0)/12700,t.get('translateY',0)/12700,s['width']['magnitude']*t.get('scaleX',1)/12700,s['height']['magnitude']*t.get('scaleY',1)/12700]
def addimg(n,page,e,xywh=None):
 oid=f'prop{n:02d}_{e["objectId"]}';x,y,w,h=xywh or geom(e)
 if not xywh:y-=35
 path=str(W/'media'/f'{oid}.png')
 imgs.append({'oid':oid,'page':page,'url':e['image']['contentUrl'],'path':path,'bounds':[x,y,w,h],'source':e['objectId'],'properties':e['image'].get('imageProperties',{})})
# Clone the later native slides before changing their ancestors.
choices=[9,9,10,11,12,13,15,16,18,19,20,21,23,24]
clones=[]
for n,old in enumerate(choices,12):
 s=d['slides'][old-1];mp={s['objectId']:f'rev_{n:02d}',**{e['objectId']:f'r{n:02d}_{e["objectId"]}' for e in walk(s['pageElements'])}};maps[n]=mp
 clones.append({'duplicateObject':{'objectId':s['objectId'],'objectIds':mp}})
 plan.append({'number':n,'exemplar':s['objectId'],'build':'duplicate native defense exemplar','content_source':old,'media':'keep, except Stage 4 image replacement on slide 12'})
 for e in walk(s['pageElements']):
  if txt(e).strip()==str(old):replace(mp[e['objectId']],str(n))
# First 11: proposal content as native text, with defense theme and footer.
for n,s in enumerate(p['slides'][:11],1):
 page=f'def_{n:02d}';source=list(walk(s['pageElements']));plan.append({'number':n,'exemplar':s['objectId'],'build':'adapt proposal native content into existing defense theme','media':'keep all content media; omit decorative title background; preserve logos'})
 for e in d['slides'][n-1]['pageElements']:
  if n==1 and 'image' in e:continue
  req.append({'deleteObject':{'objectId':e['objectId']}})
 if n>1:box(page,f'prop{n:02d}_num',str(n),900,500,35,20,14)
 texts=[e for e in source if txt(e).strip()]
 if n==1:
  main=txt(texts[0]).strip().split('\n');box(page,'prop01_title',main[0],75,65,810,105,34,white,True,'Cambria')
  box(page,'prop01_subtitle','\n'.join(main[1:]),75,180,810,65,23,white,False,'Cambria')
  for k,e in enumerate(texts[1:]):box(page,f'prop01_{e["objectId"]}',txt(e).strip(),75,258+26*k,810,30,20,white)
  logos=[e for e in source if 'image'in e and e['objectId']!='p3_i15']
  for j,e in enumerate(logos):addimg(n,page,e,[85+j*205,382,175,60])
  continue
 # Identify headline by uppercase wording, not section label.
 title=next(e for e in texts if txt(e).strip().isupper())
 for e in texts:
  oid=f'prop{n:02d}_{e["objectId"]}';t=txt(e).strip();x,y,w,h=geom(e)
  if e is title:box(page,oid,t,55,25,850,60,32 if len(t)>30 else 38,blue,True,'Cambria',True);continue
  if y<10:box(page,oid,t,600,84,305,22,13,blue,False,'Cambria',True);continue
  if 'thesis_cite' in e['objectId']:box(page,oid,t,75,476,805,20,10,black);continue
  if n==11 and e['objectId'].endswith('_59'):y=111
  else:y-=35
  fs=18
  if 120<geom(e)[1]<165 and n!=2:fs=20
  if n==2:fs=22;y=118;h=340
  if n==11 and e['objectId'].endswith('_56'):fs=15.5;h=325;y=130
  if 'sources' in e['objectId'] or e['objectId']=='byrd_example_el_5' or (n==10 and e['objectId'].endswith('_5')):fs=11
  if n==9 and e['objectId'].endswith('_2'):fs=18
  box(page,oid,t,x,y,w,h,fs)
  # Preserve bold/italic semantic runs (paragraph titles, label prefixes, limitations).
  for te in e['shape']['text']['textElements']:
   if 'textRun' not in te:continue
   st=te['textRun'].get('style',{});style={k:st[k] for k in ['bold','italic'] if k in st}
   start=te.get('startIndex',0);end=min(te.get('endIndex',len(t)),len(t))
   if style and end>start:req.append({'updateTextStyle':{'objectId':oid,'textRange':{'type':'FIXED_RANGE','startIndex':start,'endIndex':end},'style':style,'fields':','.join(style)}})
  if n in [2,7,11] and e['objectId'] in ['g3fa051c7877_8_16','g3f8a6883d4b_0_18','g3f8a6883d4b_0_56']:
   req.append({'createParagraphBullets':{'objectId':oid,'textRange':{'type':'ALL'},'bulletPreset':'BULLET_DISC_CIRCLE_SQUARE'}})
 for e in source:
  if 'image' in e:addimg(n,page,e)
  if e.get('shape',{}).get('shapeType')=='DOWN_ARROW':
   x,y,w,h=geom(e);oid=f'prop{n:02d}_{e["objectId"]}'
   req.extend([{'createShape':{'objectId':oid,'shapeType':'DOWN_ARROW','elementProperties':{'pageObjectId':page,'size':{'width':{'magnitude':w,'unit':'PT'},'height':{'magnitude':h,'unit':'PT'}},'transform':{'scaleX':1,'scaleY':1,'translateX':x,'translateY':y-35,'unit':'PT'}}}},{'updateShapeProperties':{'objectId':oid,'shapeProperties':{'shapeBackgroundFill':{'solidFill':{'color':{'rgbColor':blue}}},'outline':{'propertyState':'NOT_RENDERED'}},'fields':'shapeBackgroundFill,outline'}}])
 # Preserve source input/output blocks as meaningful diagram elements.
 if n==11:
  for suffix,color in [('_53',{'red':1,'green':.75,'blue':.2}),('_55',blue)]:
   oid=f'prop11_g3f8a6883d4b_0{suffix}'
   req.append({'updateShapeProperties':{'objectId':oid,'shapeProperties':{'shapeBackgroundFill':{'solidFill':{'color':{'rgbColor':color}}},'contentAlignment':'MIDDLE'},'fields':'shapeBackgroundFill,contentAlignment'}})
   req.append({'updateParagraphStyle':{'objectId':oid,'textRange':{'type':'ALL'},'style':{'alignment':'CENTER'},'fields':'alignment'}})
   if suffix=='_55':req.append({'updateTextStyle':{'objectId':oid,'textRange':{'type':'ALL'},'style':{'foregroundColor':{'opaqueColor':{'rgbColor':white}}},'fields':'foregroundColor'}})
# Stage 4 starts method sequence, aligned exactly to Stage 5/6.
replace(maps[12]['d09_p4_i83'],'Stage 4: semantic crop classification')
replace(maps[12]['def_09_caption'],'Semantic Crop Classifier')
req.append({'deleteObject':{'objectId':maps[12]['def_09_image']}})
imgs.append({'oid':'rev12_stage4_image','page':'rev_12','path':str(R/'artifacts/defense-30min/alignment/stage4.png'),'bounds':[50,120,860,317.214],'source':'artifacts/defense-30min/alignment/stage4.drawio'})
# Concise combined blend/DCB mechanism; same approved figure.
replace(maps[18]['d15_p4_i83'],'Expert blending + difficulty-calibrated BCE')
# Result baseline context retained with stage 4-6 numbers in caption.
replace(maps[20]['d18_p15_i177'],'Original triplet AP: Stage 4 9.07 · Stage 5 11.23 · Stage 6 11.28\nThree-seed detector means (%); architectures and studies differ.')
replace(maps[24]['d23_p4_i84'],'Three seeds on one benchmark\nOverall and rare-class gains can diverge\nLanguage content is not fully isolated\nSelected Stage 6 adaptation evaluation is pending')
replace(maps[25]['d24_p18_i230'],'Conclusion & questions')
(W/'clone-requests.json').write_text(json.dumps(clones));(W/'content-requests.json').write_text(json.dumps(req));(W/'images.json').write_text(json.dumps(imgs));(W/'slide-plan.json').write_text(json.dumps(sorted(plan,key=lambda x:x['number']),indent=2));(W/'maps.json').write_text(json.dumps(maps))
print(len(clones),len(req),len(imgs))

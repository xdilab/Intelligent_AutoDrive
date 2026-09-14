import json,math
from pathlib import Path
D=Path('artifacts/proposal-citations/slide-update'); p=json.load(open(D/'raw-template.json')); refs=json.load(open('artifacts/proposal-citations/citation-inventory.json')); req=[]; manifest=[]
def replace(id,text):
 req.extend([{'deleteText':{'objectId':id,'textRange':{'type':'ALL'}}},{'insertText':{'objectId':id,'text':text,'insertionIndex':0}}])
def style(id,size=16):
 req.append({'updateTextStyle':{'objectId':id,'textRange':{'type':'ALL'},'style':{'fontFamily':'Arial','fontSize':{'magnitude':size,'unit':'PT'},'bold':False,'foregroundColor':{'opaqueColor':{'rgbColor':{'red':.31,'green':.31,'blue':.31}}}},'fields':'fontFamily,fontSize,bold,foregroundColor'}})
def box(id,page,text,x,y,w,h,size=16):
 req.append({'createShape':{'objectId':id,'shapeType':'TEXT_BOX','elementProperties':{'pageObjectId':page,'size':{'width':{'magnitude':w,'unit':'PT'},'height':{'magnitude':h,'unit':'PT'}},'transform':{'scaleX':1,'scaleY':1,'translateX':x,'translateY':y,'unit':'PT'}}}})
 req.append({'insertText':{'objectId':id,'text':text,'insertionIndex':0}});style(id,size)
 req.append({'updateParagraphStyle':{'objectId':id,'textRange':{'type':'ALL'},'style':{'spaceAbove':{'magnitude':0,'unit':'PT'},'spaceBelow':{'magnitude':0,'unit':'PT'},'lineSpacing':100},'fields':'spaceAbove,spaceBelow,lineSpacing'}})
def author(r):
 a=r['author'].replace('{','').replace('}','').split(' and '); return a[0].split(',')[0]+(' et al.' if len(a)>1 else '')
def entry(n):
 r=refs[n-1];v=r.get('venue','').replace('IEEE Transactions on Pattern Analysis and Machine Intelligence','TPAMI').replace('Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition','CVPR')
 return f"[{n}] {author(r)} ({r['year']}). {r['title']}. {v}."
# Preserve selected-reference slide at 44; use stable inventory IDs.
s=p['slides'][43];body=s['pageElements'][2]['objectId']; selected=[1,2,4,7,8,6,26,10,9]
replace(body,'\n'.join(entry(n) for n in selected)); req.append({'deleteParagraphBullets':{'objectId':body,'textRange':{'type':'ALL'}}});style(body,14)
req.append({'updateParagraphStyle':{'objectId':body,'textRange':{'type':'ALL'},'style':{'indentStart':{'magnitude':0,'unit':'PT'},'indentEnd':{'magnitude':0,'unit':'PT'},'indentFirstLine':{'magnitude':0,'unit':'PT'},'spaceBelow':{'magnitude':7,'unit':'PT'}},'fields':'indentStart,indentEnd,indentFirstLine,spaceBelow'}})
replace(s['pageElements'][3]['objectId'],'Selected sources · Complete bibliography follows in the appendix');style(s['pageElements'][3]['objectId'],16)
start=0
for n in selected:
 t=entry(n);req.append({'updateTextStyle':{'objectId':body,'textRange':{'type':'FIXED_RANGE','startIndex':start,'endIndex':start+len(t)},'style':{'link':{'url':refs[n-1]['url']}},'fields':'link'}});start+=len(t)+1
# Native reference exemplar, six individually linked rows per appendix page.
for page in range(math.ceil(len(refs)/6)):
 sid=f'thesis_bib_20260907_{page+1:02}'; ids={s['objectId']:sid}; ids.update({e['objectId']:sid+'_e'+str(i) for i,e in enumerate(s['pageElements'])})
 req.append({'duplicateObject':{'objectId':s['objectId'],'objectIds':ids}})
 replace(ids[s['pageElements'][0]['objectId']],f'THESIS BIBLIOGRAPHY {page+1:02} / 20')
 replace(ids[s['pageElements'][1]['objectId']],'Appendix')
 replace(ids[s['pageElements'][3]['objectId']], 'Core methods, datasets & tools' if page<5 else 'Thesis sources & additional reading')
 style(ids[s['pageElements'][3]['objectId']],16)
 req.append({'deleteObject':{'objectId':ids[body]}})
 ns=list(range(page*6+1,min(page*6+7,len(refs)+1)))
 for row,n in enumerate(ns):
  id=sid+'_r'+str(n);box(id,sid,entry(n),79,197+row*47,805,45,15)
  req.append({'updateTextStyle':{'objectId':id,'textRange':{'type':'ALL'},'style':{'link':{'url':refs[n-1]['url']},'underline':False,'foregroundColor':{'opaqueColor':{'rgbColor':{'red':.31,'green':.31,'blue':.31}}}},'fields':'link,underline,foregroundColor'}})
 manifest.append({'slide_id':sid,'references':ns,'exemplar':s['objectId']})
# Small citation strips in the unused band under the university header.
mapping={3:[1,2],4:[1,2],5:[15,16],6:[1,3,2],7:[2,17],8:[2],9:[2],11:[1,2],12:[18,27,28],14:[1,5,11,12,13],15:[1,5,11,13],16:[1,28],17:[9,29],18:[26,9],19:[1,26],20:[1,26],22:[6,11,13],23:[6,11,13],24:[6,31],25:[6,10],26:[6,10],27:[10],32:[4,7,8],33:[7,8,25],34:[5,7,8,25],35:[25],36:[8,25,19,20],37:[7,8],38:[5,7],46:[10]}
for n,ns in mapping.items():
 slide=p['slides'][n-1];sid=slide['objectId'];txt='Sources: '+' · '.join(f'[{k}] {author(refs[k-1])}, {refs[k-1]["year"]}' for k in ns)
 box(f'thesis_cite_{n:02}',sid,txt,75,78,810,18,10)
 req.append({'updateParagraphStyle':{'objectId':f'thesis_cite_{n:02}','textRange':{'type':'ALL'},'style':{'alignment':'END'},'fields':'alignment'}})
 nid=slide.get('slideProperties',{}).get('notesPage',{}).get('notesProperties',{}).get('speakerNotesObjectId')
 if nid:
  note='\n\nCITATION SOURCES (bibliography IDs)\n'+'\n'.join(entry(k)+'\n'+refs[k-1]['url'] for k in ns)
  if n in [9,14,16,18]:note+='\nNumerical results/distributions reported here are this work unless explicitly labeled as paper reference results.'
  if n in [25,26,27,46]:note+='\nWolpert supports the stacking / out-of-fold training principle; the specific composition MLP is this thesis implementation.'
  if n==35:note+='\nThe fixed box window across t−3…t+4 is this thesis crop-construction implementation, not a tracking claim from the model paper.'
  req.append({'insertText':{'objectId':nid,'text':note,'insertionIndex':0}})
json.dump(req,open(D/'requests.json','w'));json.dump(manifest,open(D/'bibliography-slide-map.json','w'),indent=2);json.dump(mapping,open(D/'citation-slide-map.json','w'),indent=2)
print(len(req),'requests;',len(manifest),'bibliography slides;',len(mapping),'citation strips')

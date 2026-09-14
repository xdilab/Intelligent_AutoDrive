import re,json,math
from pathlib import Path
D=Path('artifacts/proposal-citations/prune40');deck=json.load(open(D/'before.json'));old=json.load(open('artifacts/proposal-citations/citation-inventory.json'))
bib=Path('/data/repos/thesis-proposal/bib/references.bib').read_text();records={}
for m in re.finditer(r'@\w+\{([^,]+),',bib):
 start=m.end();level=1;i=start
 while level:
  level+=(bib[i]=='{')-(bib[i]=='}');i+=1
 block=bib[start:i-1];r={'key':m.group(1)}
 for f in re.finditer(r'(\w+)\s*=\s*\{',block):
  j=f.end();k=j;depth=1
  while depth:
   depth+=(block[k]=='{')-(block[k]=='}');k+=1
  r[f.group(1)]=block[j:k-1].replace('{','').replace('}','').replace('\\_','_')
 records[r['key']]=r
order='road road-waymo jaad-crossing pie waymo bdd ava road-r resnet i3d fpn focal-loss mask-rcnn slowfast video-swin videomae class-balanced ldam balanced-softmax logit-adjust decoupling bbn ride paco vl-ltr clip coop cocoop maple tip-adapter xclip efficient-prompt vifi internvideo2 internvideo2-clip-s cge csp voc yolov8 stacking'.split();assert len(order)==len(records)==40
refs=[records[k] for k in order]
def norm(t):return re.sub(r'[^a-z0-9]','',t.lower())
mp={}
for i,r in enumerate(old,1):
 for j,t in enumerate(refs,1):
  if norm(r['title'])==norm(t['title']) or r['url'].rstrip('/')==t['url'].rstrip('/'):mp[i]=j
mp.update({15:3,16:4,18:38,26:39,25:35,10:40})
q=[]
def txt(e):return ''.join(t.get('textRun',{}).get('content','') for t in e.get('shape',{}).get('text',{}).get('textElements',[]))
def replace(id,t):q.extend([{'deleteText':{'objectId':id,'textRange':{'type':'ALL'}}},{'insertText':{'objectId':id,'text':t,'insertionIndex':0}}])
def author(r):a=r['author'].split(' and ');return a[0].split(',')[0]+(' et al.' if len(a)>1 else '')
def venue(r):
 v=r.get('journal',r.get('booktitle',r.get('howpublished','Official documentation' if r['key']=='yolov8' else 'arXiv:'+r.get('eprint',''))))
 for find,to in [('IEEE Transactions on Pattern Analysis and Machine Intelligence','TPAMI'),('International Journal of Computer Vision','IJCV'),('Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition','CVPR'),('Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition','CVPR'),('Proceedings of the IEEE International Conference on Computer Vision Workshops','ICCV Workshops'),('Proceedings of the IEEE International Conference on Computer Vision','ICCV'),('Proceedings of the IEEE/CVF International Conference on Computer Vision','ICCV'),('International Conference on Learning Representations','ICLR'),('European Conference on Computer Vision','ECCV'),('Advances in Neural Information Processing Systems','NeurIPS'),('Proceedings of the 38th International Conference on Machine Learning','ICML')]:v=v.replace(find,to)
 return v
def entry(n):
 r=refs[n-1];date=r.get('year','accessed 2026');return f"[{n}] {author(r)} ({date}). {r['title']}. {venue(r)}."
def link(id,n,start=0,end=None):
 q.append({'updateTextStyle':{'objectId':id,'textRange':{'type':'FIXED_RANGE','startIndex':start,'endIndex':end} if end else {'type':'ALL'},'style':{'link':{'url':refs[n-1]['url']},'underline':False,'foregroundColor':{'opaqueColor':{'rgbColor':{'red':.31,'green':.31,'blue':.31}}}},'fields':'link,underline,foregroundColor'}})
for s in deck['slides']:
 sid=s['objectId']
 if sid.startswith('thesis_bib_'):
  page=int(sid.split('_')[-1]);
  if page>7:q.append({'deleteObject':{'objectId':sid}});continue
  replace(sid+'_e0',f'PROPOSAL REFERENCES {page:02} / 07');replace(sid+'_e3','Same 40 sources and numbering as the written proposal')
  rows=[e for e in s['pageElements'] if re.search(r'_r\d+$',e['objectId'])]
  for i,e in enumerate(rows):
   n=(page-1)*6+i+1
   if n>40:q.append({'deleteObject':{'objectId':e['objectId']}});continue
   replace(e['objectId'],entry(n));link(e['objectId'],n)
  nid=s['slideProperties']['notesPage']['notesProperties']['speakerNotesObjectId'];replace(nid,'PROPOSAL REFERENCES\nSame source set and numbering as the written proposal: 38 research papers and 2 implementation resources.\n\n'+'\n\n'.join(entry(n)+'\n'+refs[n-1]['author']+'\n'+refs[n-1]['url'] for n in range((page-1)*6+1,min(page*6+1,41))))
  continue
 for e in s.get('pageElements',[]):
  if e['objectId'].startswith('thesis_cite_'):
   t=txt(e)
   def ren(m):
    ns=[mp[int(v.strip())] for v in m.group(1).split(',') if int(v.strip()) in mp];return '['+', '.join(map(str,ns))+']' if ns else ''
   t=re.sub(r'\[([0-9, ]+)\]',ren,t);t=re.sub(r' +',' ',t);replace(e['objectId'],t)
 # Convert prior citation block IDs, retaining author names and URLs for supplemental attribution.
 np=s.get('slideProperties',{}).get('notesPage',{});nid=np.get('notesProperties',{}).get('speakerNotesObjectId')
 ne=next((e for e in np.get('pageElements',[]) if e['objectId']==nid),None)
 if ne and 'CITATION SOURCES' in txt(ne):
  oldtext=txt(ne)
  newtext=re.sub(r'\[(\d+)\](?= [A-Z])',lambda m:'['+str(mp[int(m[1])])+']' if int(m[1]) in mp else '[Supplementary source]',oldtext)
  newtext=newtext.replace('CITATION SOURCES (bibliography IDs)','CITATION SOURCES (proposal numbering; supplementary sources retained by name)')
  if oldtext!=newtext:replace(nid,newtext)
# Selected concluding references use the same IDs.
s=deck['slides'][43];body='g3fa12704523_50_1194';selected=[1,2,8,21,26,34,13,39,40];replace(body,'\n'.join(entry(n) for n in selected));offset=0
for n in selected:link(body,n,offset,offset+len(entry(n)));offset+=len(entry(n))+1
replace('g3fa12704523_50_1195','Selected sources · 40 proposal references in appendix')
json.dump(refs,open(D/'proposal-40.json','w'),indent=2);json.dump(mp,open(D/'old-to-proposal-numbering.json','w'),indent=2);json.dump(q,open(D/'requests.json','w'));print(len(q),'requests',len(mp),'overlapping library identities')

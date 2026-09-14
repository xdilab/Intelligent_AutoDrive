import json,re
from pathlib import Path
D=Path('artifacts/proposal-citations');p=json.load(open(D/'notes-order-before.json'));old=json.load(open(D/'slide-update/raw-template.json'))
def get(s):
 np=s.get('slideProperties',{}).get('notesPage',{});nid=np.get('notesProperties',{}).get('speakerNotesObjectId');e=next((e for e in np.get('pageElements',[]) if e['objectId']==nid),{});t=''.join(t.get('textRun',{}).get('content','') for t in e.get('shape',{}).get('text',{}).get('textElements',[]));return nid,t
original={s['objectId']:get(s)[1].strip() for s in old['slides']};req=[];audit=[]
sourceheads={'SOURCE','SOURCES','SOURCE CHECK','TECHNICAL SOURCE','EDITABLE FIGURE','REFERENCE LINKS','SOURCE NOTE','METHOD ATTRIBUTION','SOURCE / INTERPRETATION','ACADEMIC CALENDAR CROSS-CHECK'}
speakheads={'SPEAK','DIAGRAM CUE','TABLE CUE','VISUAL WALKTHROUGH','VISUAL EXAMPLE','TRANSITION','DELIVERY','CONNECTION TO THE PREVIOUS SLIDE','VISUAL WALKTHROUGH AND SOURCE'}
for n,s in enumerate(p['slides'],1):
 if n>46:continue
 nid,full=get(s)
 if not full.strip():continue
 body=full;cite=''
 if 'CITATION SOURCES' in body:
  marker=original[s['objectId']][:100];pos=body.find(marker);assert pos>=0,(n,'boundary not found');cite=body[:pos].strip();body=body[pos:]
 # Split existing labeled sections; title-case Sources is also an attribution heading.
 lines=body.splitlines();chunks=[];heading='';buf=[]
 for line in lines:
  h=line.strip()
  ishead=bool(re.fullmatch(r'[A-Z][A-Z0-9 :?()–/—-]{2,}',h)) or h in ['Sources:','Source:']
  if ishead:
   if buf or heading:chunks.append((heading,'\n'.join(buf).strip()))
   heading=h;buf=[]
  else:buf.append(line)
 if buf or heading:chunks.append((heading,'\n'.join(buf).strip()))
 speaks=[];asks=[];sources=[]
 for h,b in chunks:
  value=(h+'\n'+b).strip()
  if not value:continue
  if h.upper().rstrip(':') in sourceheads:sources.append(value)
  elif h=='' or h in speakheads:speaks.append((h,value))
  else:asks.append(value)
 # Make explicit SPEAK blocks first, transitions last among delivery notes.
 speaks.sort(key=lambda x:0 if x[0]=='SPEAK' else 2 if x[0]=='TRANSITION' else 1)
 if cite:sources.append(cite)
 sections=[]
 if speaks:sections.append('SPEAKER NOTES\n\n'+'\n\n'.join(v for h,v in speaks))
 if asks:sections.append('IF ASKED / SUPPORTING DETAIL\n\n'+'\n\n'.join(asks))
 if sources:sections.append('SOURCES & CITATIONS\n\n'+'\n\n'.join(sources))
 result='\n\n'.join(sections)+'\n'
 # Exact non-whitespace character inventory of original chunks is preserved.
 from collections import Counter
 originalchars=Counter(re.sub(r'\s','',body+cite));reorderedchars=Counter(re.sub(r'\s','',''.join(v for h,v in speaks)+''.join(asks)+''.join(sources)));assert originalchars==reorderedchars,n
 req.extend([{'deleteText':{'objectId':nid,'textRange':{'type':'ALL'}}},{'insertText':{'objectId':nid,'insertionIndex':0,'text':result}}]);audit.append({'slide':n,'slide_id':s['objectId'],'notes_id':nid,'speaker_sections':len(speaks),'supporting_sections':len(asks),'source_sections':len(sources),'after':result})
json.dump(req,open(D/'notes-order-requests.json','w'));json.dump(audit,open(D/'notes-order-audit.json','w'),indent=2);print('Prepared',len(audit),'slides; content-preservation checks passed.')
for n in [12,15,27,34,36]:
 a=next(a for a in audit if a['slide']==n);print(n,a['after'][:90].replace('\n',' / '))

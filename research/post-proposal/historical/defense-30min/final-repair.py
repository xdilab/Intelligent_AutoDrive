import json
from pathlib import Path
W=Path('artifacts/defense-30min');p=json.load(open(W/'raw-repaired-output.json'))['structuredContent'];R=[]
def trans(e,x,y,sx=None,sy=None):
 t=dict(e['transform']);t.update(translateX=x*12700,translateY=y*12700,unit='EMU')
 if sx is not None:t['scaleX']=sx
 if sy is not None:t['scaleY']=sy
 R.append({'updatePageElementTransform':{'objectId':e['objectId'],'applyMode':'ABSOLUTE','transform':t}})
for n,s in enumerate(p['slides'],1):
 for e in s.get('pageElements',[]):
  t=''.join(x.get('textRun',{}).get('content','') for x in e.get('shape',{}).get('text',{}).get('textElements',[])).strip()
  tr=e.get('transform',{});y=tr.get('translateY',0)/12700;x=tr.get('translateX',0)/12700
  if n==8 and t=='Dashed border: frozen weights':trans(e,x,405)
  if n==15:
   if t=='Phrase matrix: 184 prompts':trans(e,x,435)
   if not t and 'shape' in e and 462<x<584 and 435<y<450:trans(e,x,y-28)
  coords={'d30_line_0034':(404,348,470,415),'d30_line_0055':(404,348,470,415),'d30_line_0059':(260,348,227,411),'d30_line_0013':(535,408,535,370)}
  if e['objectId'] in coords:
   a,b,c,d=coords[e['objectId']];trans(e,a,b,(c-a)*12700/3000000,(d-b)*12700/3000000)
(W/'final-repair-requests.json').write_text(json.dumps(R));print(len(R))

import json
from pathlib import Path
w=Path(__file__).parent
p=json.loads((w/'before.json').read_text())['structuredContent']
E={e['objectId']:e for s in p['slides']+p['layouts'] for e in s.get('pageElements',[])}
ref=E['r13_d09_p4_i83']; layout=E['p21_i17']
style=next(t['textRun']['style'] for t in layout['shape']['text']['textElements'] if 'textRun'in t).copy()
style.update(fontSize={'magnitude':38,'unit':'PT'},smallCaps=False)
para=next(t['paragraphMarker']['style'] for t in ref['shape']['text']['textElements'] if 'paragraphMarker'in t)
names=['Outline','Motivation','Problem: understanding the scene','Related work: pedestrian behavior','Related work: road events','Benchmark: ROAD-Waymo','One box, one road event','Research gap: rare road events','Proposed solution','Inputs & outputs']
req=[];changed=[]
for s,title in zip(p['slides'][1:11],names):
 e=next(e for e in s['pageElements'] if ''.join(t.get('textRun',{}).get('content','') for t in e.get('shape',{}).get('text',{}).get('textElements',[])).strip().isupper())
 id=e['objectId'];changed.append(id)
 req.extend([{'deleteText':{'objectId':id,'textRange':{'type':'ALL'}}},{'insertText':{'objectId':id,'text':title,'insertionIndex':0}},{'updateTextStyle':{'objectId':id,'textRange':{'type':'ALL'},'style':style,'fields':','.join(style)}},{'updateParagraphStyle':{'objectId':id,'textRange':{'type':'ALL'},'style':para,'fields':','.join(para)}},{'updatePageElementTransform':{'objectId':id,'transform':ref['transform'],'applyMode':'ABSOLUTE'}}])
 num=next(e for e in s['pageElements'] if e['objectId'].endswith('_num'))
 ns={'fontFamily':'Book Antiqua','weightedFontFamily':{'fontFamily':'Book Antiqua','weight':400},'fontSize':{'magnitude':14,'unit':'PT'},'foregroundColor':{'opaqueColor':{'themeColor':'ACCENT1'}},'bold':False,'italic':False}
 req.extend([{'updatePageElementTransform':{'objectId':num['objectId'],'transform':E['r13_d09_p4_i85']['transform'],'applyMode':'ABSOLUTE'}},{'updateTextStyle':{'objectId':num['objectId'],'textRange':{'type':'ALL'},'style':ns,'fields':','.join(ns)}}])
 changed.append(num['objectId'])
(w/'requests.json').write_text(json.dumps(req))
(w/'plan.json').write_text(json.dumps({'backup':'1DKYoVmxzBeXGUOlC845DId4OSQ6iIJomCS3HxA2lMt0','exemplar':'r13_d09_p4_i83','layout':'p21_i17','slide_ids':[s['objectId'] for s in p['slides'][1:11]],'changed_ids':changed,'titles':names,'scope':'Opening content slides 2–11; title slide and Stage 4 already use their respective intended styles. Preserve existing content, master, layout and all images.'},indent=2))
print(len(req),'requests',p['revisionId'])

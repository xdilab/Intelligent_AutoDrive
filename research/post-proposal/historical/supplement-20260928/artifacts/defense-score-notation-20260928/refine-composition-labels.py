from pathlib import Path
import xml.etree.ElementTree as E,shutil,re,json,hashlib
p=Path('/data/repos/wiki/artifacts/defense-score-notation-20260928');backup=p/'composition-layout-backup';backup.mkdir(exist_ok=True)
for ext in ['drawio','png']:
 if not (backup/f'composition.{ext}').exists():shutil.copy2(p/f'composition.{ext}',backup/f'composition.{ext}')
t=E.parse(backup/'composition.drawio');root=t.find('.//root');cells={c.get('id'):c for c in root}
def label(id,text,x,y,w,h=50,fs=30):
 c=cells[id];c.set('value',text);g=c.find('mxGeometry')
 for k,v in dict(x=x,y=y,width=w,height=h).items():g.set(k,str(v))
 style=re.sub(r'fontSize=[^;]*;|fontStyle=[^;]*;','',c.get('style'));c.set('style',style+f'fontSize={fs};fontStyle=0;')
label('primitive-label','49 primitive scores',1220,85,460,55)
root.remove(cells['primitive-example'])
label('duplex-label','49 duplex scores',2220,215,290)
label('triplet-label','86 triplet scores',2220,375,290)
label('phrase-label','Stage 6 only<br>Phrase-composition scores',1280,480,540,90)
root.remove(cells['interpret'])
label('foot','Illustrative labels. C = concatenation.',40,1010,2430,55,27)
E.indent(t);t.write(p/'composition.drawio',encoding='utf-8',xml_declaration=True)
old={c.get('id'):c for c in E.parse(backup/'composition.drawio').findall('.//mxCell')};new={c.get('id'):c for c in t.findall('.//mxCell')}
changed=[id for id in new if E.tostring(new[id])!=E.tostring(old[id])]
assert set(changed)=={'primitive-label','duplex-label','triplet-label','phrase-label','foot'}
assert set(old)-set(new)=={'primitive-example','interpret'}
(p/'composition-layout-validation.json').write_text(json.dumps({'changed_labels':changed,'removed_redundant_captions':['primitive-example','interpret'],'computational_nodes_and_edges_unchanged':True,'drawio_sha256':hashlib.sha256((p/'composition.drawio').read_bytes()).hexdigest()},indent=2))

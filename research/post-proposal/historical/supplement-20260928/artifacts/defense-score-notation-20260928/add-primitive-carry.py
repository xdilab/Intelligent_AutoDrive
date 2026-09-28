from pathlib import Path
import xml.etree.ElementTree as E,copy,shutil,re,json,hashlib
p=Path(__file__).parent;b=p/'composition-carry-backup';b.mkdir(exist_ok=True)
for ext in ['drawio','png']:
 if not (b/f'composition.{ext}').exists():shutil.copy2(p/f'composition.{ext}',b/f'composition.{ext}')
t=E.parse(b/'composition.drawio');root=t.find('.//root');c={e.get('id'):e for e in root}
def place(id,x,y,w,h):
 g=c[id].find('mxGeometry')
 for k,v in dict(x=x,y=y,width=w,height=h).items():g.set(k,str(v))
def clone(src,id,value,x,y,w,h):
 e=copy.deepcopy(c[src]);e.set('id',id);e.set('value',value);root.append(e);c[id]=e;place(id,x,y,w,h);return e
def wire(src,id,source,target,ex,ey,ix,iy,points=()):
 e=copy.deepcopy(c[src]);e.set('id',id);e.set('source',source);e.set('target',target)
 s=re.sub(r'(?:exitX|exitY|entryX|entryY)=[^;]*;','',e.get('style'));e.set('style',s+f'exitX={ex};exitY={ey};entryX={ix};entryY={iy};')
 a=e.find('mxGeometry/Array');a.clear();a.set('as','points')
 for x,y in points:E.SubElement(a,'mxPoint',x=str(x),y=str(y))
 root.append(e);c[id]=e;return e
# A visible fork carries the same49 scores both to the CompMLP and the final head output.
fork=E.SubElement(root,'mxCell',id='primitive-fork',value='',vertex='1',parent='registration_frame',style='ellipse;fillColor=#111111;strokeColor=#111111;')
E.SubElement(fork,'mxGeometry',x='1567',y='287',width='6',height='6',attrib={'as':'geometry'});c['primitive-fork']=fork
old=c['primitive-concat'];root.remove(old)
a=wire('flat-primitives','primitive-fork-in','primitive','primitive-fork',1,.5,0,.5);a.set('style',a.get('style')+'endArrow=none;')
wire('features-concat','primitive-concat','primitive-fork','concat',.5,1,.5,0)
clone('duplex','primitive-output','p(y | x)',2140,260,290,60)
label=clone('duplex-label','primitive-output-label','48 primitives + agentness',2070,200,420,45);label.set('style',label.get('style')+'fontSize=28;')
wire('features-concat','primitive-carry','primitive-fork','primitive-output',1,.5,0,.5)
clone('duplex-label','output-title','184 head scores',2070,130,420,50)
place('duplex',2140,450,290,60);place('duplex-label',2100,390,370,45)
place('triplet',2140,640,290,60);place('triplet-label',2100,580,370,45)
for id,pts in [('mlp-duplex',[(2050,472.5),(2050,480)]),('mlp-triplet',[(2050,557.5),(2050,670)])]:
 a=c[id].find('mxGeometry/Array');a.clear();a.set('as','points')
 for x,y in pts:E.SubElement(a,'mxPoint',x=str(x),y=str(y))
e=c['mlp-duplex'];e.set('style',re.sub(r'exitY=[^;]+;', 'exitY=0.29411764705882354;',e.get('style')));a=e.find('mxGeometry/Array');a.clear();a.set('as','points')
# Relocate the compact legend to unused lower-left space, keeping the output column quiet.
place('legend-ice',80,930,42,42);place('legend-frozen',135,927,160,46)
place('legend-fire',325,930,42,42);place('legend-trained',375,927,165,46)
E.indent(t);t.write(p/'composition.drawio',encoding='utf-8',xml_declaration=True)
(p/'composition-carry-validation.json').write_text(json.dumps({'primitive_count':49,'primitive_breakdown':{'agentness':1,'agent':10,'action':22,'location':16},'duplex':49,'triplet':86,'head_output_count':184,'carry':'primitive -> fork -> primitive-output; fork -> concat -> CompMLP','implementation':'ROAD_Reason/research/post-proposal/stage56-full/architecture.py:18-21','detector_caveat':'Final detector assembly keeps YOLO agentness/agent; this diagram shows the184-score classifier head output.','drawio_sha256':hashlib.sha256((p/'composition.drawio').read_bytes()).hexdigest()},indent=2))

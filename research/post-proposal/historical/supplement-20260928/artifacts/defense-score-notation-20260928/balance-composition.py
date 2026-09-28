from pathlib import Path
import xml.etree.ElementTree as E,shutil,re,json,hashlib
p=Path(__file__).parent;b=p/'composition-golden-backup';b.mkdir(exist_ok=True)
for ext in ['drawio','png']:
 if not (b/f'composition.{ext}').exists():shutil.copy2(p/f'composition.{ext}',b/f'composition.{ext}')
t=E.parse(b/'composition.drawio');root=t.find('.//root');c={e.get('id'):e for e in root}
def place(id,x,y,w,h):
 g=c[id].find('mxGeometry')
 for k,v in dict(x=x,y=y,width=w,height=h).items():g.set(k,str(v))
def font(id,size=32,bold=False):
 s=re.sub(r'fontSize=[^;]*;|fontStyle=[^;]*;','',c[id].get('style',''));c[id].set('style',s+f'fontSize={size};fontStyle={1 if bold else 0};')
def group(id,x,y,w):
 g=c[id].find('mxGeometry');factor=w/float(g.get('width'));h=float(g.get('height'))*factor
 for child in root:
  if child.get('parent')==id:
   q=child.find('mxGeometry')
   for k in ['x','y','width','height']:q.set(k,str(float(q.get(k,0))*factor))
 place(id,x,y,w,h);return h
def wire(id,ex,ey,ix,iy,points=()):
 e=c[id];s=re.sub(r'(?:exitX|exitY|entryX|entryY)=[^;]*;','',e.get('style'));e.set('style',s+f'exitX={ex};exitY={ey};entryX={ix};entryY={iy};')
 a=e.find('mxGeometry/Array');a.clear();a.set('as','points')
 for x,y in points:E.SubElement(a,'mxPoint',x=str(x),y=str(y))
# Evidence/readout zones divide near x=1570 (62.3% of the fixed2520px canvas).
# Main feature flow centered at y515; upper/lower branches are balanced around it.
h=322.0404774675291*230/280;group('crops',80,515-h/2,230)
place('clip-label',35,675,340,55);font('clip-label',32)
place('video',415,430,275,170);font('video',34);place('video-badge',672,412,33,33)
group('crop',820,490,260);place('crop',820,490,260,50)
# Original vector is single row; match its tiles to the new50px height.
for child in root:
 if child.get('parent')=='crop':child.find('mxGeometry').set('height','50')
place('crop-label',785,560,330,50);font('crop-label',32)
place('flat',840,220,220,136);font('flat',34);place('flat-badge',1041,202,33,33)
place('primitive',1150,220,300,140);font('primitive',29)
place('primitive-label',1070,160,460,45);font('primitive-label',32);c['primitive-label'].set('value','48 primitives + agentness')
place('concat',1545,490,50,50);font('concat',32)
place('comp',1700,430,275,170);font('comp',37);place('comp-badge',1956,412,33,33)
place('duplex',2140,360,290,60);place('triplet',2140,610,290,60)
for id in ['duplex','triplet']:font(id,32);c[id].set('style',c[id].get('style')+'fontStyle=2;')
place('duplex-label',2100,300,370,45);place('triplet-label',2100,550,370,45)
for id in ['duplex-label','triplet-label']:font(id,32)
place('phrase',1150,707.5,300,65);font('phrase',32);c['phrase'].set('style',c['phrase'].get('style')+'fontStyle=2;')
place('phrase-label',1050,610,500,80);font('phrase-label',30)
# Horizontal evidence chain, explicit vertical branches, simple rounded output elbows.
wire('clip-enc',1,.5,0,.5);wire('enc-crop',1,.5,0,.5)
wire('crop-flat',.5,0,.5,1);wire('flat-primitives',1,.5,0,68/140)
wire('primitive-concat',1,.5,.5,0,[(1570,290)])
wire('features-concat',1,.5,0,.5);wire('concat-mlp',1,.5,0,.5)
wire('phrase-concat',1,.5,.5,1,[(1570,740)])
wire('mlp-duplex',1,.25,0,.5,[(2050,472.5),(2050,390)])
wire('mlp-triplet',1,.75,0,.5,[(2050,557.5),(2050,640)])
# One centered example row with generous separation from the architecture.
place('example',300,912,980,46);font('example',32);c['example'].set('style',c['example'].get('style')+'align=right;')
place('example-source',1350,935,1,1)
place('example-triplet',1580,900,820,70);font('example-triplet',32)
wire('example-arrow',1,.5,0,.5)
place('foot',40,1020,2430,40);font('foot',27)
for id in ['legend-ice','legend-fire']:
 g=c[id].find('mxGeometry');g.set('y','132')
for id in ['legend-frozen','legend-trained']:
 g=c[id].find('mxGeometry');g.set('y','129')
E.indent(t);t.write(p/'composition.drawio',encoding='utf-8',xml_declaration=True)
# No scientific labels or connections removed or rewritten.
old={e.get('id'):e for e in E.parse(b/'composition.drawio').findall('.//mxCell')};new={e.get('id'):e for e in t.findall('.//mxCell')}
assert old.keys()==new.keys()
for id,e in new.items():
 assert old[id].get('value')==e.get('value') or (id=='primitive-label' and e.get('value')=='48 primitives + agentness'),id
 assert old[id].get('source')==e.get('source') and old[id].get('target')==e.get('target'),id
for id in ['crops_clips-55','crops_clips-58','crops_clips-62']:
 a=old[id].find('mxGeometry');z=new[id].find('mxGeometry');assert abs(float(a.get('width'))/float(a.get('height'))-float(z.get('width'))/float(z.get('height')))<1e-9
(p/'composition-golden-validation.json').write_text(json.dumps({'canvas':[2520,1080],'main_flow_y':515,'evidence_readout_split':1570/2520,'composition_mlp_ratio':275/170,'architecture_and_labels_unchanged':True,'crop_aspect_preserved':True,'minimum_upper_score_label_y':160,'drawio_sha256':hashlib.sha256((p/'composition.drawio').read_bytes()).hexdigest()},indent=2))

"""Illustrated Stage 4–6 derivatives of approved architecture + contextual visual ancestors."""
from pathlib import Path
import json,copy,hashlib,xml.etree.ElementTree as E
A=Path(__file__).resolve().parent;ROOT=A.parents[1];CTX=ROOT/'artifacts/contextual-evolution-20260924/simplified'
bp=CTX.parent/'build-diagrams.py';ns={'__file__':str(bp)};exec(compile(bp.read_text().split('\nfor spec in SPECS:')[0],str(bp),'exec'),ns);Base=ns['Figure']
source=E.parse(CTX/'01-mlp.drawio');sc={c.get('id'):c for c in source.findall('.//mxCell')}
def absxy(id):
 c=sc[id];g=c.find('mxGeometry');x=float(g.get('x',0));y=float(g.get('y',0));par=c.get('parent')
 if par in sc and sc[par].find('mxGeometry') is not None:
  xx,yy=absxy(par);x+=xx;y+=yy
 return x,y
manifest=[]
class Illustrated(Base):
 def clone(self,id):
  orig=sc[id];c=copy.deepcopy(orig);c.set('parent','1');x,y=absxy(id);c.find('mxGeometry').attrib.update(x=str(x),y=str(y));self.root.append(c);self.cells[id]=c
  todo=[id]
  while todo:
   parent=todo.pop()
   for child in sc.values():
    if child.get('parent')==parent:
     cc=copy.deepcopy(child);self.root.append(cc);self.cells[cc.get('id')]=cc;todo.append(cc.get('id'))
  return c
 def build(self):
  self.graph.set('pageWidth','2520');self.cells['registration_frame'].find('mxGeometry').set('width','2520')
  ids=['title','subtitle','accent','legend-ice','legend-frozen','legend-fire','legend-trained','frame','frame-label','yolo','yolo-badge','candidates','candidates-label','entire','entire-label','clips','clips-label','video','video-badge','encoder-name','crop-pool','crop','crop-label','out','footer']
  if self.num in [4,6]:ids+=['phrases','text','text-badge','bank','bank-label']
  for id in ids:self.clone(id)
  self.cells['title'].set('value',f'Stage {self.num} · '+{4:'Semantic crop classifier',5:'Semantic crop classifier + Comp MLP',6:'Semantic crop classifier + phrase fusion + Comp MLP'}[self.num])
  self.cells['title'].set('style',self.cells['title'].get('style')+'fontSize=35;')
  self.cells['title'].find('mxGeometry').set('width','2040')
  self.cells['subtitle'].set('value',{4:'Classify crop features with frozen label phrases',5:'Learn compositions from crop features and flat primitive scores',6:'Add phrase composition scores to the composition MLP'}[self.num])
  self.cells['entire-label'].set('value','Video clip')
  self.cells['encoder-name'].set('value','InternVideo2-CLIP-S')
  self.cells['footer'].set('value','ROAD-Waymo illustrative GT box · fixed crop window across 8 frames · inference uses YOLO candidate boxes')
  self.edge('frame-yolo','frame','yolo',[(210,227),(280,227)])
  self.edge('yolo-boxes','yolo','candidates',[(470,227),(545,227)])
  self.edge('frame-clip','frame','entire',[(40,227),(20,227),(20,420),(80,420)])
  self.node('crop-window','',171,534,6,6,'ellipse;fillColor=#111111;strokeColor=#111111;spacing=0;')
  self.edge('clip-window','entire','crop-window',[(174,504.333333333),(174,534)],arrow=False)
  self.edge('boxes-window','candidates','crop-window',[(630,850/3),(630,335),(292,335),(292,537),(177,537)],arrow=False)
  self.edge('window-crops','crop-window','clips',[(174,540),(174,580)])
  self.edge('clips-video','clips','video',[(223.20794046875,635),(300,635)])
  self.edge('video-pool','video','crop-pool',[(490,635),(730,635)])
  self.edge('pool-crop','crop-pool','crop',[(855,635),(875,635)])
  self.node('crop-fork','',1147,632,6,6,'ellipse;fillColor=#111111;strokeColor=#111111;spacing=0;')
  self.edge('crop-fork-in','crop','crop-fork',[(1105,635),(1147,635)],arrow=False)
  self.box('assembly','Score assembly\nEvent scores × q',2040,442,180,110,'gold',extra='fontSize=23;')
  self.edge('boxes-assembly','candidates','assembly',[(715,227),(2130,227),(2130,442)])
  self.edge('assembly-output','assembly','out',[(2220,497),(2260,497)])
  self.text('detector-carry','Candidate boxes + confidence q + agent',1170,185,780,35,24,False,'center')
  if self.num in [5,6]:
   self.box('flat-head','Linear classifier',1250,350,190,90,'red',trained=True,extra='fontSize=26;')
   self.vector('flat-scores',1530,380,160,31)
   self.text('flat-label','Flat scores',1510,330,200,35,26,False,'center')
   self.node('flat-fork','',1732,392,6,6,'ellipse;fillColor=#111111;strokeColor=#111111;spacing=0;')
   self.edge('crop-flat','crop-fork','flat-head',[(1150,632),(1150,395),(1250,395)])
   self.edge('flat-score','flat-head','flat-scores',[(1440,395),(1530,395)])
   self.edge('flat-score-fork','flat-scores','flat-fork',[(1690,395),(1732,395)],arrow=False)
   self.edge('flat-assembly','flat-fork','assembly',[(1738,395),(2000,395),(2000,475),(2040,475)])
   self.text('flat-kept','Action / location',1780,340,250,35,24,False,'center')
   self.box('concat','C',1795,620,30,30,'gold',extra='ellipse;fontSize=23;spacing=0;')
   self.edge('flat-concat','flat-fork','concat',[(1735,398),(1735,485),(1810,485),(1810,620)])
   self.text('primitive-label','Primitive scores',1515,445,205,35,24,False,'center')
   self.edge('crop-concat','crop-fork','concat',[(1153,635),(1795,635)])
   self.box('comp-mlp','Comp MLP',1895,590,220,90,'red',trained=True,extra='fontSize=29;')
   self.edge('concat-mlp','concat','comp-mlp',[(1825,635),(1895,635)])
   self.vector('comp-scores',2180,620,160,31)
   self.text('comp-label','Composition scores',2140,688,260,35,25,False,'center')
   self.edge('mlp-scores','comp-mlp','comp-scores',[(2115,635),(2180,635)])
   self.edge('comp-assembly','comp-scores','assembly',[(2260,620),(2260,585),(2130,585),(2130,552)])
  if self.num in [4,6]:
   self.box('projection','Projection',1250,750,190,90,'red',trained=True,extra='fontSize=26;')
   self.box('phrase-classifier','Cosine +\ncalibration',1535,750,195,90,'red',trained=True,extra='fontSize=26;')
   self.vector('phrase-scores',1790,780,160,31,'purple')
   self.text('phrase-label','Phrase scores',1760,840,220,35,26,False,'center')
   self.edge('crop-projection','crop-fork','projection',[(1150,638),(1150,795),(1250,795)])
   self.edge('projection-cosine','projection','phrase-classifier',[(1440,795),(1535,795)])
   self.edge('phrases-text','phrases','text',[(210,843),(280,843)])
   self.edge('text-bank','text','bank',[(470,843),(910,843)])
   self.edge('bank-cosine','bank','phrase-classifier',[(1070,843),(1120,843),(1120,905),(1480,905),(1480,815),(1535,815)])
   self.edge('cosine-scores','phrase-classifier','phrase-scores',[(1730,795),(1790,795)])
   if self.num==4:
    self.edge('phrase-assembly','phrase-scores','assembly',[(1950,795),(2000,795),(2000,520),(2040,520)])
   else:
    self.edge('phrase-concat','phrase-scores','concat',[(1870,780),(1870,735),(1810,735),(1810,650)])
    self.text('phrase-compositions','Composition scores',1930,735,285,35,24)
  self.box('training',{4:'Head scores are sigmoid probabilities before confidence weighting',5:'Head scores are sigmoid probabilities before confidence weighting',6:'Head scores are sigmoid probabilities before confidence weighting'}[self.num],1385,935,1095,70,'gold',extra='fontSize=25;')
  self.schedule=[r['id'] for r in self.routes]
 def save(self):
  # Parenting allows editable tensor tiles and photo stacks without duplicate frames.
  positions={id:{k:float(c.find('mxGeometry').get(k,0)) for k in ['x','y','width','height']} for id,c in self.cells.items()}
  for c in self.root:
   if c.get('parent')=='1' and c.get('id')!='registration_frame':c.set('parent','registration_frame')
  E.indent(self.tree);path=A/(self.slug+'.drawio');self.tree.write(path,encoding='utf-8',xml_declaration=True)
  manifest.append({'stage':self.num,'slug':self.slug,'positions':positions,'routes':self.routes,'badge_ids':self.badges,'source_context':str(CTX/'01-mlp.drawio'),'source_architecture':str(ROOT/f'artifacts/defense-30min/alignment/stage{self.num}.drawio')})
for n in [4,5,6]:
 f=Illustrated((n,f'stage{n}',f'Stage {n}','',''));f.build();f.save()
locked=['frame','frame-label','yolo','candidates','entire','clips','video','crop-pool','crop','crop-label','assembly','out']
assert all(all(m['positions'][k]==manifest[0]['positions'][k] for k in locked) for m in manifest)
(A/'manifest.json').write_text(json.dumps(manifest,indent=2));(A/'alignment-check.json').write_text(json.dumps({'canvas':[2520,1080],'locked_components':locked,'maximum_coordinate_delta':0,'passed':True},indent=2))
paths=[CTX/'01-mlp.drawio',CTX/'image-provenance.json',bp]+[ROOT/f'artifacts/defense-30min/alignment/stage{n}.drawio' for n in [4,5,6]]
(A/'provenance.json').write_text(json.dumps({'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'image_provenance':json.load(open(CTX/'image-provenance.json')),'change':'Illustrated visual restyle only. Original Stage4–6 architectures, not contextual extensions.'},indent=2))
print('Built 3 aligned illustrated figures')

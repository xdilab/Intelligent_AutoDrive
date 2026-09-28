from pathlib import Path
import json,copy,hashlib,xml.etree.ElementTree as E
W=Path('/data/repos/wiki');A=Path(__file__).parent
bp=W/'artifacts/defense-stage456-illustrated-20260928/build.py';ns={'__file__':str(bp)};exec(compile(bp.read_text().split('\nfor n in [4,5,6]:')[0],str(bp),'exec'),ns)
Base=ns['Base'];sc=ns['sc'];absxy=ns['absxy'];manifest=[]
class Detail(Base):
 def setup(self):
  self.graph.set('pageWidth','2520');self.cells['registration_frame'].find('mxGeometry').set('width','2520')
 def photo(self,id,source,x,y,w,h):
  orig=sc[source];g=orig.find('mxGeometry');sx=w/float(g.get('width'));sy=h/float(g.get('height'))
  def cp(src,newid,parent,root=False):
   c=copy.deepcopy(sc[src]);c.set('id',newid);c.set('parent',parent);gg=c.find('mxGeometry')
   for k,scale in [('x',sx),('y',sy),('width',sx),('height',sy)]:gg.set(k,str(float(gg.get(k,0))*scale))
   if root:gg.set('x',str(x));gg.set('y',str(y))
   self.root.append(c);self.cells[newid]=c
   for child in sc.values():
    if child.get('parent')==src:cp(child.get('id'),newid+'_'+child.get('id'),newid)
  cp(source,id,'1',True)
 def legend(self,train=False):
  self.badge('legend-ice',2040,28,'ice',42);self.text('legend-frozen','Frozen',2095,25,160,46,32)
  if train:self.badge('legend-fire',2290,28,'fire',42);self.text('legend-trained','Trained',2340,25,165,46,32)
 def write(self):
  for c in self.root:
   if c.get('parent')=='1' and c.get('id')!='registration_frame':c.set('parent','registration_frame')
  E.indent(self.tree);self.tree.write(A/(self.slug+'.drawio'),encoding='utf-8',xml_declaration=True)
  manifest.append({'slug':self.slug,'routes':self.routes,'badges':self.badges})

def composition():
 f=Detail((1,'composition','Primitive-conditioned composition','',''));f.setup();f.legend(True)
 f.text('scope','Stages 5 and 6: learn triplet scores from shared primitive evidence',40,25,1880,60,42,True)
 f.photo('crops','clips',70,190,280,382)
 f.text('clip-label','8-frame crop sequence',40,600,380,80,36,False,'center')
 f.inherited('video','c10','Video\nencoder',440,280,280,190)
 f.text('enc-name','InternVideo2-CLIP-S',420,500,325,70,33,False,'center')
 f.vector('crop',850,355,300,52);f.text('crop-label','Crop features',820,425,360,50,38,False,'center')
 f.box('flat','Linear\nclassifier',870,130,260,135,'red',trained=True,extra='fontSize=39;')
 f.vector('primitive',1290,170,350,52);f.text('primitive-label','Primitive scores',1260,100,410,60,38,False,'center')
 f.text('primitive-example','Agent · action · location\n+ agentness',1240,240,450,100,35,False,'center')
 f.box('concat','C',1750,354,55,55,'gold',extra='ellipse;fontSize=36;spacing=0;')
 f.box('comp','Comp MLP',1900,315,240,130,'red',trained=True,extra='fontSize=42;')
 f.vector('duplex',2260,280,210,45);f.text('duplex-label','49 duplex scores',2170,202,330,65,34,False,'center')
 f.vector('triplet',2260,440,210,45);f.text('triplet-label','86 triplet scores',2170,505,330,65,34,True,'center')
 f.edge('clip-enc','crops','video',[(350,375),(440,375)])
 f.edge('enc-crop','video','crop',[(720,381),(850,381)])
 f.edge('crop-flat','crop','flat',[(1000,355),(1000,265)])
 f.edge('flat-primitives','flat','primitive',[(1130,197.5),(1290,197.5)])
 f.edge('primitive-concat','primitive','concat',[(1640,197.5),(1777.5,197.5),(1777.5,354)])
 f.edge('features-concat','crop','concat',[(1150,381),(1750,381)])
 f.edge('concat-mlp','concat','comp',[(1805,381),(1900,381)])
 f.edge('mlp-duplex','comp','duplex',[(2140,350),(2200,350),(2200,302.5),(2260,302.5)])
 f.edge('mlp-triplet','comp','triplet',[(2140,410),(2200,410),(2200,462.5),(2260,462.5)])
 f.vector('phrase',1390,585,320,50,'purple');f.text('phrase-label','Stage 6 also adds\nphrase composition scores',1270,665,540,100,36,False,'center')
 f.edge('phrase-concat','phrase','concat',[(1710,610),(1777.5,610),(1777.5,409)])
 f.text('example','Bus + MovAway + OutgoLane',500,810,1080,70,48,True,'center')
 f.box('example-triplet','Bus-MovAway-OutgoLane',1735,795,730,95,'gray',extra='fontSize=44;')
 f.node('example-source','',1580,845,1,1,'fillColor=none;strokeColor=none;')
 f.edge('example-arrow','example-source','example-triplet',[(1581,845),(1735,845)])
 f.text('interpret','Shared primitive scores and crop features condition learned compositions.',460,920,2010,62,42,False,'center')
 f.text('foot','Illustrative bus labels. C = concatenation. Standalone contextual heads predict 184 scores directly.',40,1010,2430,55,34,False,'center')
 f.write()

def roi():
 f=Detail((2,'roi-align','RoIAlign','',''));f.setup();f.legend()
 f.text('scope','Contextual heads: extract the actor region from the entire-frame feature map',40,25,1940,70,40,True)
 f.photo('frame','candidates',50,360,330,220);f.text('frame-label','Entire-frame clip',40,625,370,55,38,False,'center')
 f.inherited('yolo','c6','YOLOv8x',460,110,275,145)
 f.box('coords','Candidate box\n(x₁, y₁, x₂, y₂)',900,110,375,145,'gray',extra='fontSize=38;')
 f.inherited('video','c10','Video\nencoder',460,380,275,180)
 f.text('encoder','InternVideo2-CLIP-S',420,605,350,55,32,False,'center')
 # Schematic map from established visual ancestor, same aspect ratio as input photo.
 f.node('map','',880,360,330,220,'container=1;pointerEvents=0;fillColor=none;strokeColor=none;')
 for row in range(16):
  for col in range(16):f.node(f'map-{row}-{col}','',col*330/16,row*220/16,330/16,220/16,'rounded=0;fillColor='+['#85B6E0','#D6E8FA','#4F91C6'][(row+col)%3]+';strokeColor=#3977A8;strokeWidth=0.6;',parent='map')
 f.node('map-roi','',330*.03733618,220*.33923291,330*(.32780179-.03733618),220*(.76802331-.33923291),'rounded=0;fillColor=none;strokeColor=#e6ad00;strokeWidth=4;',parent='map')
 f.text('map-label','Keyframe feature map\n16 × 16 locations',800,615,490,100,38,False,'center')
 f.box('align','RoIAlign',1390,415,270,110,'gold',extra='fontSize=42;')
 # Exact 7x7 illustrative pooled spatial grid. Not measured activations.
 f.node('grid','',1770,340,260,260,'container=1;pointerEvents=0;fillColor=none;strokeColor=none;')
 for row in range(7):
  for col in range(7):f.node(f'bin-{row}-{col}','',col*260/7,row*260/7,260/7,260/7,'rounded=0;fillColor='+['#85B6E0','#D6E8FA','#4F91C6'][(row+col)%3]+';strokeColor=#3977A8;',parent='grid')
 f.text('grid-label','7 × 7 region features',1720,630,365,60,36,False,'center')
 f.box('mean','Spatial\nmean',2140,415,160,110,'gold',extra='fontSize=36;')
 f.vector('roi',2370,442,110,56)
 f.text('roi-label','RoI\nfeatures',2300,585,200,100,36,False,'center')
 f.edge('frame-yolo','frame','yolo',[(215,360),(215,182.5),(460,182.5)])
 f.edge('yolo-coords','yolo','coords',[(735,182.5),(900,182.5)])
 f.edge('frame-video','frame','video',[(380,470),(460,470)])
 f.edge('video-map','video','map',[(735,470),(880,470)])
 f.edge('coords-align','coords','align',[(1275,182.5),(1525,182.5),(1525,415)])
 f.text('coords-note','Keep fractional coordinates',1610,200,650,55,34)
 f.edge('map-align','map','align',[(1210,470),(1390,470)])
 f.edge('align-grid','align','grid',[(1660,470),(1770,470)])
 f.edge('grid-mean','grid','mean',[(2030,470),(2140,470)])
 f.edge('mean-vector','mean','roi',[(2300,470),(2370,470)])
 # Zoom into bilinear sampling: four neighboring map positions surround a gold sample.
 for id,x,y in [('a',650,770),('b',820,770),('c',650,895),('d',820,895)]:f.node('point-'+id,'',x,y,18,18,'ellipse;fillColor=#4F91C6;strokeColor=#3977A8;')
 f.node('sample','',712,820,22,22,'ellipse;fillColor=#f2c64d;strokeColor=#d6b656;')
 for id,pt in [('a',(659,779)),('b',(829,779)),('c',(659,904)),('d',(829,904))]:
  f.edge('weight-'+id,'point-'+id,'sample',[pt,(723,831)],dashed=True,arrow=False)
 f.text('sample-label','One sampling point',530,943,430,50,32,False,'center')
 f.text('explain','Bilinear interpolation mixes nearby feature values.\nBin pooling gives a fixed grid; spatial mean gives one vector.',1040,805,1430,150,41)
 f.text('foot','Illustrative GT box and schematic features. Evaluation uses YOLO boxes; pixel crops form a separate branch.',40,1010,2430,55,34,False,'center')
 f.write()
def tail():
 import base64
 f=Detail((3,'tail-case','A tail prediction can lose rank','',''));f.setup()
 f.text('scope','Large vehicle stopped at a junction',40,25,2200,70,48,True)
 img=Path('/data/datasets/ROAD_plusplus/rgb-images/train_00797/00018.jpg');uri='data:image/jpeg,'+base64.b64encode(img.read_bytes()).decode()
 f.node('photo','',40,140,1060,1060*2/3,'shape=image;imageAspect=0;aspect=fixed;image='+uri+';')
 xy=[.3964844048023224,.44062501192092896,.4613281488418579,.5337890982627869]
 f.node('candidate','',40+1060*xy[0],140+1060*2/3*xy[1],1060*(xy[2]-xy[0]),1060*2/3*(xy[3]-xy[1]),'rounded=0;fillColor=none;strokeColor=#e6ad00;strokeWidth=5;')
 f.text('label','LarVeh-Stop-Jun',40,885,1060,65,46,True,'center')
 f.text('tail','Deep triplet tail: 1,895 training boxes',40,962,1060,65,38,False,'center')
 f.text('table-heading','Rank of the matched ground truth',1190,135,1250,70,42,True)
 f.text('budget','Top 615 predictions per class: diagnostic budget',1190,230,1250,70,36)
 rows=[('Stage 4 · phrase classifier','288','Inside',True),('Stage 5 · visual composition','81,918','Outside',False),('Stage 6 · phrase fusion','4,708','Outside',False)]
 for i,(label,rank,status,ok) in enumerate(rows):
  y=350+145*i
  f.box('row'+str(i),'',1190,y,1250,115,'green' if ok else 'gray',extra='strokeWidth=1;')
  f.text('name'+str(i),label,1220,y+24,740,65,40)
  f.text('rank'+str(i),rank,1960,y+24,245,65,42,True,'center')
  f.text('status'+str(i),status,2220,y+24,210,65,34,False,'center')
 f.text('takeaway','Useful phrase evidence can lose rank after fusion.',1190,830,1250,115,44,True)
 f.text('scope-note','ROAD-Waymo · original controlled study, seed 0 · candidate IoU 0.679 · post-hoc ranking example',40,1040,2420,35,30,False,'center')
 f.write()
composition();roi();tail()
(A/'figure-manifest.json').write_text(json.dumps(manifest,indent=2))
print('wrote', [x['slug'] for x in manifest])

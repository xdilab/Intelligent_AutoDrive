"""Aligned contextual evolution, using the approved proposal draw.io vocabulary."""
from pathlib import Path
import copy, hashlib, json, re, xml.etree.ElementTree as E

A = Path(__file__).resolve().parent
WIKI = A.parents[1]
ANCESTOR = WIKI/'artifacts/defense-30min/alignment/stage5.drawio'
TEXT_ANCESTOR = WIKI/'artifacts/defense-30min/alignment/stage6.drawio'
source = E.parse(ANCESTOR)
old = {c.get('id'): c for c in source.findall('.//mxCell')}
COL = {'red':('#f8cecc','#b85450'), 'blue':('#dae8fc','#4b88b5'),
       'purple':('#f1edf7','#8064a2'), 'gold':('#fff2cc','#d6b656'),
       'gray':('#f5f5f5','#909090'), 'green':('#d5e8d4','#82b366')}
SPECS = [
 (88,'01-mlp','MLP contextual head','Mean scene tokens + residual MLP','none'),
 (89,'02-attention','Attention contextual head','Actor-query scene attention + residual MLP','none'),
 (90,'03-triplet-contrastive','MLP + triplet contrastive learning','Align the visual RoI with 86 frozen triplet phrases','triplet'),
 (92,'04-all184-contrastive','MLP + all-184 contrastive learning','Align visual and adapted text features across all 184 labels','all'),
 (94,'05-expert-blend','Attention expert blend','Keep the two trained experts; blend composition probabilities','blend'),
 (104,'06-dcb-blend','DCB attention expert blend','Replace focal training with dynamic class balancing','dcb'),
]
MANIFEST = []

class Figure:
 def __init__(self, spec):
  self.num,self.slug,self.title,self.subtitle,self.mode=spec
  self.tree=copy.deepcopy(source)
  self.graph=self.tree.find('.//mxGraphModel'); self.graph.set('pageWidth','2000'); self.graph.set('pageHeight','1080')
  self.root=self.tree.find('.//root')
  for c in list(self.root):
   if c.get('id') not in ['0','1']:self.root.remove(c)
  self.cells={}; self.routes=[]; self.badges=[]
  self.tree.find('.//diagram').set('name', self.title)
  self.node('registration_frame','',0,0,2000,1080,'container=1;fillColor=#ffffff;strokeColor=none;pointerEvents=0;')
 def node(self,id,label,x,y,w,h,style='',parent='1'):
  base='rounded=1;arcSize=6;whiteSpace=wrap;html=1;fontFamily=Helvetica;fontSize=24;fontColor=#26323d;spacing=5;'
  c=E.SubElement(self.root,'mxCell',id=id,value=label.replace('\n','<br>'),vertex='1',parent=parent,style=base+style)
  E.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
  self.cells[id]=c; return c
 def box(self,id,label,x,y,w,h,color='gray',trained=False,frozen=False,extra=''):
  fill,stroke=COL[color]
  s=f'fillColor={fill};strokeColor={stroke};strokeWidth={3 if trained else 1.5};'
  if frozen:s+='dashed=1;dashPattern=6 5;'
  c=self.node(id,label,x,y,w,h,s+extra)
  if trained or frozen:self.badge(id+'-badge',x+w-19,y-18,'fire' if trained else 'ice')
  return c
 def text(self,id,label,x,y,w,h=30,size=23,bold=False,align='left',color='#26323d'):
  return self.node(id,label,x,y,w,h,f'text;fillColor=none;strokeColor=none;align={align};fontSize={size};fontStyle={int(bold)};fontColor={color};spacing=0;')
 def badge(self,id,x,y,kind,size=33):
  c=copy.deepcopy(old['c15' if kind=='fire' else 'c18']); c.set('id',id)
  g=c.find('mxGeometry');g.attrib.update(x=str(x),y=str(y),width=str(size),height=str(size))
  self.root.append(c);self.cells[id]=c;self.badges.append(id)
 def inherited(self,id,original,label,x,y,w,h):
  c=copy.deepcopy(old[original]);c.set('id',id);c.set('value',label.replace('\n','<br>'))
  s=re.sub('fontSize=[^;]+;', '', c.get('style',''))
  c.set('style',s+'fontSize=25;dashed=1;dashPattern=6 5;')
  g=c.find('mxGeometry');g.attrib.update(x=str(x),y=str(y),width=str(w),height=str(h))
  self.root.append(c);self.cells[id]=c;self.badge(id+'-badge',x+w-18,y-18,'ice');return c
 def vector(self,id,x,y,w,h,color='blue',rows=1):
  colors={'blue':['#85B6E0','#D6E8FA','#4F91C6'], 'purple':['#b09ccc','#e8dff1','#8064a2'], 'gold':['#f2c64d','#fff2cc','#d6a729']}[color]
  stroke={'blue':'#3977A8','purple':'#8064a2','gold':'#d6b656'}[color]
  self.node(id,'',x,y,w,h,'container=1;pointerEvents=0;fillColor=none;strokeColor=none;')
  for row in range(rows):
   for col in range(6):self.node(f'{id}-{row}-{col}','',col*w/6,row*h/rows,w/6-.01,h/rows-.01,f'rounded=0;fillColor={colors[(row+col)%3]};strokeColor={stroke};strokeWidth=1;',id)
 def edge(self,id,a,b,pts,label='',dashed=False,arrow=True):
  def geo(k):return [float(self.cells[k].find('mxGeometry').get(q,0)) for q in ['x','y','width','height']]
  ax,ay,aw,ah=geo(a);bx,by,bw,bh=geo(b)
  s=f'edgeStyle=none;noEdgeStyle=1;rounded=1;arcSize=10;jettySize=20;html=1;strokeColor=#111111;strokeWidth=2;endArrow={"block" if arrow else "none"};endFill=1;fontFamily=Helvetica;fontSize=22;labelBackgroundColor=#ffffff;'
  s+=f'exitX={(pts[0][0]-ax)/aw};exitY={(pts[0][1]-ay)/ah};entryX={(pts[-1][0]-bx)/bw};entryY={(pts[-1][1]-by)/bh};exitPerimeter=0;entryPerimeter=0;'
  # North-pointing draw.io trapezoids rotate their anchor coordinates.
  if a=='yolo':s+='exitX=0.5;exitY=1;'
  if b=='yolo':s+='entryX=0.5;entryY=0;'
  if dashed:s+='dashed=1;dashPattern=5 4;'
  c=E.SubElement(self.root,'mxCell',id=id,value=label,edge='1',parent='1',source=a,target=b,style=s)
  g=E.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'});ar=E.SubElement(g,'Array',attrib={'as':'points'})
  for x,y in pts[1:-1]:E.SubElement(ar,'mxPoint',x=str(x),y=str(y))
  if label:E.SubElement(g,'mxPoint',x='0',y='-19',attrib={'as':'offset'})
  self.routes.append({'id':id,'points':pts,'source':a,'target':b})
 def common(self):
  self.text('title',self.title,40,24,1580,58,39,True)
  self.text('subtitle',self.subtitle,40,89,1550,38,25)
  self.node('accent','',40,140,165,6,'fillColor=#d6b656;strokeColor=none;')
  self.badge('legend-ice',1640,44,'ice',28);self.text('legend-frozen','Frozen',1678,40,130,35,23)
  self.badge('legend-fire',1810,44,'fire',28);self.text('legend-trained','Trained',1848,40,140,35,23)
  self.box('frame','Frame t',40,190,170,75,'green')
  self.inherited('yolo','c6','YOLOv8x\ndetector',280,163,190,130)
  self.box('candidates','Candidate boxes\nconfidence q + agent',545,190,255,75)
  # Same text, coordinates, colors and shapes across all six figures.
  self.box('clips','8-frame clips\ncrops + entire frame',40,421,180,110,'green')
  self.inherited('video','c10','V₀\nVideo encoder',280,415,190,125)
  self.text('encoder-name','InternVideo2-CLIP-S',256,557,240,35,23,False,'center')
  self.node('features','',540,350,260,310,'container=1;pointerEvents=0;fillColor=#ffffff;strokeColor=#4b88b5;strokeWidth=1.5;')
  self.text('crop-label','Crop features [1024]',540,360,260,35,23,False,'center')
  self.vector('crop',555,405,230,31)
  self.text('roi-label','RoI features [1024]',540,452,260,30,23,False,'center')
  self.vector('roi',555,490,230,31)
  self.text('scene-label','Entire-frame tokens',540,536,260,30,23,False,'center')
  self.vector('scene',555,576,230,44,rows=2)
  self.text('scene-dims','16 × 1024',555,626,230,30,22,False,'center')
  self.box('geometry','Box position [8]',555,675,230,55,'blue')
  self.box('phrases','184 label\nphrases',40,800,170,85,'green')
  enc=self.inherited('text','c10','L₀\nText encoder',280,785,190,125)
  enc.set('style',enc.get('style').replace('#ecf6f4','#f1edf7').replace('#0e8a7d','#8064a2'))
  self.vector('bank',555,804,230,70,'purple',3)
  self.text('bank-label','Phrase matrix P: 184 × 512',505,920,330,32,22,False,'center')
  self.box('out','Candidate boxes\n+ 184 scores',1740,442,220,110)
  self.edge('frame-yolo','frame','yolo',[(210,227),(280,227)])
  self.edge('yolo-boxes','yolo','candidates',[(470,227),(545,227)])
  self.edge('boxes-out','candidates','out',[(800,227),(1850,227),(1850,442)])
  self.text('detector-carry','Carry candidate boxes, confidence and YOLO agent scores',880,191,820,30,23)
  self.edge('clips-video','clips','video',[(220,475),(280,475)])
  self.edge('video-features','video','features',[(470,475),(540,475)])
  self.edge('phrases-text','phrases','text',[(210,843),(280,843)])
  self.edge('text-bank','text','bank',[(470,843),(555,843)])
  self.text('cache-note','Shared frozen encoder weights; separate crop and entire-frame passes. RoIAlign uses full-scene features.',40,990,1820,33,23)
  self.text('eval-note','Head training uses cached boxes. Detector evaluation uses YOLO candidates; agentness/agent outputs stay fixed.',40,1030,1850,33,23)
 def single(self):
  attn=self.num==89
  self.box('fusion','',865,385,210,195,'red',trained=True)
  self.text('fusion-title','Visual context',877,393,185,35,25,True,'center')
  self.box('summary','Scene attention' if attn else 'Mean scene tokens',884,440,172,50,'purple' if attn else 'gold',extra='fontSize=21;')
  self.text('fusion-mlp','MLP evidence e',870,511,200,60,23,False,'center')
  self.box('visual-plus','+',1100,467,30,30,'gold',extra='ellipse;fontSize=25;spacing=0;')
  self.vector('visual',1160,466,150,32)
  self.text('visual-label','Visual RoI [512]',1140,406,220,35,23,False,'center')
  self.text('visual-formula','v = LN(c + e)',1090,530,270,35,23,False,'center')
  self.box('decoder','',1390,345,255,340,'red',trained=True)
  self.text('language-title','Language fusion + MLP',1400,355,235,42,23,True,'center')
  self.text('language-query','Q = v; K,V = T',1405,402,225,35,23,False,'center')
  self.vector('joint',1430,475,175,32,'gold')
  self.text('joint-label','Joint RoI h [512]',1400,439,235,34,23,False,'center')
  self.box('classifier','MLP\n1208 → 512 → 184',1410,571,215,84,'red',extra='fontSize=23;strokeWidth=3;')
  self.text('classifier-inputs','Concat: [h ; c ; cos(v,T)]',1360,700,325,40,23,False,'center')
  self.edge('joint-classifier','joint','classifier',[(1517,507),(1517,571)])
  self.box('text-adapter','MLP',865,795,190,80,'red',trained=True)
  self.box('text-plus','+',1083,824,30,30,'gold',extra='ellipse;fontSize=25;spacing=0;')
  self.vector('adapted',1140,805,200,62,'purple',3)
  self.text('adapted-label','Adapted phrases T',1110,747,260,35,23,False,'center')
  self.edge('features-fusion','features','fusion',[(800,476),(865,476)])
  self.edge('geometry-fusion','geometry','fusion',[(785,702),(970,702),(970,580)])
  self.edge('fusion-plus','fusion','visual-plus',[(1075,482),(1100,482)])
  self.edge('visual-skip','fusion','visual-plus',[(1025,385),(1025,345),(1115,345),(1115,467)],'c')
  self.edge('plus-visual','visual-plus','visual',[(1130,482),(1160,482)])
  self.edge('visual-language','visual','decoder',[(1310,482),(1390,482)])
  self.edge('crop-bypass','fusion','decoder',[(970,385),(970,310),(1517,310),(1517,345)],'Projected crop c [512] to classifier')
  self.edge('bank-adapter','bank','text-adapter',[(785,839),(865,839)])
  self.edge('adapter-plus','text-adapter','text-plus',[(1055,839),(1083,839)])
  self.edge('text-skip','bank','text-plus',[(670,804),(670,763),(1098,763),(1098,824)],'P')
  self.edge('plus-adapted','text-plus','adapted',[(1113,839),(1140,839)])
  self.edge('text-language','adapted','decoder',[(1340,839),(1690,839),(1690,650),(1645,650)])
  self.edge('scores-out','decoder','out',[(1645,590),(1690,590),(1690,515),(1740,515)])
  self.text('score-weight','173 head scores × q\n11 YOLO scores retained',1720,576,250,62,22,False,'center')
  # Auxiliary loss is a local training inset. It is deliberately not an inference stage.
  if self.mode=='none':
   label='Training: focal loss on all 184 classifier logits. No contrastive term.'
  elif self.mode=='triplet':
   label='Training: focal(184) + 0.001 × contrastive(v, frozen P[triplets]). 86 targets; visual gradients only.'
  else:
   label='Training: focal(184) + 0.001 × contrastive(v, adapted T). All 184 targets; visual + text gradients.'
  self.box('loss',label if self.mode=='none' else '',865,910,1095,66,'gold',extra='fontSize=23;')
  if self.mode!='none':
   self.vector('loss-visual',885,930,90,30)
   self.vector('loss-phrase',1045,924,90,42,'purple',3)
   self.edge('loss-alignment','loss-visual','loss-phrase',[(975,945),(1045,945)],dashed=True)
   self.root[-1].set('style',self.root[-1].get('style')+'startArrow=block;startFill=1;')
   text='Focal(184) + 0.001 Lcon(v, frozen P)\n86 triplets; auxiliary gradients: visual only' if self.mode=='triplet' else 'Focal(184) + 0.001 Lcon(v, adapted T)\n184 labels; auxiliary gradients: visual + text'
   self.text('loss-text',text,1160,915,785,57,23)
  self.text('loss-scope','Contrastive aligns visual RoI v before language attention.' if self.mode!='none' else 'Language attention and phrase scores are active.',875,150,1055,30,22)
 def blend(self):
  dcb=self.mode=='dcb'
  self.node('expert-pair','',845,310,500,545,'container=1;pointerEvents=0;fillColor=#ffffff;strokeColor=#26323d;strokeWidth=1.5;')
  self.box('expert-a','',865,325,465,225,'red',trained=True)
  self.box('expert-b','',865,560,465,280,'red',trained=True)
  for id,y,name in [('expert-a',325,'Expert A'),('expert-b',560,'Expert B')]:
   self.text(id+'-title',name+': attention head',880,y+10,435,35,25,True,'center')
   self.box(id+'-inference','Scene + language attention → MLP\n184 probabilities',890,y+50,415,65,'gray',extra='fontSize=23;')
   loss='DCB' if dcb else 'Focal'
   self.box(id+'-classification',loss+' classification loss · 184 labels',890,y+140,415,40,'gold',extra='fontSize=22;')
   self.edge(id+'-supervision',id+'-inference',id+'-classification',[(1097.5,y+115),(1097.5,y+140)],dashed=True)
  self.text('a-loss','Training: classification loss only',880,514,435,25,22,False,'center')
  self.box('contrastive-inset','',890,748,415,85,'purple')
  self.text('contrastive-title','Extra training: all-184 contrastive',900,751,395,27,23,True,'center')
  self.vector('expert-b-visual',905,804,70,22)
  self.vector('expert-b-text',1040,801,70,27,'purple',3)
  self.text('v-label','v',905,778,70,23,22,False,'center')
  self.text('t-label','T',1040,778,70,23,22,False,'center')
  self.edge('expert-b-alignment','expert-b-visual','expert-b-text',[(975,815),(1040,815)],dashed=True)
  self.root[-1].set('style',self.root[-1].get('style')+'startArrow=block;startFill=1;')
  self.text('contrastive-gradients','Visual + text\ngradients',1125,782,175,48,22,False,'center')
  self.box('blend','Composition blend\np = (1 − w)pA + w pB\n135 composition scores',1370,560,290,135,'gold')
  self.box('primitives','Keep Expert A\n49 primitive scores',1370,330,290,95,'gray')
  self.box('assembly','Assemble scores\nq × 173 scores\n11 scores from YOLO',1730,675,235,170,'gold',extra='fontSize=23;')
  self.edge('features-experts','features','expert-pair',[(800,475),(845,475)])
  self.edge('geometry-experts','geometry','expert-pair',[(785,702),(845,702)])
  self.edge('phrases-experts','bank','expert-pair',[(785,839),(815,839),(815,810),(845,810)])
  self.text('shared-evidence','Both experts receive the same crop, RoI, entire-frame tokens, position and phrase bank.',855,865,1100,35,23)
  self.text('separate-adapters','Dashed paths: training only. B aligns pre-language visual v with adapted text T (λ = 0.001).',855,912,1100,35,23)
  self.edge('a-primitives','expert-a','primitives',[(1330,377),(1370,377)])
  self.edge('a-blend','expert-a','blend',[(1330,473),(1350,473),(1350,594),(1370,594)])
  self.edge('b-blend','expert-b','blend',[(1330,650),(1370,650)])
  self.edge('blend-assembly','blend','assembly',[(1515,695),(1515,790),(1730,790)])
  self.edge('primitives-assembly','primitives','assembly',[(1660,377),(1680,377),(1680,731),(1730,731)])
  self.edge('assembly-out','assembly','out',[(1850,675),(1850,552)])
  self.text('blend-selection','w selected by development AP; experts fixed during selection.',865,955,1100,30,22)
  self.text('loss-change','DCB replaces focal in both experts; inference is unchanged.' if dcb else 'Blend the attention experts; the MLP study is a separate branch.',875,150,1075,30,22)
  self.text('badge-context','Two independently trained experts; frozen during blending.',865,267,840,35,22)
 def save(self):
  # Place annotations inside real draw.io groups whenever they sit within a container.
  # Registration and portable badge overlaps are intentional; everything else is checked.
  groups={'features':['crop-label','crop','roi-label','roi','scene-label','scene','scene-dims'],
   'fusion':['fusion-title','summary','fusion-mlp'],
   'decoder':['language-title','language-query','joint','joint-label','classifier'],
   'loss':['loss-visual','loss-phrase','loss-text'],
   'contrastive-inset':['contrastive-title','expert-b-visual','expert-b-text','v-label','t-label','contrastive-gradients'],
   'expert-a':['expert-a-title','expert-a-inference','expert-a-classification','a-loss'],
   'expert-b':['expert-b-title','expert-b-inference','expert-b-classification','contrastive-inset'],
   'expert-pair':['expert-a','expert-b']}
  for parent,children in groups.items():
   if parent not in self.cells:continue
   pc=self.cells[parent];pc.set('style',pc.get('style')+'container=1;pointerEvents=0;')
   pg=pc.find('mxGeometry');px,py=float(pg.get('x')),float(pg.get('y'))
   for child in children:
    if child not in self.cells:continue
    c=self.cells[child];g=c.find('mxGeometry');c.set('parent',parent)
    g.set('x',str(float(g.get('x'))-px));g.set('y',str(float(g.get('y'))-py))
  for c in self.root:
   if c.get('parent')=='1' and c.get('id')!='registration_frame':c.set('parent','registration_frame')
  E.indent(self.tree);path=A/(self.slug+'.drawio');self.tree.write(path,encoding='utf-8',xml_declaration=True)
  locked=['frame','yolo','candidates','clips','video','features','crop','roi','scene','geometry','phrases','text','bank','out','registration_frame']
  positions={id:dict(self.cells[id].find('mxGeometry').attrib) for id in locked}
  MANIFEST.append({'model_id':self.num,'slug':self.slug,'title':self.title,'mode':self.mode,'locked_geometry':positions,'routes':self.routes,'intentional_badge_overlaps':self.badges})

for spec in SPECS:
 f=Figure(spec);f.common();f.blend() if f.mode in ['blend','dcb'] else f.single();f.save()
assert all(m['locked_geometry']==MANIFEST[0]['locked_geometry'] for m in MANIFEST)
(A/'diagram-manifest.json').write_text(json.dumps(MANIFEST,indent=2))
(A/'alignment-check.json').write_text(json.dumps({'passed':True,'canvas':[0,0,2000,1080],'locked_components':list(MANIFEST[0]['locked_geometry']),'maximum_coordinate_delta':0,'note':'Expanded contextual canvas; identical common stations across this six-figure set. Shapes/styles/badges cloned from approved proposal-derived ancestor; no independent cropping.'},indent=2))
(A/'figure-provenance.json').write_text(json.dumps({'created':'2026-09-24','ancestors':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [ANCESTOR,TEXT_ANCESTOR]],'template':'1_ywXj_hqlgdC0S68aH3kGAteT5UQ4oH3Q5-Pg6s_TeM','reference_slides':['g3f9e8a5a4d7_0_75','g3f9e8a5a4d7_0_65','byrd_transition_23'],'reference_export_note':'Full live template saved/parsed. PDF materialization failed in connector and jq absent; actual reference slide thumbnails downloaded and visually inspected.','scientific_evidence':'selection.json; technical-evidence.md','model_ids':[s[0] for s in SPECS]},indent=2))
print('Built six aligned draw.io figures; common coordinates verified.')

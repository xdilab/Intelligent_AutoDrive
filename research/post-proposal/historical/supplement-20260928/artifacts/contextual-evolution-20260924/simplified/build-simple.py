"""Presentation overview derived from the verified detailed draw.io generator."""
from pathlib import Path
import json,hashlib,base64,copy,xml.etree.ElementTree as E
from urllib.parse import quote
A=Path(__file__).resolve().parent;P=A.parent
ns={'__file__':str(P/'build-diagrams.py')}
exec(compile((P/'build-diagrams.py').read_text().split('\nfor spec in SPECS:')[0],str(P/'build-diagrams.py'),'exec'),ns)
Base=ns['Figure'];SPECS=ns['SPECS'];manifest=[]
class Simple(Base):
 def common(self):
  super().common()
  if self.num in [88,89]:self.cells['subtitle'].set('value','Combine actor appearance with the surrounding scene' if self.num==88 else 'Let the actor attend to the surrounding scene')
  for id in ['scene-dims','detector-carry','cache-note','eval-note']:
   self.root.remove(self.cells.pop(id))
  labels={'frame':'Frame','candidates':'Candidate boxes','clips':'8-frame clips\ncrops + entire frame','video':'Video\nencoder','text':'Text\nencoder','crop-label':'Crop features','roi-label':'RoI features','geometry':'Box position','phrases':'Label\nphrases','bank-label':'Phrase matrix','out':'Candidate boxes\n+ label scores'}
  for id,label in labels.items():self.cells[id].set('value',label.replace('\n','<br>'))
  for id in [id for id in self.cells if id=='bank' or id.startswith('bank-') and id!='bank-label']:
   self.root.remove(self.cells.pop(id))
  self.vector('bank',590,812,160,62,'purple',3)
  for r in self.routes:
   if r['id']=='text-bank':r['points'][-1]=(590,843)
  for id,y in [('roi-label',360),('roi',405),('scene-label',460),('scene',500),('crop-label',565),('crop',620)]:self.cells[id].find('mxGeometry').set('y',str(y))
  self.cells['features'].set('style',self.cells['features'].get('style')+'strokeColor=none;fillColor=none;')
  self.text('footer','Frozen encoders, trained contextual heads. Detector evaluation uses the same YOLO candidates.',40,1030,1900,35,25)
 def single(self):
  attn=self.num==89
  self.box('head','',865,350,780,340,'gray',extra='fillColor=none;strokeColor=none;')
  self.box('fusion','MLP',900,445,195,95,'red',trained=True,extra='fontSize=29;')
  self.box('visual-plus','+',1125,478,30,30,'gold',extra='ellipse;fontSize=27;spacing=0;')
  self.vector('visual',1180,462,160,62)
  self.text('visual-label','Visual RoI',1180,424,160,35,26,False,'center')
  self.node('decoder','',1380,583,110,170,'container=1;fillColor=none;strokeColor=none;pointerEvents=0;')
  self.node('attention-visual','',1380,583,110,110,'ellipse;fillColor=#dae8fc;strokeColor=#4b88b5;strokeWidth=2;fillOpacity=85;')
  self.node('attention-text','',1380,643,110,110,'ellipse;fillColor=#e1d5e7;strokeColor=#8064a2;strokeWidth=2;fillOpacity=75;')
  self.badge('decoder-badge',1480,555,'fire')
  self.text('decoder-title','Cross-attention',1525,573,220,35,25,True,'center')
  self.vector('joint',1550,637,160,62,'gold')
  self.text('joint-label','Joint RoI',1550,712,160,35,26,False,'center')
  self.box('classifier','Classifier',1740,642,160,52,'red',trained=True,extra='fontSize=25;')
  self.node('visual-fork','',822,490,6,6,'ellipse;fillColor=#111111;strokeColor=#111111;spacing=0;')
  self.edge('features-fork','features','visual-fork',[(800,493),(822,493)],arrow=False)
  self.box('position-merge','C',845,478,30,30,'gold',extra='ellipse;fontSize=23;spacing=0;')
  self.edge('features-merge','visual-fork','position-merge',[(828,493),(845,493)],arrow=False)
  self.edge('features-head','position-merge','fusion',[(875,493),(900,493)])
  self.edge('geometry-head','geometry','position-merge',[(785,702),(860,702),(860,508)])
  self.edge('fusion-plus','fusion','visual-plus',[(1095,493),(1125,493)])
  self.edge('visual-skip','visual-fork','visual-plus',[(825,490),(825,409),(1140,409),(1140,478)],'Crop only')
  self.edge('fusion-visual','visual-plus','visual',[(1155,493),(1180,493)])
  self.edge('visual-language','visual','decoder',[(1340,493),(1435,493),(1435,583)])
  self.edge('attention-joint','decoder','joint',[(1481.1,668),(1550,668)])
  self.edge('joint-classifier','joint','classifier',[(1710,668),(1740,668)])
  self.edge('head-out','classifier','out',[(1820,642),(1820,552)])
  self.box('text-adapter','MLP',900,795,195,95,'red',trained=True,extra='fontSize=29;')
  self.box('text-plus','+',1125,828,30,30,'gold',extra='ellipse;fontSize=27;spacing=0;')
  self.vector('adapted',1180,812,160,62,'purple',3)
  self.text('adapted-label','Adapted phrases',1150,881,230,35,25,False,'center')
  self.node('text-fork','',822,840,6,6,'ellipse;fillColor=#111111;strokeColor=#111111;spacing=0;')
  self.edge('bank-fork','bank','text-fork',[(750,843),(822,843)],arrow=False)
  self.edge('bank-adapter','text-fork','text-adapter',[(828,843),(900,843)])
  self.edge('adapter-plus','text-adapter','text-plus',[(1095,843),(1125,843)])
  self.edge('text-skip','text-fork','text-plus',[(825,840),(825,759),(1140,759),(1140,828)],'Phrases')
  self.edge('adapter-phrases','text-plus','adapted',[(1155,843),(1180,843)])
  self.edge('phrases-head','adapted','decoder',[(1340,843),(1435,843),(1435,753)])
  if self.mode=='none':
   self.box('training','Training: focal classification on all 184 labels',865,935,1095,70,'gold',extra='fontSize=27;')
  else:
   self.box('training','Training: focal + contrastive · '+('86 frozen triplet phrases' if self.mode=='triplet' else 'all 184 adapted phrases'),865,935,1095,70,'purple',extra='fontSize=25;')
   if self.mode=='triplet':
    self.vector('contrastive-target',1180,700,160,62,'purple',3)
    self.text('contrastive-target-label','Frozen 86',1180,770,160,28,23,False,'center')
    self.edge('contrastive-loss','visual','contrastive-target',[(1260,524),(1260,700)],'Contrastive loss',dashed=True)
   else:
    self.edge('contrastive-loss','visual','adapted',[(1260,524),(1260,812)],'Contrastive loss',dashed=True)
   self.root[-1].set('style',self.root[-1].get('style')+'startArrow=block;startFill=1;strokeColor=#8064a2;')
  self.schedule=['frame-yolo','yolo-boxes','clips-video','video-features','phrases-text','text-bank','features-fork','features-merge','geometry-head','features-head','fusion-plus','visual-skip','fusion-visual','bank-fork','bank-adapter','adapter-plus','text-skip','adapter-phrases','visual-language','phrases-head','attention-joint','joint-classifier','boxes-out','head-out']
 def blend(self):
  dcb=self.mode=='dcb';loss='DCB' if dcb else 'Focal'
  self.box('head','',845,350,520,540,'gray',extra='fillColor=none;strokeColor=none;')
  self.box('expert-a','',890,380,450,210,'red',trained=True)
  self.text('expert-a-title','Expert A · attention head',905,395,420,40,29,True,'center')
  self.text('expert-a-loss',loss+' classification',905,502,420,33,27,False,'center')
  self.text('expert-a-note','No contrastive loss',905,548,420,30,25,False,'center')
  self.box('expert-b','',890,605,450,305,'red',trained=True)
  self.text('expert-b-title','Expert B · attention head',905,620,420,40,29,True,'center')
  self.text('expert-b-loss',loss+' classification',905,727,420,33,27,False,'center')
  for key,y in [('a',445),('b',670)]:
   self.box('expert-'+key+'-scene','Scene attention',905,y,175,45,'purple',extra='fontSize=22;')
   self.box('expert-'+key+'-mlp','MLP',1170,y,145,45,'red',extra='fontSize=25;')
   self.box('expert-'+key+'-join','C',1110,y+8,30,30,'gold',extra='ellipse;fontSize=23;spacing=0;')
   self.edge('expert-'+key+'-summary','expert-'+key+'-scene','expert-'+key+'-join',[(1080,y+23),(1110,y+23)])
   self.edge('expert-'+key+'-concat','expert-'+key+'-join','expert-'+key+'-mlp',[(1140,y+23),(1170,y+23)])
  self.box('extra-contrastive','',910,775,410,120,'purple')
  self.text('extra-label','All-184 contrastive loss · training',915,780,400,30,23,True,'center')
  self.vector('expert-visual',925,845,130,40)
  self.vector('expert-text',1140,845,130,40,'purple',3)
  self.edge('extra-alignment','expert-visual','expert-text',[(1055,865),(1140,865)],dashed=True)
  self.root[-1].set('style',self.root[-1].get('style')+'startArrow=block;startFill=1;')
  self.text('extra-legend','Visual RoI',920,815,145,27,23,False,'center')
  self.text('extra-text','Adapted phrases',1110,815,185,27,23,False,'center')
  self.box('blend','Blend\ncompositions',1440,470,230,155,'gold',extra='fontSize=29;')
  self.node('shared-inputs','',860,360,4,495,'fillColor=#111111;strokeColor=#111111;')
  self.node('position-merge','',817,472,6,6,'ellipse;fillColor=#111111;strokeColor=#111111;')
  self.edge('features-merge','features','position-merge',[(800,475),(817,475)],arrow=False)
  self.edge('features-head','position-merge','shared-inputs',[(823,475),(860,475)])
  self.edge('geometry-head','geometry','position-merge',[(785,702),(820,702),(820,478)])
  self.edge('phrases-head','bank','shared-inputs',[(750,843),(825,843),(825,830),(860,830)])
  self.edge('inputs-a','shared-inputs','expert-a',[(864,460),(890,460)])
  self.edge('inputs-b','shared-inputs','expert-b',[(864,700),(890,700)])
  self.edge('a-blend','expert-a','blend',[(1340,505),(1440,505)])
  self.edge('b-blend','expert-b','blend',[(1340,675),(1400,675),(1400,585),(1440,585)])
  self.edge('primitive-carry','expert-a','out',[(1340,430),(1395,430),(1395,335),(1710,335),(1710,475),(1740,475)])
  self.text('primitive-label','Keep A’s primitive scores',1385,292,400,35,25,False,'center')
  self.edge('blend-out','blend','out',[(1670,545),(1705,545),(1705,525),(1740,525)])
  self.text('blend-note','Same architecture, different training.',885,921,1050,35,27)
  self.text('selection-note','Blend weight selected on development data.',885,973,1070,35,25)
  self.schedule=['frame-yolo','yolo-boxes','clips-video','video-features','phrases-text','text-bank','features-merge','geometry-head','features-head','phrases-head','inputs-a','expert-a-summary','expert-a-concat','inputs-b','expert-b-summary','expert-b-concat','a-blend','b-blend','primitive-carry','boxes-out','blend-out']
 def extraction(self):
  # Widen only the shared evidence station; every architecture keeps the same transform.
  self.graph.set('pageWidth','2320')
  self.cells['registration_frame'].find('mxGeometry').set('width','2320')
  for id,c in self.cells.items():
   if c.get('parent')!='1' or id in ['registration_frame','candidates']:continue
   g=c.find('mxGeometry');x=float(g.get('x',0))
   if x>=500:g.set('x',str(x+320))
  for r in self.routes:
   if r['id']=='yolo-boxes':continue
   pts=[]
   for j,(x,y) in enumerate(r['points']):
    fixed=r['id']=='boxes-out' and j==0
    pts.append((x+320 if x>=500 and not fixed else x,y))
   r['points']=pts
   edge=next(c for c in self.root if c.get('id')==r['id']);arr=edge.find('mxGeometry/Array')
   for pt,(x,y) in zip(arr,pts[1:-1]):pt.set('x',str(x));pt.set('y',str(y))
  for id in ['clips','video','video-badge','encoder-name','clips-video','video-features']:
   for c in list(self.root):
    if c.get('id')==id:self.root.remove(c)
   self.cells.pop(id,None)
  self.routes=[r for r in self.routes if r['id'] not in ['clips-video','video-features']]
  self.badges.remove('video-badge')
  # Reuse the proposal's verified ROAD-Waymo bus and fixed 2x padded crop window.
  prov=json.loads((P.parent/'stage6-worked-example/provenance.json').read_text())
  frames=Path('/data/datasets/ROAD_plusplus/rgb-images')/prov['source_example']['video']
  from PIL import Image
  iw,ih=Image.open(frames/'00058.jpg').size
  b=prov['source_example']['box'];crop=prov['crop_xyxy_px']
  def photo(id,fid,x,y,w,h,cropped=False,parent='1'):
   data='data:image/jpeg;base64,'+base64.b64encode((frames/f'{fid:05d}.jpg').read_bytes()).decode()
   if cropped:
    x1,y1,x2,y2=crop
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{x2-x1}" height="{y2-y1}" viewBox="{x1} {y1} {x2-x1} {y2-y1}" preserveAspectRatio="xMidYMid meet"><image href="{data}" width="{iw}" height="{ih}"/></svg>'
   else:
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="300" height="200" viewBox="0 0 {iw} {ih}" preserveAspectRatio="xMidYMid meet"><image href="{data}" width="{iw}" height="{ih}"/></svg>'
   data='data:image/svg+xml,'+quote(svg,safe='')
   self.node(id,'',x,y,w,h,'shape=image;imageAspect=0;image='+data+';strokeColor=none;',parent)
  # Frame/candidate photographs are actual examples; overlaid boxes are illustrative GT.
  for id in ['frame','candidates']:
   self.cells[id].set('value','');self.cells[id].set('style',self.cells[id].get('style')+'container=1;pointerEvents=0;fillColor=none;strokeColor=none;')
  for id in ['frame','candidates']:
   self.cells[id].find('mxGeometry').attrib.update(y='170',width='170',height=str(340/3))
  photo('source-photo',58,0,0,170,340/3,parent='frame')
  photo('candidate-photo',58,0,0,170,340/3,parent='candidates')
  for r in self.routes:
   if r['id']=='boxes-out':r['points'][0]=(715,227)
  for id,bb,col,dash in [('object-box',b,'#eaa400',''),('padded-box',[crop[0]/iw,crop[1]/ih,crop[2]/iw,crop[3]/ih],'#0e8a7d','dashed=1;')]:
   self.node(id,'',bb[0]*170,bb[1]*340/3,(bb[2]-bb[0])*170,(bb[3]-bb[1])*340/3,'fillColor=none;strokeColor='+col+';strokeWidth=2;'+dash,'candidates')
  self.text('frame-label','Frame t',40,290,170,30,23,False,'center')
  self.text('candidates-label','YOLO candidate boxes',505,135,300,30,23,False,'center')
  # Filmstrip stacks use real frames from the same clip, not a second illustrative scene.
  for id,y,iscrop,label in [('clips',580,True,'Crops'),('entire',375,False,'Entire frames')]:
   tw,th=((crop[2]-crop[0])*170/iw,(crop[3]-crop[1])*170/iw) if iscrop else (170,340/3)
   x=174-(tw+18)/2
   self.node(id,'',x,y,tw+18,th+16,'container=1;fillColor=none;strokeColor=none;pointerEvents=0;')
   for k,fid in enumerate([55,58,62]):photo(id+'-'+str(fid),fid,k*9,k*8,tw,th,iscrop,parent=id)
   self.text(id+'-label',label,70,y+th+25 if iscrop else y-40,208,30,23,False,'center')
  self.inherited('video','c10','Video\nencoder',300,575,190,110)
  self.inherited('scene-video','c10','Video\nencoder',300,365,190,110)
  self.text('encoder-name','InternVideo2-CLIP-S · shared weights',300,715,530,30,23,False,'center')
  self.vector('spatial-map',520,365,170,340/3,rows=4)
  self.node('map-roi','',b[0]*170,b[1]*340/3,(b[2]-b[0])*170,(b[3]-b[1])*340/3,'rounded=0;fillColor=none;strokeColor=#eaa400;strokeWidth=3;','spatial-map')
  self.text('map-label','Map at t',520,335,170,28,23,False,'center')
  self.box('crop-pool','Pool',730,610,125,50,'gold',extra='fontSize=23;')
  self.box('roi-align','RoIAlign',730,390,125,55,'gold',extra='fontSize=23;')
  self.box('scene-pool','Pool',730,495,125,50,'gold',extra='fontSize=23;')
  self.edge('boxes-roialign','candidates','roi-align',[(630,850/3),(630,315),(792,315),(792,390)])
  self.edge('clip-entire','frame','entire',[(40,227),(20,227),(20,420),(80,420)])
  self.edge('entire-crops','entire','clips',[(174,1513/3),(174,580)])
  self.edge('clips-video','clips','video',[(174+((crop[2]-crop[0])*170/iw+18)/2,635),(300,635)])
  self.edge('crop-encode-pool','video','crop-pool',[(490,635),(730,635)])
  self.edge('crop-pool-feature','crop-pool','crop',[(855,635),(875,635)])
  self.edge('entire-video','entire','scene-video',[(268,420),(300,420)])
  self.edge('scene-encode-map','scene-video','spatial-map',[(490,420),(520,420)])
  self.edge('map-roialign','spatial-map','roi-align',[(690,420),(730,420)])
  self.edge('roi-feature','roi-align','roi',[(855,420),(875,420)])
  self.edge('map-pool','spatial-map','scene-pool',[(605,1435/3),(605,522),(730,522)])
  self.edge('pool-scene','scene-pool','scene',[(855,522),(875,522)])
  # The same fixed candidate window selects the actor from the entire-frame clip.
  for id,bb,col,dash in [('clip-object-box',b,'#eaa400',''),('clip-padded-box',[crop[0]/iw,crop[1]/ih,crop[2]/iw,crop[3]/ih],'#0e8a7d','dashed=1;')]:
   self.node(id,'',18+bb[0]*170,16+bb[1]*340/3,(bb[2]-bb[0])*170,(bb[3]-bb[1])*340/3,'fillColor=none;strokeColor='+col+';strokeWidth=2;'+dash,'entire')
  self.node('feature-collector','',1117,405,4,246,'fillColor=#111111;strokeColor=#111111;')
  for id,y in [('roi',420),('scene',522),('crop',635)]:self.edge(id+'-collect',id,'feature-collector',[(1105,y),(1117,y)],arrow=False)
  for r in self.routes:
   if r['source']=='features':
    r['source']='feature-collector';r['points'][0]=(1121,r['points'][0][1])
    edge=next(c for c in self.root if c.get('id')==r['id']);edge.set('source','feature-collector')
  self.cells['footer'].set('value','ROAD-Waymo illustrative GT box · fixed window across 8 frames · gold: actor RoI · teal: padded crop window')
  self.cells['footer'].find('mxGeometry').set('width','2220')
  prefix=['frame-yolo','yolo-boxes','boxes-roialign','clip-entire','entire-video','scene-encode-map','map-roialign','roi-feature','map-pool','pool-scene','entire-crops','clips-video','crop-encode-pool','crop-pool-feature','roi-collect','scene-collect','crop-collect']
  self.schedule=prefix+[id for id in self.schedule if id not in prefix+['video-features']]
  # A fixed 200px lane separates scene summarization from the residual MLP.
  # Preserve every shared station across the evolution set, including blends.
  self.graph.set('pageWidth','2520')
  self.cells['registration_frame'].find('mxGeometry').set('width','2520')
  for id,c in self.cells.items():
   if c.get('parent')!='1' or id=='registration_frame':continue
   g=c.find('mxGeometry');x=float(g.get('x',0))
   if x>=1117:g.set('x',str(x+200))
  for r in self.routes:
   r['points']=[(x+200 if x>=1117 else x,y) for x,y in r['points']]
   edge=next(c for c in self.root if c.get('id')==r['id'])
   arr=edge.find('mxGeometry/Array')
   for pt,(x,y) in zip(arr,r['points'][1:-1]):pt.set('x',str(x));pt.set('y',str(y))
  if self.mode not in ['blend','dcb']:
   attn=self.num==89
   self.box('scene-summary','Scene\nattention' if attn else 'Mean pool',1135,492,160,60,'red' if attn else 'gold',trained=attn,extra='fontSize=24;')
   for c in list(self.root):
    if c.get('id')=='scene-collect':self.root.remove(c)
   self.routes=[r for r in self.routes if r['id']!='scene-collect']
   self.edge('scene-summary-in','scene','scene-summary',[(1105,522),(1135,522)])
   self.edge('scene-summary-out','scene-summary','feature-collector',[(1295,522),(1317,522)],arrow=False)
   if attn:self.text('scene-query-note','Query: crop + position',1115,561,200,48,20,False,'center')
   i=self.schedule.index('scene-collect');self.schedule[i:i+1]=['scene-summary-in','scene-summary-out']
  # Re-pin anchors after resizing photo and feature stations; animation endpoints agree.
  for r in self.routes:
   edge=next(c for c in self.root if c.get('id')==r['id'])
   extra=''
   for side,key,point in [('exit','source',r['points'][0]),('entry','target',r['points'][-1])]:
    g=self.cells[r[key]].find('mxGeometry');x,y,w,h=[float(g.get(k,0)) for k in ['x','y','width','height']]
    xx,yy=(point[0]-x)/w,(point[1]-y)/h
    if r[key]=='yolo':xx,yy=(.5,1) if side=='exit' else (.5,0)
    extra+=f'{side}X={xx};{side}Y={yy};'
   edge.set('style',edge.get('style')+extra)

  (A/'image-provenance.json').write_text(json.dumps({'source':str(P.parent/'stage6-worked-example/provenance.json'),'example':prov['source_example'],'frames':[55,58,62],'frame_sha256':{str(i):prov['frame_sha256'][str(i)] for i in [55,58,62]},'image_tile_aspect_ratio':'Entire frames and map:170x113.333; crop tiles use the actual padded-window dimensions at the same display scale, preserving native proportions','scope':'Illustrative GT box, not a measured YOLO prediction; same fixed padded crop window. Feature map is schematic, not a measured activation.'},indent=2))
 def save(self):
  self.extraction()
  groups={'features':['crop-label','crop','roi-label','roi','scene-label','scene'],
    'decoder':['attention-visual','attention-text'],
    'head':['fusion','fusion-note','visual-plus','visual','visual-label','decoder','expert-a','expert-b','shared-inputs'],
    'training':['align-visual','align-text','training-caption','training-scope'],
    'extra-contrastive':['extra-label','expert-visual','expert-text','extra-legend','extra-text','extra-training'],
    'expert-a':['expert-a-title','expert-a-loss','expert-a-note','expert-a-scene','expert-a-join','expert-a-mlp'],
    'expert-b':['expert-b-title','expert-b-loss','extra-contrastive','expert-b-scene','expert-b-join','expert-b-mlp']}
  # Convert absolute coordinates after all parent positions have been captured.
  positions={id:{k:float(c.find('mxGeometry').get(k,0)) for k in ['x','y','width','height']} for id,c in self.cells.items()}
  for parent,children in groups.items():
   if parent not in self.cells:continue
   pc=self.cells[parent];pc.set('style',pc.get('style')+'container=1;pointerEvents=0;')
   for child in children:
    if child not in self.cells:continue
    c=self.cells[child];g=c.find('mxGeometry');c.set('parent',parent)
    for axis in ['x','y']:g.set(axis,str(positions[child][axis]-positions[parent][axis]))
  for c in self.root:
   if c.get('parent')=='1' and c.get('id')!='registration_frame':c.set('parent','registration_frame')
  E.indent(self.tree);path=A/(self.slug+'.drawio');self.tree.write(path,encoding='utf-8',xml_declaration=True)
  locked=['frame','yolo','candidates','clips','video','features','crop','roi','scene','geometry','phrases','text','bank','out','scene-video','spatial-map','roi-align','feature-collector','crop-pool','scene-pool','entire','registration_frame']
  manifest.append({'model_id':self.num,'slug':self.slug,'title':self.title,'mode':self.mode,'locked_geometry':{k:positions[k] for k in locked},'routes':self.routes,'flow_order':self.schedule,'intentional_badge_overlaps':self.badges})
for spec in SPECS:
 f=Simple(spec);f.common();f.blend() if f.mode in ['blend','dcb'] else f.single();f.save()
assert all(m['locked_geometry']==manifest[0]['locked_geometry'] for m in manifest)
(A/'diagram-manifest.json').write_text(json.dumps(manifest,indent=2))
(A/'alignment-check.json').write_text(json.dumps({'passed':True,'locked_components':list(manifest[0]['locked_geometry']),'maximum_coordinate_delta':0,'canvas':[2520,1080]},indent=2))
(A/'figure-provenance.json').write_text(json.dumps({'date':'2026-09-25','tier':'simplified presentation overview','parent_generator':str(P/'build-diagrams.py'),'sha256':hashlib.sha256((P/'build-diagrams.py').read_bytes()).hexdigest(),'technical_source':str(P/'technical-evidence.md'),'exceptions':'Tensor dimensions and classifier concatenation are in companion details; crop-only/text residuals remain visible. Original architectures unchanged.'},indent=2))
print('Six simplified figures built; all common stations match.')

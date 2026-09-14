from pathlib import Path
import xml.etree.ElementTree as E
import re,copy
OUT=Path('/data/repos/wiki/artifacts/proposal-figures/horizontal')
class Fig:
 def __init__(self,p):
  self.t=E.parse(p);self.r=self.t.find('.//root');self.d={c.get('id'):c for c in self.r};self.orig=copy.deepcopy(self.d)
 def pos(self,id,x,y,w,h,parent='1'):
  c=self.d[id];c.set('parent',parent);g=c.find('mxGeometry');g.attrib={'x':str(x),'y':str(y),'width':str(w),'height':str(h),'as':'geometry'}
 def sty(self,id,**kw):
  c=self.d[id];s=c.get('style','')
  for k,v in kw.items():s=re.sub(r'(?:^|;)'+k+r'=[^;]*','',s).strip(';')+';'+k+'='+str(v)+';'
  c.set('style',s)
 def edge(self,id,ex=1,ey=.5,ix=0,iy=.5,points=None):
  self.sty(id,exitX=ex,exitY=ey,entryX=ix,entryY=iy,rounded=1,jettySize=16)
  c=self.d[id];g=c.find('mxGeometry');g.clear();g.attrib={'relative':'1','as':'geometry'}
  if points:
   a=E.SubElement(g,'Array',{'as':'points'})
   for x,y in points:E.SubElement(a,'mxPoint',x=str(x),y=str(y))
 def remove(self,id):self.r.remove(self.d[id]);del self.d[id]
 def save(self,name,w,h):
  # Keep original symbols, palette, and components. Titles belong to the native slide.
  for c in self.d.values():
   if c.get('vertex') and c.get('value'):
    self.sty(c.get('id'),fontSize=20)
    c.set('value',re.sub(r'font-size:\s*\d+px','font-size:19px',c.get('value')))
  self.t.find('.//mxGraphModel').set('pageWidth',str(w));self.t.find('.//mxGraphModel').set('pageHeight',str(h))
  self.t.write(OUT/(name+'.drawio'),encoding='utf-8',xml_declaration=True)

f=Fig('/data/repos/wiki/artifacts/proposal-figures/paper_stage0_retinanet.drawio')
for id in ['c2','bar1']:f.remove(id)
# Flatten the original ego group so all coordinates are explicit.
f.pos('1DZrssdnUPbAtZ1wRXhg-1',0,0,1,1)
f.pos('c3',10,280,165,75);f.pos('c5',220,257,170,120)
for cube,label,y in [('c11','c12',120),('c9','c10',220),('c7','c8',320)]:f.pos(cube,455,y,145,50);f.pos(label,405,y+54,240,28)
for cube,label,y in [('c13','c14',120),('c15','c16',220),('c17','c18',320)]:f.pos(cube,850,y,150,50);f.pos(label,810,y+54,245,28)
f.pos('c19',735,232,26,26);f.pos('c20',735,332,26,26)
for id,y in [('c21',121),('c22',221),('c23',321)]:f.pos(id,1110,y,150,48)
f.pos('nms1',1360,220,175,90);f.pos('out1',1600,220,230,100);f.pos('out_note',1555,330,320,50)
for id,x,w in [('c37',690,160),('c60',900,140),('c61',1090,150),('c62',1290,110),('c63',1450,155)]:f.pos(id,x,15,w,65)
f.pos('c38',680,82,315,25);f.pos('ego_tag',1610,20,215,60)
f.pos('c39',445,450,985,145)
f.pos('c40',0,147,985,25,'c39')
# Original class and box subnet components, now side by side.
for ids,base,label in [(['c41','c42','c43','c44','c45','c55','c57'],0,'class subnet'),(['c46','c47','c48','c49','c50','c56','c58'],495,'box subnet')]:
 title,inp,conv,outconv,outtensor,indim,outdim=ids
 for id,x,y,w,h in [(title,base+10,5,180,25),(inp,base+10,42,65,38),(conv,base+110,32,110,60),(outconv,base+250,32,95,60),(outtensor,base+380,42,60,38),(indim,base+0,99,230,28),(outdim,base+235,99,245,28)]:f.pos(id,x,y,w,h,'c39')
f.pos('c68',10,420,420,225)
f.d['c68'].set('value','LEGEND<br>T = 8 frames; C-levels carry T/2<br>Input: H × W = 600 × 840; 3 = RGB<br>s = stride 8, 16, 32; A = 9 anchors<br>184 scores: 1 agentness, 10 agent,<br>22 action, 16 location, 49 duplex, 86 triplet<br>4 = box offsets; ego branch unused')
f.edge('c6');f.edge('c24',points=[(420,317),(420,345)])
f.edge('c25',0,.5,0,.5,[(420,345),(420,245)]);f.edge('c26',0,.5,0,.5,[(420,245),(420,145)])
for id in ['c27','c28','c30','c31','c33','c34','c35','c36','ne4','c64','c65','c66','c67','c51','c52','c53','c54']:f.edge(id)
f.edge('c29',0,.5,.5,0,[(790,145),(748,145)]);f.edge('c32',0,.5,.5,0,[(790,245),(748,245)])
f.edge('ne1',1,.5,0,.2,[(1300,145),(1300,238)])
f.edge('ne2',1,.5,0,.5,[(1320,245),(1320,265)])
f.edge('ne3',1,.5,0,.8,[(1340,345),(1340,292)])
f.edge('ne5',.5,0,0,.5,[(527,47)])
f.edge('c59',.5,1,1,0,[(1185,420),(1430,420)])
# Place output tensors after their conv, retaining original inset topology (unwired visual tensors).
f.save('retinanet-horizontal',1840,650)

f=Fig('/home/brandon/Downloads/evolution_diagrams/paper_stage0_yolov8x.drawio')
for id in ['c2','c3','c64']:f.remove(id)
f.pos('c4',10,245,165,75);f.pos('c5',220,222,170,120)
for cube,label,y in [('c11','c12',65),('c9','c10',170),('c7','c8',280)]:f.pos(cube,455,y,145,50);f.pos(label,405,y+54,245,48 if cube=='c11' else 28)
f.pos('c16',700,65,155,265)
for cube,label,y in [('c20','c21',65),('c22','c23',170),('c24','c25',280)]:f.pos(cube,950,y,150,50);f.pos(label,910,y+54,245,28)
for id,y in [('c29',66),('c30',171),('c31',281)]:f.pos(id,1200,y,150,48)
f.pos('c35',1430,170,140,75);f.pos('c36',1630,155,210,105)
f.pos('c42',445,420,985,150)
f.pos('c44',0,152,985,25,'c42')
for ids,base in [(['c43','c45','c46','c47','c48','c49','c50'],0),(['c51','c52','c53','c54','c55','c56','c57'],495)]:
 title,inp,conv,outconv,outtensor,indim,outdim=ids
 for id,x,y,w,h in [(title,base+10,5,180,25),(inp,base+10,42,65,38),(conv,base+110,32,110,60),(outconv,base+250,32,95,60),(outtensor,base+380,42,60,38),(indim,base,99,230,28),(outdim,base+235,99,245,28)]:f.pos(id,x,y,w,h,'c42')
f.pos('c58',10,385,420,210)
f.d['c58'].set('value','LEGEND<br>H × W = input pixels (imgsz 1280)<br>s = stride 8, 16, 32<br>Cₗ = channels: 320 / 640 / 640<br>4 = box sides; 16 = DFL bins per side<br>n = boxes kept by NMS; 3 = RGB')
f.pos('c41',1460,360,380,205)
f.d['c41'].set('value','C = 10 ROAD-Waymo agent classes<br>Anchor-free: one prediction per cell<br>Box = 4 sides × 16 DFL bins')
for id in ['c6','c17','c18','c19','c26','c27','c28','c32','c33','c34','c40','c59','c60','c61','c62']:f.edge(id)
f.edge('c13',points=[(420,282),(420,305)])
f.edge('c14',0,.5,0,.5,[(420,305),(420,195)]);f.edge('c15',0,.5,0,.5,[(420,195),(420,90)])
for id,iy in [('c17',.095),('c18',.49),('c19',.905)]:f.edge(id,1,.5,0,iy)
for id,ey in [('c26',.095),('c27',.49),('c28',.905)]:f.edge(id,1,ey,0,.5)
f.edge('c37',1,.5,0,.2,[(1380,90),(1380,185)])
f.edge('c38',1,.5,0,.5,[(1395,195),(1395,207.5)])
f.edge('c39',1,.5,0,.8,[(1410,305),(1410,230)])
f.edge('c63',.5,1,1,0,[(1275,390),(1430,390)])
f.save('yolo-horizontal',1850,610)

f=Fig('/data/repos/wiki/artifacts/proposal-figures/internvideo2-proposal.drawio')
for id in ['c2','c3','c30']:f.remove(id)
for id,x,y,w,h in [('c4',10,100,190,85),('c5',250,107,210,72),('c6',510,100,215,85),('c7',775,80,235,125),('c11',1090,100,230,85),('c12',1380,100,200,85),('c15',1060,270,270,85),('c16',1410,270,230,85),('c19',1650,280,200,55),('c20',1630,425,225,100),('c22',10,430,210,85),('c23',300,420,250,110),('c24',650,435,220,80),('c28',900,440,480,75),('c29',10,560,1840,100)]:f.pos(id,x,y,w,h)
f.d['c29'].set('value','LEGEND: tokens = 8 frames × 16 × 16 patches; 3 = RGB; 224 = crop pixels; 77 = max text tokens; 184 = class phrases<br>1024 = vision token dimension; 512 = CLIP space. Green = inputs; yellow = operations; dashed teal = frozen towers; orange = our feature tap.<br>All encoder weights are frozen. The text tower runs once, offline; the crop feature feeds proposed Stage 4.')
for id in ['c8','c9','c10','c13','c14','c18','c25','c26']:f.edge(id)
f.edge('c17',.5,1,0,.5,[(893,312.5)])
f.edge('c21',1,.5,.5,0,[(1742.5,142.5)])
f.edge('c27',1,.5,0,.5,[(895,475),(895,400),(1580,400),(1580,475)])
f.save('internvideo-horizontal',1870,670)
# Final routing pass: keep wires clear of dimension labels and retain explicit tensor outputs.
for name in ['retinanet','yolo','internvideo']:
 f=Fig(OUT/(name+'-horizontal.drawio'))
 if name=='retinanet':
  f.edge('c25',0,.5,0,.5,[(395,345),(395,245)])
  f.edge('c26',0,.5,0,.5,[(395,245),(395,145)])
  f.edge('c32',0,.5,.5,0,[(800,245),(800,300),(748,300)])
  f.pos('c63',1450,10,155,80)
  f.pos('nms1',1360,215,175,100)
  pairs=[('c44','c45'),('c49','c50')]
 elif name=='yolo':
  f.edge('c14',0,.5,0,.5,[(395,305),(395,195)])
  f.edge('c15',0,.5,0,.5,[(395,195),(395,90)])
  pairs=[('c47','c48'),('c54','c55')]
 else:
  f.pos('c19',1390,360,270,32);pairs=[]
 for a,b in pairs:
  id='output_'+a
  c=E.SubElement(f.r,'mxCell',id=id,parent=f.d[a].get('parent'),source=a,target=b,edge='1',style='edgeStyle=orthogonalEdgeStyle;rounded=1;jettySize=16;html=1;strokeColor=#111111;endArrow=block;endFill=1;exitX=1;exitY=0.5;entryX=0;entryY=0.5;')
  E.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'});f.d[id]=c
 f.save(name+'-horizontal',1880,650)
for name in ['retinanet','yolo']:
 f=Fig(OUT/(name+'-horizontal.drawio'))
 if name=='retinanet':f.pos('c3',10,307.5,165,75);f.pos('c5',220,285,170,120);f.edge('c24')
 else:f.pos('c4',10,267.5,165,75);f.pos('c5',220,245,170,120);f.edge('c13')
 f.save(name+'-horizontal',1880,650)

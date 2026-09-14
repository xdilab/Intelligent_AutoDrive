from pathlib import Path
import xml.etree.ElementTree as E
import base64

out=Path(__file__).parent
doc=E.Element('mxfile',host='app.diagrams.net');d=E.SubElement(doc,'diagram',name='One road-user example');m=E.SubElement(d,'mxGraphModel',page='0');r=E.SubElement(m,'root')
E.SubElement(r,'mxCell',id='0');E.SubElement(r,'mxCell',id='1',parent='0')
base='whiteSpace=wrap;html=1;fontFamily=Helvetica;fontSize=24;fontColor=#203040;spacing=5;'
def box(i,v,x,y,w,h,st=''):
 c=E.SubElement(r,'mxCell',id=i,value=v,vertex='1',parent='1',style=base+st)
 E.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
 return c
def text(i,v,x,y,w,h):return box(i,v,x,y,w,h,'text;fillColor=none;strokeColor=none;')
def edge(i,s,t,st=''):
 c=E.SubElement(r,'mxCell',id=i,source=s,target=t,edge='1',parent='1',style='edgeStyle=orthogonalEdgeStyle;rounded=0;strokeWidth=2;strokeColor=#111111;endArrow=block;endFill=1;exitX=1;exitY=0.5;entryX=0;entryY=0.5;'+st)
 E.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'})
text('h1','One box at frame t',0,0,350,45)
text('h2','Evidence for this box',400,0,300,45)
text('h3','Learned composition',770,0,230,45)
text('h4','Composition scores',1055,0,325,45)
raw=base64.b64encode(Path('/data/datasets/road_waymo/rgb-images/train_00407/00001.jpg').read_bytes()).decode()
box('photo','',0,85,350,350*2/3,'shape=image;imageAspect=1;image=data:image/jpeg,'+raw+';')
# Verified annotation b_49. This is an illustrative reference box, not a claimed YOLO result.
box('car-box','',350*.28479405572916666,85+(350*2/3)*.589880753125,350*(.5648957166666667-.28479405572916666),(350*2/3)*(.94812225703125-.589880753125),'fillColor=none;strokeColor=#F6B81A;strokeWidth=3;rounded=0;')
text('photo-note','ROAD-Waymo example',0,325,350,40)
box('scores','Primitive scores · 49<br><br>Car · Stop · VehLane · …',400,85,300,115,'rounded=1;arcSize=6;fillColor=#f5f5f5;strokeColor=#666666;')
box('features','',465,260,170,70,'group;html=1;container=1;')
colors=['#F2C75C','#FFE6A0','#DDAE39']
for row in range(3):
 for col in range(6):
  c=E.SubElement(r,'mxCell',id=f'tile_{row}_{col}',value='',vertex='1',parent='features',style=f'html=1;rounded=0;fillColor={colors[(row+col)%3]};strokeColor=#B18500;strokeWidth=1;')
  E.SubElement(c,'mxGeometry',x=str(col*170/6),y=str(row*70/3),width=str(170/6),height=str(70/3),attrib={'as':'geometry'})
text('feature-label','RoI features · 256',400,215,300,40)
box('mlp','Composition MLP<br><br>Learn scores for<br>combinations',770,150,230,160,'rounded=1;arcSize=6;fillColor=#f8cecc;strokeColor=#b85450;strokeWidth=3;')
box('outputs','Duplex · 49 scores<br>Car-Stop · …<br><br>Triplet · 86 scores<br>Car-Stop-VehLane · …',1055,115,325,225,'rounded=1;arcSize=6;fillColor=#f3f7fb;strokeColor=#004684;')
edge('e1','scores','mlp','entryY=0.25;')
edge('e2','features','mlp','entryY=0.75;')
edge('e3','mlp','outputs')
text('arch','Linear → ReLU → Linear',735,350,300,40)
text('fixed','Boxes and primitive outputs stay fixed',400,402,630,40)
box('legend','',1110,408,32,25,'rounded=0;fillColor=#f8cecc;strokeColor=#b85450;strokeWidth=3;')
text('legend-text','Trained module',1150,400,215,40)
E.ElementTree(doc).write(out/'composition-example.drawio',encoding='utf-8',xml_declaration=True)

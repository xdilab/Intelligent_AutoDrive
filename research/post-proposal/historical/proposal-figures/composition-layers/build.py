from pathlib import Path
import xml.etree.ElementTree as E

out=Path(__file__).parent
doc=E.Element('mxfile',host='app.diagrams.net')
d=E.SubElement(doc,'diagram',name='Composition MLP')
m=E.SubElement(d,'mxGraphModel',page='0',pageWidth='1320',pageHeight='390')
r=E.SubElement(m,'root'); E.SubElement(r,'mxCell',id='0'); E.SubElement(r,'mxCell',id='1',parent='0')
base='whiteSpace=wrap;html=1;fontFamily=Helvetica;fontSize=22;fontColor=#1f2937;spacing=5;'
def box(i,v,x,y,w,h,style=''):
 c=E.SubElement(r,'mxCell',id=i,value=v,style=base+style,vertex='1',parent='1')
 E.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
def label(i,v,x,y,w,h,style=''):
 box(i,v,x,y,w,h,'text;strokeColor=none;fillColor=none;'+style)
def arrow(i,s,t,style=''):
 c=E.SubElement(r,'mxCell',id=i,source=s,target=t,edge='1',parent='1',style='edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor=#111111;strokeWidth=2;endArrow=block;endFill=1;exitX=1;exitY=0.5;entryX=0;entryY=0.5;'+style)
 E.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'})
label('scope','For each candidate box at frame t',0,0,1320,40,'align=left;')
box('module','',455,65,495,220,'rounded=1;arcSize=6;fillColor=#f8cecc;strokeColor=#b85450;strokeWidth=3;')
label('title','Composition MLP',465,75,475,40,'fontStyle=1;')
box('scores','Primitive scores<br>49',0,85,220,85,'rounded=1;fillColor=#f5f5f5;strokeColor=#666666;')
box('features','RoI features<br>256',0,215,220,70,'rounded=0;fillColor=#ffe6cc;strokeColor=#d79b00;')
box('concat','Concatenate<br>305',265,155,150,80,'rounded=0;fillColor=#fff2cc;strokeColor=#d6b656;')
box('fc1','Linear<br>305 → 512',480,155,145,80,'rounded=0;fillColor=#ffffff;strokeColor=#b85450;strokeWidth=3;')
box('relu','ReLU',665,155,100,80,'rounded=0;fillColor=#fff2cc;strokeColor=#d6b656;')
box('fc2','Linear<br>512 → 135',805,155,125,80,'rounded=0;fillColor=#ffffff;strokeColor=#b85450;strokeWidth=3;')
box('sigmoid','Sigmoid',985,155,105,80,'rounded=0;fillColor=#fff2cc;strokeColor=#d6b656;')
box('output','135 scores<br>49 duplex<br>86 triplet',1130,135,190,120,'rounded=1;arcSize=6;fillColor=#f8cecc;strokeColor=#b85450;')
arrow('a1','scores','concat','entryY=0.25;')
arrow('a2','features','concat','entryY=0.75;')
for n,(a,b) in enumerate([('concat','fc1'),('fc1','relu'),('relu','fc2'),('fc2','sigmoid'),('sigmoid','output')],3): arrow('a'+str(n),a,b)
label('agentness','49 includes agentness',0,290,245,35,'fontSize=21;align=left;')
box('trained-key','',470,327,36,25,'rounded=0;fillColor=#ffffff;strokeColor=#b85450;strokeWidth=3;')
label('trained-label','Trained layer',515,320,170,40,'fontSize=21;align=left;')
box('op-key','',740,327,36,25,'rounded=0;fillColor=#fff2cc;strokeColor=#d6b656;')
label('op-label','Fixed operation',785,320,190,40,'fontSize=21;align=left;')
feature=next(c for c in r if c.get('id')=='features')
feature.set('value',''); feature.set('style','group;html=1;container=1;')
colors=['#F2C75C','#FFE6A0','#DDAE39']
for row in range(3):
 for col in range(6):
  tile=E.SubElement(r,'mxCell',id=f'features_roi_feature_{row}_{col}',value='',vertex='1',parent='features',style=f'html=1;rounded=0;fillColor={colors[(row+col)%3]};strokeColor=#B18500;strokeWidth=1;')
  E.SubElement(tile,'mxGeometry',x=str(col*220/6),y=str(row*70/3),width=str(220/6),height=str(70/3),attrib={'as':'geometry'})
label('features_grid_label','RoI features · 256',0,180,220,30)
E.ElementTree(doc).write(out/'composition-layers.drawio',encoding='utf-8',xml_declaration=True)

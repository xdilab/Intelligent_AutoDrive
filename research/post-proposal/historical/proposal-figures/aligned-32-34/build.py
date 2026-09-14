from pathlib import Path
import copy
import xml.etree.ElementTree as E
base=Path('/data/repos/wiki/artifacts/proposal-figures');out=base/'aligned-32-34'
t=E.parse(base/'paper_stage4_phrase_prop-clean-arrow.drawio');r=t.find('.//root')
keep={'0','1','c4','c5','c6','c7','c8','c9','c10','c26','c26_label','time_note','tap_t','all_bracket','all_flow','crop_join','crop_join_to_input','c35','crop_wire_label'}
# Preserve the complete source frame/YOLO/crop/encoder/feature modules and their connecting edges.
for c in list(r):
 i=c.get('id','');parent=c.get('parent')
 if i in keep or i.startswith('time_slice_') or parent=='c26':continue
 if c.get('edge')=='1' and c.get('source') in keep and c.get('target') in keep:continue
 r.remove(c)
base_style='html=1;whiteSpace=wrap;fontFamily=Helvetica;fontSize=18;fontColor=#203040;'
def box(i,v,x,y,w,h,st=''):
 c=E.SubElement(r,'mxCell',id=i,value=v,vertex='1',parent='1',style=base_style+st)
 E.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
 return c
def label(i,v,x,y,w,h):return box(i,v,x,y,w,h,'text;fillColor=none;strokeColor=none;')
def edge(i,s,target,points=None,style=''):
 c=E.SubElement(r,'mxCell',id=i,edge='1',source=s,target=target,parent='1',style='edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#d79b00;strokeWidth=2;endArrow=block;endFill=1;'+style)
 g=E.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'})
 if points:
  a=E.SubElement(g,'Array',attrib={'as':'points'})
  for x,y in points:E.SubElement(a,'mxPoint',x=str(x),y=str(y))
# Explicit temporal input, identical to stage diagrams.
label('crop_dim','1024 dimensions',400,470,180,25)
label('detail_title','Expanded crop input: one fixed window across eight frames',670,190,790,32)
label('legend','Dashed models: frozen',1140,75,340,30)
box('detail','',650,230,820,270,'rounded=1;arcSize=5;fillColor=none;strokeColor=#999999;dashed=1;')
for j in range(8):
 x=675+98*j
 box(f'frame{j}','',x,245,76,64,'rounded=1;fillColor=#f3f7fb;strokeColor=#aabac6;')
 box(f'obj{j}','',x+20+j*2,269,12,18,'rounded=0;fillColor=#607d8b;strokeColor=none;')
 box(f'window{j}','',x+11,254,51,45,'rounded=0;fillColor=none;strokeColor=#d79b00;strokeWidth=2;dashed=1;')
 label(f'time{j}',('t' if j==3 else f't − {3-j}' if j<3 else f't + {j-3}'),x-4,309,84,27)
 box(f'crop{j}','',x,408,76,64,'rounded=1;fillColor=#fff2cc;strokeColor=#d79b00;strokeWidth=2;')
 box(f'cropobj{j}','',x+12+j*3,430,17,23,'rounded=0;fillColor=#607d8b;strokeColor=none;')
box('target_box','',675+98*3+24,267,21,24,'fillColor=none;strokeColor=#004684;strokeWidth=2;')
box('crop_op','Same padded window · crop + resize to 224 × 224',675,353,762,35,'rounded=0;fillColor=#fff2cc;strokeColor=#d79b00;')
edge('box_to_detail','c6','crop_op',[(615,146),(615,370)],'exitX=1;exitY=0.5;entryX=0;entryY=0.5;')
box('frames_bracket','',675,339,762,1,'fillColor=#111111;strokeColor=#111111;')
edge('frames_to_op','frames_bracket','crop_op',None,'exitX=0.5;exitY=1;entryX=0.5;entryY=0;')
edge('op_to_crops','crop_op','crop3',None,'exitX=0.435;exitY=1;entryX=0.5;entryY=0;')
label('detail_footer','Eight resized crops for the same candidate box at t',680,471,752,28)
# Dashed leader identifies the inset as an expansion of the retained crop-input module.
edge('detail_leader','c7','detail',[(67,520),(635,520),(635,480)],'strokeColor=#999999;dashed=1;endArrow=none;exitX=0.5;exitY=1;entryX=0;entryY=0.93;')
t.write(out/'crop-aligned.drawio',encoding='utf-8',xml_declaration=True)

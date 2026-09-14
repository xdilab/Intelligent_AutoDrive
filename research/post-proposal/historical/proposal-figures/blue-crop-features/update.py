from pathlib import Path
import xml.etree.ElementTree as E
base=Path('/data/repos/wiki/artifacts/proposal-figures')
jobs=[('paper_stage4_phrase_prop-clean-arrow','c26','crop features',14),('box-crop-explainer','target_feature','crop features',24),('horizontal/internvideo-horizontal','c16','crop features · 1024',20)]
for name,ident,label,fs in jobs:
 p=base/(name+'.drawio');(base/'blue-crop-features'/(p.stem+'-before.drawio')).write_bytes(p.read_bytes())
 t=E.parse(p);r=t.find('.//root');c=next(c for c in r if c.get('id')==ident);g=c.find('mxGeometry');w=float(g.get('width'));h=float(g.get('height'))
 c.set('value','');c.set('style','group;html=1;container=1;')
 colors=['#85B6E0','#D6E8FA','#4F91C6']
 for row in range(3):
  for col in range(6):
   v=E.SubElement(r,'mxCell',id=f'{ident}_blue_{row}_{col}',value='',vertex='1',parent=ident,style=f'html=1;rounded=0;fillColor={colors[(row+col)%3]};strokeColor=#3977A8;strokeWidth=1;')
   E.SubElement(v,'mxGeometry',x=str(col*w/6),y=str(row*h/3),width=str(w/6),height=str(h/3),attrib={'as':'geometry'})
 v=E.SubElement(r,'mxCell',id=ident+'_label',value=label,vertex='1',parent='1',style=f'text;html=1;whiteSpace=wrap;fillColor=none;strokeColor=none;fontFamily=Helvetica;fontSize={fs};fontColor=#203040;align=center;verticalAlign=middle;')
 E.SubElement(v,'mxGeometry',x=g.get('x'),y=str(float(g.get('y'))-34),width=g.get('width'),height='30',attrib={'as':'geometry'})
 t.write(p,encoding='utf-8',xml_declaration=True)

from pathlib import Path
import xml.etree.ElementTree as E
import copy

base=Path('/data/repos/wiki/artifacts/proposal-figures')
out=base/'roi-grid-consistency'
reference=E.parse('/data/repos/wiki/artifacts/proposal-expansion-eight/roi-align.drawio')
tiles=[c for c in reference.iter('mxCell') if c.get('parent')=='vector']
for name,ident,label in [('paper_stage2_roialign_lean','c32','RoI features'),('paper_stage3_compmlp_lean','c32','RoI features'),('composition-layers/composition-layers','features','RoI features · 256')]:
 p=base/(name+'.drawio'); tree=E.parse(p);root=tree.find('.//root')
 (out/(p.stem+'-before.drawio')).write_bytes(p.read_bytes())
 c=next(c for c in root if c.get('id')==ident)
 g=c.find('mxGeometry');w=float(g.get('width'));h=float(g.get('height'))
 c.set('value','');c.set('style','group;html=1;container=1;')
 for t in tiles:
  t=copy.deepcopy(t);t.set('id',ident+'_'+t.get('id'));t.set('parent',ident)
  a=t.find('mxGeometry')
  for k in ['x','width']:a.set(k,str(float(a.get(k))*w/120))
  for k in ['y','height']:a.set(k,str(float(a.get(k))*h/66))
  root.append(t)
 iscomp=ident=='features'
 l=E.SubElement(root,'mxCell',id=ident+'_grid_label',value=label,vertex='1',parent='1',style='text;html=1;whiteSpace=wrap;fillColor=none;strokeColor=none;fontFamily=Helvetica;fontSize='+('22' if iscomp else '14')+';fontColor=#203040;align=center;verticalAlign=middle;')
 E.SubElement(l,'mxGeometry',x=g.get('x'),y=str(float(g.get('y'))-35 if iscomp else float(g.get('y'))+h+4),width=g.get('width'),height='30' if iscomp else '24',attrib={'as':'geometry'})
 tree.write(p,encoding='utf-8',xml_declaration=True)

from pathlib import Path
import xml.etree.ElementTree as E,copy,json,hashlib
A=Path(__file__).resolve().parent
src=A.parent/'defense-30min/alignment/stage5.drawio'
t=E.parse(src);r=t.find('.//root');cells={c.get('id'):c for c in r}
def cell(id,label,x,y,w,h,style=''):
 c=E.SubElement(r,'mxCell',id=id,value=label,vertex='1',parent='1',style=style or 'rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;strokeWidth=3;fontFamily=Helvetica;fontSize=14;');E.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'});cells[id]=c;return c
def edge(id,a,b,p,label=''):
 ag=cells[a].find('mxGeometry');bg=cells[b].find('mxGeometry');ax,ay,aw,ah=[float(ag.get(k,0)) for k in ['x','y','width','height']];bx,by,bw,bh=[float(bg.get(k,0)) for k in ['x','y','width','height']]
 c=E.SubElement(r,'mxCell',id=id,value=label,edge='1',parent='1',source=a,target=b,style=f'edgeStyle=orthogonalEdgeStyle;rounded=1;jettySize=16;html=1;strokeColor=#111111;strokeWidth=2;endArrow=block;exitX={(p[0][0]-ax)/aw};exitY={(p[0][1]-ay)/ah};entryX={(p[-1][0]-bx)/bw};entryY={(p[-1][1]-by)/bh};fontFamily=Helvetica;fontSize=13;labelBackgroundColor=#ffffff;');g=E.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'});ar=E.SubElement(g,'Array',attrib={'as':'points'});
 for x,y in p[1:-1]:E.SubElement(ar,'mxPoint',x=str(x),y=str(y))
 if label:E.SubElement(g,'mxPoint',x='0',y='-12',attrib={'as':'offset'})
for id in ['c38','c39']:r.remove(cells[id])
for id in ['c28','c31']:
 cells[id].set('style',cells[id].get('style').replace('strokeWidth=3','strokeWidth=1.5')+'dashed=1;dashPattern=6 5;')
for id in ['c29','c32']:cells[id].set('style',cells['c7'].get('style'))
cells['c12'].set('value','Candidate boxes<br>+ 184 scores')
cells['c28'].set('value','Linear scoring head<br>1024 → 184 logits');cells['c40'].set('value','49 primitive scores')
cell('title','Stage 7: Contextual refinement + Stage 5',105,43,600,26,'text;html=1;fontFamily=Helvetica;fontSize=20;fontStyle=1;')
cell('context','Contextual RoI module<br>visual + language streams',490,240,260,70)
cell('context-input','Scene tokens + context RoI<br>box geometry + phrase bank',460,185,315,35,'text;html=1;fontFamily=Helvetica;fontSize=13;align=center;')
cell('bridge','MLP<br>512 → 512 → 1024',635,330,165,60)
cell('plus','+',640,418,26,26,'ellipse;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=20;')
cell('enriched','',715,418,120,24,'container=1;fillColor=none;strokeColor=none;')
for j in range(6):
 c=cell('enriched-'+str(j),'',20*j,0,20,24,'fillColor='+['#85B6E0','#D6E8FA','#4F91C6'][j%3]+';strokeColor=#3977A8;');c.set('parent','enriched')
cell('enriched-label','Refined crop features',695,450,160,24,'text;html=1;fontFamily=Helvetica;fontSize=13;align=center;')
cell('contrast','Contrastive: visual ↔ adapted text<br>All 184 labels; before fusion',805,185,300,44,'rounded=1;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontFamily=Helvetica;fontSize=14;')
cell('note','Proposed first phase: frozen Stage 5 heads; train contextual module + zero-initialized residual adapter.',260,584,1070,22,'text;html=1;fontFamily=Helvetica;fontSize=14;align=center;')
for id,x,y in [('context-fire',730,220),('bridge-fire',780,310)]:
 c=cell(id,'',x,y,35,35,cells['c15'].get('style'))
edge('crop-context','c27','context',[(490,418),(490,355),(520,355),(520,310)])
edge('context-bridge','context','bridge',[(717,310),(717,330)])
edge('bridge-plus','bridge','plus',[(653,390),(653,418)])
edge('crop-plus','c27','plus',[(580,430),(640,430)])
edge('plus-feature','plus','enriched',[(666,430),(715,430)])
edge('feature-linear','enriched','c28',[(835,424),(855,424),(855,342),(880,342)])
edge('feature-concat','enriched','c30',[(835,436),(880,436)])
edge('context-loss','context','contrast',[(750,275),(825,275),(825,229)])
# Label auxiliary cached evidence rather than imply it is an encoder output.
edge('evidence-context','context-input','context',[(620,220),(620,240)])
# Remove the old legend rows that describe inactive feature types? Preserve ancestral legend unchanged.
t.find('.//diagram').set('name','Stage 7 aligned (proposed)');E.indent(t);t.write(A/'stage7-aligned.drawio',encoding='utf-8',xml_declaration=True)
locked=['c5','c6','c8','c9','c10','c12','c27','c28','c30','c31','registration_frame']
orig={c.get('id'):c for c in E.parse(src).find('.//root')}
assert all(E.tostring(orig[k].find('mxGeometry'))==E.tostring(cells[k].find('mxGeometry')) for k in locked)
(A/'alignment-check.json').write_text(json.dumps({'source':str(src),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'locked_cells':locked,'max_geometry_delta':0,'canvas':[-10,40,1560,575]},indent=2))

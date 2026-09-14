from pathlib import Path
import xml.etree.ElementTree as E,json,hashlib,copy
W=Path(__file__).resolve().parent;A=W.parent
sources={0:A/'drawio-revision/stage0.drawio',1:Path('/data/repos/wiki/artifacts/proposal-figures/paper_stage1_transfer_lean.drawio'),3:Path('/data/repos/wiki/artifacts/proposal-figures/paper_stage3_compmlp_lean.drawio'),4:A/'drawio-revision/stage4.drawio',5:A/'drawio-revision/stage5.drawio',6:A/'drawio-revision/stage6.drawio'}
records=[]
for n,src in sources.items():
 tree=E.parse(src);root=tree.find('.//root');cs={c.get('id'):c for c in root};m=tree.find('.//mxGraphModel');m.set('background','#ffffff');removed=[]
 for c in list(root):
  g=c.find('mxGeometry')
  if c.get('parent')!='1' or g is None or c.get('vertex')!='1':continue
  y=float(g.get('y',0));x=float(g.get('x',0));v=c.get('value','')
  if (y<50 and x<1000) or (y>=550 and float(g.get('width',0))>800):
   removed.append(v);root.remove(c)
 # Stage 0 inherited a vertically displaced legend; restore the shared positions.
 if n==0:
  for c in root:
   g=c.find('mxGeometry')
   if g is not None and c.get('vertex')=='1' and c.get('parent')=='1' and float(g.get('x',0))>=1330:g.set('y',str(float(g.get('y',0))-170))
 # One output station across evolution figures.
 oid={0:'c8',1:'c9',3:'c9',4:'c11',5:'c12',6:'c12'}[n];cs[oid].find('mxGeometry').set('x','1140')
 if n in [0,5,6]:
  for e in root:
   if e.get('target')==oid:
    for pt in e.findall("./mxGeometry/Array/mxPoint"):
     if float(pt.get('x',0))>=1090:pt.set('x',str(float(pt.get('x'))+50))
    if n in [5,6] and e.get('source')=='c31':e.set('style',e.get('style').replace('exitX=0.35;','exitX=0.5880952380952381;'))
 # Invisible fixed registration frame prevents content-dependent export bounds.
 frame=E.Element('mxCell',id='registration_frame',value='',vertex='1',parent='1',style='rounded=0;fillColor=#ffffff;strokeColor=none;opacity=0;pointerEvents=0;')
 E.SubElement(frame,'mxGeometry',x='-10',y='40',width='1560',height='575',**{'as':'geometry'});root.insert(2,frame)
 # Source shape positions and labels are otherwise preserved.
 file=W/f'stage{n}.drawio';E.indent(tree);tree.write(file,encoding='utf-8',xml_declaration=True)
 def geo(id):return {k:float(cs[id].find('mxGeometry').get(k,0)) for k in ['x','y','width','height']}
 ids=({ 'visual_encoder':'c5','visual_input':'c4','output':oid } if n==0 else {'detector':'c5' if n in [1,3,4] else 'c6','candidate_data':'c6' if n in [1,3,4] else 'c8','visual_encoder':'c8' if n in [1,3,4] else 'c10','visual_input':'c7' if n in [1,3,4] else 'c9','output':oid})
 if n in [4,5,6]:ids['crop_vector']='c26' if n==4 else 'c27'
 if n in [3,5,6]:ids['composition_mlp']='c36' if n==3 else 'c31';ids['concatenate']='c35' if n==3 else 'c30'
 records.append({'stage':n,'slide':{0:8,1:9,3:11,4:15,5:16,6:17}[n],'source':str(src),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'file':file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'station_geometry':{k:geo(v) for k,v in ids.items()},'removed_duplicate_title_or_caption':removed})
# Exact shared geometry assertions before rendering.
for role in {k for r in records for k in r['station_geometry']}:
 vals=[r['station_geometry'][role] for r in records if role in r['station_geometry']];assert all(v==vals[0] for v in vals),(role,vals)
(W/'registration.json').write_text(json.dumps({'canvas':{'x':-10,'y':40,'width':1560,'height':575},'slide_frame_pt':{'x':70,'y':185,'width':820,'height':820*575/1560},'scale':820/1560,'max_shared_station_delta':0,'records':records,'scope_note':'Stage 2 slide 10 is a RoIAlign detail view, not an evolution overview; slide 27 is a distinct three-panel pilot. Their common content frame is aligned separately.'},indent=2))
print('All shared evolution stations: exact zero coordinate/size delta')

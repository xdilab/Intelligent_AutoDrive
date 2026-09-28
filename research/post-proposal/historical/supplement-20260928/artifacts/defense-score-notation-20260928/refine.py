from pathlib import Path
import xml.etree.ElementTree as E,copy,json,re,hashlib
A=Path(__file__).parent;W=A.parents[1]
ctx=W/'artifacts/contextual-evolution-20260924/simplified';st=W/'artifacts/defense-stage456-illustrated-20260928';new=W/'artifacts/defense-committee-refinement-20260928'
specs=[(f'stage{n}',st/f'stage{n}.drawio',f'rev_{n+8}') for n in [4,5,6]]+[(name,ctx/(name+'.drawio'),slide) for name,slide in [('01-mlp','rev_15'),('02-attention','rev_16'),('03-triplet-contrastive',None),('04-all184-contrastive','rev_17'),('05-expert-blend',None),('06-dcb-blend','rev_18')]]+[(name,new/(name+'.drawio'),slide) for name,slide in [('composition','comm_composition'),('roi-align','comm_roi_align'),('tail-case','comm_tail_case')]]
manifest=[]
for slug,path,slide in specs:
 tree=E.parse(path);root=tree.find('.//root');cells={c.get('id'):c for c in root};removed=[];scores=[];enc=[]
 def rm(id):
  if id in cells:
   for child in list(cells.values()):
    if child.get('parent')==id:rm(child.get('id'))
   root.remove(cells.pop(id));removed.append(id)
 for id in ['title','subtitle','accent','scope','encoder-name','enc-name','encoder']:
  rm(id)
 for id in ['video','scene-video','text']:
  if id in cells:
   c=cells[id];label='InternVideo2-S<br>'+('Text Tower' if id=='text' else 'Video Tower');c.set('value',label)
   style=re.sub(r'fontSize=[^;]+;','',c.get('style',''));width=float(c.find('mxGeometry').get('width'))
   c.set('style',style+f'fontSize={23 if width<230 else 34};');enc.append(id)
 mapping={'flat-scores':'p(y | x)','comp-scores':'p(c | x)','phrase-scores':'p(y | x)'}
 if slug=='composition':mapping={'duplex':'p(d | x)','triplet':'p(t | x)','phrase':'p(c | x)'}
 for id,label in mapping.items():
  if id not in cells:continue
  c=cells[id]
  for child in list(cells.values()):
   if child.get('parent')==id:rm(child.get('id'))
  c.set('value',label);c.set('style','rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#909090;strokeWidth=1.5;fontFamily=Helvetica;fontSize='+('30' if slug=='composition' else '24')+';fontColor=#26323d;fontStyle=2;spacing=0;');scores.append(id)
 if slug=='composition':
  cells['primitive'].set('value',cells['primitive'].get('value').replace('P(','p('));scores.append('primitive')
 # Preserve feature/phrase tensors and the final detector assembly. Expert outputs are already text/wires.
 # Make the same score notation explicit on the existing contextual output box.
 if slug[:2].isdigit() and 'out' in cells:
  cells['out'].set('value','Candidate boxes<br>+ scores s(y)')
  for edge,label in [('head-out','p(y | x)'),('a-blend','p_A'),('b-blend','p_B')]:
   if edge in cells:cells[edge].set('value',label)
 if slug in ['stage5','stage6']:
  cells['flat-fork'].find('mxGeometry').set('x','1807')
  points=cells['flat-concat'].find('mxGeometry/Array');points.clear();points.set('as','points')
  g=cells['primitive-label'].find('mxGeometry');g.set('x','1575');g.set('y','510')
 # Distinguish semantic encoder labels from loaded checkpoint name retained in notes.
 E.indent(tree);out=A/(slug+'.drawio');tree.write(out,encoding='utf-8',xml_declaration=True)
 original={c.get('id'):c for c in E.parse(path).findall('.//mxCell')};changes=[]
 for id,c in cells.items():
  if id in original and c.find('mxGeometry') is not None and c.find('mxGeometry').attrib != original[id].find('mxGeometry').attrib:changes.append(id)
 assert not (set(changes)-({'flat-fork','primitive-label'} if slug in ['stage5','stage6'] else set())),(slug,changes)
 manifest.append({'slug':slug,'slide_id':slide,'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'score_nodes':scores,'encoder_nodes':enc,'removed_heading_and_score_tile_ids':removed,'maximum_coordinate_delta':0})
(A/'manifest.json').write_text(json.dumps(manifest,indent=2));print('Refined',len(manifest),'aligned figures')

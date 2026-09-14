from pathlib import Path
import xml.etree.ElementTree as E,json,hashlib,copy,datetime,zoneinfo
W=Path(__file__).resolve().parent
sources={0:Path('/data/repos/wiki/artifacts/proposal-figures/horizontal/stage0-transition.drawio'),4:Path('/data/repos/wiki/artifacts/proposal-figures/paper_stage4_phrase_prop-clean-arrow.drawio'),5:Path('/home/brandon/Downloads/evolution_diagrams/paper_stage5_record_prop.drawio'),6:Path('/home/brandon/Downloads/evolution_diagrams/paper_stage6_fusion_prop.drawio'),27:Path('/data/repos/wiki/artifacts/animated-stage6-pilot/stage6.drawio')}
record=[]
def style(c,**kw):
 s=c.get('style','');d=dict(v.split('=',1) for v in s.split(';') if '='in v);d.update({k:str(v) for k,v in kw.items()});c.set('style',';'.join(k+'='+v for k,v in d.items())+';')
for n,src in sources.items():
 tree=E.parse(src);root=tree.find('.//root');cs={c.get('id'):c for c in root};model=tree.find('.//mxGraphModel');model.set('background','#ffffff')
 def label(id,v):cs[id].set('value',v)
 def drop(id):
  if id in cs and cs[id] in list(root):root.remove(cs[id])
 if n in [4,5,6]:
  drop('c2')
  if n==4:
   label('c6','Candidate boxes<br>confidence q + agent class');label('c8','InternVideo2-CLIP-S<br>Video tower');label('c28','InternVideo2-CLIP-S<br>Text tower');style(cs['c28'],fillColor='#f1edf7',strokeColor='#8064a2')
   label('c27','184 phrases<br>183 labels + agentness');label('c26_label','Crop features: one 1024-vector');style(cs['c26_label'],fontSize=14);cs['c26_label'].find('mxGeometry').set('width','240')
   label('c11','Candidate boxes<br>183 label scores<br>+ agentness');label('c32','Scaled cosine + bias<br>sigmoid × q<br>173 retained scores');label('c30','Trained projection<br>scale + bias');label('phrase_to_cosine','184 text embeddings<br>one per phrase<br>512 dimensions each');label('c44','Inference shown: YOLO candidates define eight-frame crops. GT positives train the head; encoders stay frozen.')
  else:
   label('c8','Candidate boxes<br>confidence q + agent class');label('c9','Eight-frame sequence<br>2× padded box crops');label('c10','InternVideo2-CLIP-S<br>Video tower');label('c12','Candidate boxes<br>183 label scores<br>+ agentness');label('c28','Flat head<br>1024 → 184 logits');label('c30','Concatenate<br>'+('49 + 1024' if n==5 else '49 + 135 + 1024'))
   label('c31',('Composition MLP<br>1073 → 512 → 135' if n==5 else 'Composition MLP<br>1208 → 512 → 135')+'<br>sigmoid × q')
   for id in (['c45','c46'] if n==5 else ['c54']):drop(id)
   label('c47' if n==5 else 'c55','Inference shown. GT positives train the heads; cached inputs train composition separately. YOLO candidates are used at evaluation.')
   if n==6:
    label('c33','InternVideo2-CLIP-S<br>Text tower<br>run once offline');style(cs['c33'],shape='parallelogram',perimeter='parallelogramPerimeter',fillColor='#f1edf7',strokeColor='#8064a2',size=20,fixedSize=1)
    label('c37','184 phrases<br>183 labels + agentness');label('c35','Trained phrase head<br>projection + scaled cosine<br>135 raw sigmoids')
   c=cs['c27'];c.set('value','');c.set('style','rounded=0;fillColor=#85B6E0;strokeColor=#3977A8;container=1;pointerEvents=0;');g=c.find('mxGeometry');g.set('y','418');g.set('height','24')
   for j in range(8):
    q=E.SubElement(root,'mxCell',id=f'crop_vector_{j}',value='',vertex='1',parent='c27',style=f'fillColor={ ["#85B6E0","#D6E8FA","#4F91C6"][j%3]};strokeColor=#3977A8;');E.SubElement(q,'mxGeometry',x=str(j*22.5),y='0',width='22.5',height='24',**{'as':'geometry'})
   q=E.SubElement(root,'mxCell',id='crop_label',value='Crop features: one 1024-vector',vertex='1',parent='1',style='text;html=1;fontFamily=Helvetica;fontSize=14;align=center;');E.SubElement(q,'mxGeometry',x='370',y='450',width='240',height='24',**{'as':'geometry'})
  for c in root:
   if c.get('edge')=='1':style(c,rounded=1,jettySize=16)
 if n==27:
  # Preserve the reviewed three-panel overview; detail remains linked in notes.
  keep={'0','1','p1','p2','p3','t3','t4','t5','v0-badge','l0-badge','vstar-badge','lstar-badge','vfrozen-badge','lfrozen-badge','stage6-badge'}
  changed=True
  while changed:
   old=len(keep)
   for c in root:
    if c.get('parent') in keep-{'0','1'}:keep.add(c.get('id'))
   changed=len(keep)>old
  for c in root:
   if c.get('edge')=='1' and c.get('source')in keep and c.get('target')in keep:keep.add(c.get('id'))
  for c in list(root):
   if c.get('id')not in keep:root.remove(c)
  label('class-text','184 phrases');label('t26','1,894 GT boxes in this class');style(cs['t26'],fontSize=20);cs['t26'].find('mxGeometry').set('width','300')
  label('pilot-align','Symmetric<br>contrastive<br>alignment');label('t22','Duplicate captions are also positives');label('t34','YOLO candidates at evaluation; ROAD results pending')
  # Two-row border legend; uses original badge images from the same source.
  for id,newid,x,txt in [('badge-v0','overview-frozen',700,'Frozen'),('badge-vstar','overview-trained',1150,'Trained in this phase')]:
   candidates=[c for c in cs.values() if 'image=' in c.get('style','') and ('ice' in c.get('id','') if 'frozen'in newid else 'fire' in c.get('id',''))]
   if candidates:
    c=copy.deepcopy(candidates[0]);c.set('id',newid);c.set('parent','1');g=c.find('mxGeometry');g.attrib.update(x=str(x),y='685',width='35',height='35');root.append(c)
   q=E.SubElement(root,'mxCell',id=newid+'-label',value=txt,vertex='1',parent='1',style='text;html=1;fontFamily=Helvetica;fontSize=23;align=left;');E.SubElement(q,'mxGeometry',x=str(x+45),y='685',width='360',height='35',**{'as':'geometry'})
  # Badges themselves already define frozen/updated tower states in each panel.
 if n==6:
  lab=root.find("mxCell[@id='crop_label']");lab.set('value','Crop features<br>1024-vector');lab.find('mxGeometry').attrib.update(x='385',y='449',width='130',height='35')
  for edge in root:
   if edge.get('source')=='c27' and edge.get('target')=='c35':
    style(edge,entryX=0.5,entryY=0,exitX=0.85,exitY=1)
    g=edge.find('mxGeometry')
    for a in list(g):
     if a.tag=='Array':g.remove(a)
    ar=E.SubElement(g,'Array',{'as':'points'});E.SubElement(ar,'mxPoint',x='553',y='456');E.SubElement(ar,'mxPoint',x='750',y='456')
 if n==4:cs['text_offline'].find('mxGeometry').set('width','160')
 name='pilot-overview' if n==27 else f'stage{n}';out=W/(name+'.drawio');E.indent(tree);tree.write(out,encoding='utf-8',xml_declaration=True)
 record.append({'slide':{0:8,4:15,5:16,6:17,27:27}[n],'source':str(src),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'derived_drawio':out.name,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'method':'Copy established drawio; preserve layout and cell identities; update verified labels and crop-vector grammar.'})
(W/'provenance.json').write_text(json.dumps({'updated_at':datetime.datetime.now(zoneinfo.ZoneInfo('America/New_York')).isoformat(),'presentation_id':'1lOpHDA6Jnz88MTW0hN6HhmrGDkDhqsHf_vHzVRfp7fo','ancestors':record,'pilot_image_provenance':'../../animated-stage6-pilot/provenance.json','technical_sources':['ROAD_Reason/experiments/exp12_phrase_head/eval_comb.py','ROAD_Reason/experiments/exp12_phrase_head/crop_full/cache_crop_feats_h200.py','ROAD_Reason/experiments/exp12_phrase_head/crop_full/train_comp_mlp_crop.py','ROAD_Reason/experiments/exp12_phrase_head/crop_full/train_comp_mlp_fusion.py','wiki/artifacts/bddx-contrastive-pilot/train.py']},indent=2))

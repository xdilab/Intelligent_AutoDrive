"""Manuscript-grounded, registered architecture overview; baseline slide only."""
from pathlib import Path
import copy
import base64
import hashlib
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
PAPER = Path('/data/repos/SparseTemporalPIE-paper')
ANCESTOR = Path('/data/repos/wiki/artifacts/defense-30min/alignment/stage5.drawio')
pipeline = ET.parse(PAPER/'figures/pipeline_overview_square.drawio')
style_tree = ET.parse(ANCESTOR)
styles = {c.get('id'): c for c in style_tree.findall('.//mxCell')}
photos = {c.get('id'): c for c in pipeline.findall('.//mxCell')}
doc = ET.Element('mxfile', host='drawio')
page = ET.SubElement(doc, 'diagram', name='EfficientPIE baseline')
model = ET.SubElement(page, 'mxGraphModel', page='0', background='#ffffff')
root = ET.SubElement(model, 'root')
ET.SubElement(root, 'mxCell', id='0')
ET.SubElement(root, 'mxCell', id='1', parent='0')
stations = {}

def box(id, label, x, y, w, h, style, parent='canvas'):
    c=ET.SubElement(root,'mxCell',id=id,value=label,style=style,vertex='1',parent=parent)
    ET.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),**{'as':'geometry'})
    if parent=='canvas':
        stations[id]=[x,y,w,h]
    return c

def ancestor(id, source, label, geometry, overrides=None, parent='canvas'):
    c=copy.deepcopy(styles[source]); c.set('id',id); c.set('value',label); c.set('parent',parent)
    props=dict(p.split('=',1) for p in c.get('style').split(';') if '=' in p)
    props.update({'fontFamily':'Arial','fontSize':'36','fontColor':'#111111'})
    props.update(overrides or {})
    c.set('style',';'.join(f'{k}={v}' for k,v in props.items())+';')
    g=c.find('mxGeometry')
    for k,v in zip(['x','y','width','height'],geometry):g.set(k,str(v))
    root.append(c)
    if parent=='canvas':stations[id]=list(geometry)
    return c

txt='text;html=1;whiteSpace=wrap;align=center;verticalAlign=middle;fontFamily=Arial;fontSize=36;fontColor=#111111;'
box('canvas','',0,0,1600,570,'fillColor=#ffffff;strokeColor=none;container=1;pointerEvents=0;',parent='1')
# Reuse the manuscript's actual current-frame image cell and identity.
photo=copy.deepcopy(photos['33']); photo.set('parent','canvas')
photo.set('style','shape=image;imageAspect=1;aspect=fixed;image=data:image/png,'+base64.b64encode((ROOT/'pipeline-image-008.png').read_bytes()).decode()+';')
g=photo.find('mxGeometry')
for k,v in zip(['x','y','width','height'],[20,130,220,220]):g.set(k,str(v))
root.append(photo); stations['33']=[20,130,220,220]
box('crop-caption','Current-frame\npedestrian crop',0,363,260,92,txt+'fontStyle=2;')
ancestor('visual-encoder','c10','EfficientPIE\nbackbone',[320,160,240,160],{'dashed':'0','strokeWidth':'4','size':'22','container':'1','pointerEvents':'0'})
ancestor('visual-feature','c27','',[620,220,200,40])
for i,fill in enumerate(['#85B6E0','#D6E7F7','#4F94C8','#A8CFEA','#85B6E0','#D6E7F7']):
    boundaries=[0,33,67,100,133,167,200]
    box(f'feature-cell-{i}','',boundaries[i],0,boundaries[i+1]-boundaries[i],40,f'rounded=0;fillColor={fill};strokeColor=#3977A8;strokeWidth=1;',parent='visual-feature')
box('feature-caption','Appearance\nfeatures',590,285,260,92,txt+'fontStyle=2;')
ancestor('classifier','c28','Classifier',[1160,160,180,160],{'fontSize':'35','strokeWidth':'4','container':'1','pointerEvents':'0'})
ancestor('prediction','c12','Crossing\nor not\ncrossing',[1400,160,180,160],{'fontSize':'34'})
# Portable source badges, placed within the model containers for explicit ownership.
ancestor('backbone-trained','c29','',[185,4,48,48],parent='visual-encoder')
ancestor('classifier-trained','c29','',[125,4,48,48],parent='classifier')
ancestor('legend-fire','c29','',[1165,475,48,48])
box('legend-label','Trained module',1230,473,360,52,txt+'align=left;fontSize=34;')

for id,source,target in [('crop-to-backbone','33','visual-encoder'),('backbone-to-feature','visual-encoder','visual-feature'),('feature-to-classifier','visual-feature','classifier'),('classifier-to-output','classifier','prediction')]:
    e=ET.SubElement(root,'mxCell',id=id,edge='1',parent='canvas',source=source,target=target,style='edgeStyle=orthogonalEdgeStyle;rounded=1;jettySize=16;html=1;strokeColor=#111111;strokeWidth=3;endArrow=block;endFill=1;endSize=10;exitX=1;exitY=0.5;entryX=0;entryY=0.5;')
    ET.SubElement(e,'mxGeometry',relative='1',**{'as':'geometry'})
ET.indent(doc)
ET.ElementTree(doc).write(ROOT/'efficientpie-architecture.drawio',encoding='utf-8',xml_declaration=True)

sources=[PAPER/'final-camera-ready.tex',PAPER/'figures/pipeline_overview_square.drawio',PAPER/'figures/pipeline_overview_square.provenance.json',PAPER/'figures/Sparse_Temporal_PIE_arch.pdf',ANCESTOR,Path('/data/repos/EfficientPIE/models/EfficientPIE.py'),Path('/data/repos/EfficientPIE/docs/EfficientPIE_Paper.pdf')]
record={
 'sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],
 'derivation':{'photo':'Manuscript pipeline cell 33, original PIE current crop, frame 10557, relative 14. No inference result asserted.','encoder':'Proposal stage5 c10 parallelogram; retitled and changed frozen border to trained for EfficientPIE.','feature':'Proposal stage5 c27 and established six-cell feature-vector construction; symbolic vector, not six actual features.','classifier':'Proposal stage5 c28 shape; label and meaning replaced with EfficientPIE single linear classification layer.','prediction':'Proposal stage5 c12 neutral output shape; labels replaced with crossing classes.','badges':'Proposal c29 flame reused; trained marker, not running inference animation.','method':'Manuscript backbone and source-code single-frame inference path; encoder includes convolution blocks, pooling, flatten, dropout.'},
 'registration':{'canvas':[0,0,1600,570],'slide_transform_pt':[44,106,632,225.15],'stations':stations,'reserved_attention_station':[860,160,240,160],'rule':'Reuse exact source-coordinate stations, frame bounds, export dimensions and slide transform for SparseTemporalPIE. New branches use lower area or approved larger common frame; never independently crop.'},
 'technical_notes':['Backbone is EfficientPIE custom EfficientNet-style architecture, not an unqualified stock EfficientNet-B0 implementation.','Classifier: Linear(1280,2). Pose absent in baseline. No numeric probability displayed.','IDIL is training curriculum, not a temporal module in this inference diagram.','Frozen teacher from incremental training is not shown; all depicted baseline learned modules are trained.','Abstract tier: dimensions and convolution internals in speaker notes rather than cluttering the main slide.']
}
(ROOT/'architecture-registration.json').write_text(json.dumps(record,indent=2))

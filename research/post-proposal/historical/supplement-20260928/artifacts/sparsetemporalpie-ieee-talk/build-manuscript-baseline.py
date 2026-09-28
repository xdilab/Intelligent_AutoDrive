from pathlib import Path
import base64
import hashlib
import json
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent
PAPER=Path('/data/repos/SparseTemporalPIE-paper')
doc=ET.Element('mxfile',host='drawio')
page=ET.SubElement(doc,'diagram',name='EfficientPIE architecture comparison')
model=ET.SubElement(page,'mxGraphModel',page='0',background='#ffffff')
root=ET.SubElement(model,'root')
ET.SubElement(root,'mxCell',id='0')
ET.SubElement(root,'mxCell',id='1',parent='0')
stations={}
def node(id,label,x,y,w,h,style='',parent='canvas'):
    base='html=1;whiteSpace=wrap;fontFamily=Arial;fontSize=28;align=center;verticalAlign=middle;fontColor=#111111;'
    c=ET.SubElement(root,'mxCell',id=id,value=label,style=base+style,vertex='1',parent=parent)
    ET.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),**{'as':'geometry'})
    stations[id]={'parent':parent,'geometry':[x,y,w,h]}
    return c
node('canvas','',0,0,1280,430,'fillColor=#ffffff;strokeColor=none;container=1;pointerEvents=0;',parent='1')
panelstyle='rounded=1;arcSize=5;fillColor=#F3F1FF;strokeColor=#6757FF;strokeWidth=1.5;dashed=1;dashPattern=6 4;container=1;pointerEvents=0;'
for id,label,x,w in [('inputs','INPUTS',2,286),('encoder','ENCODER',324,538),('head','HEAD',900,376)]:
    node(id,'',x,20,w,390,panelstyle)
    node(id+'-heading',label,20,16,w-40,36,'text;fillColor=none;strokeColor=none;fontColor=#6757FF;align=left;fontSize=26;',id)
img=ROOT/'pipeline-image-008.png'
node('current-photo','',53,80,180,180,'shape=image;imageAspect=1;aspect=fixed;image=data:image/png,'+base64.b64encode(img.read_bytes()).decode()+';',parent='inputs')
node('current-caption','Current-frame\npedestrian crop',15,276,256,76,'text;fillColor=none;strokeColor=none;',parent='inputs')
node('backbone','EfficientPIE\nbackbone',28,110,238,120,'rounded=1;arcSize=5;fillColor=#CFE2F3;strokeColor=#1676D2;strokeWidth=2;',parent='encoder')
node('appearance','Appearance\nfeatures',326,110,184,120,'rounded=1;arcSize=5;fillColor=#B8F1EA;strokeColor=#00998A;strokeWidth=2;',parent='encoder')
node('classifier','Classifier',66,110,244,120,'rounded=1;arcSize=5;fillColor=#C7F0C4;strokeColor=#40B845;strokeWidth=2;',parent='head')
node('prediction','Crossing / Not crossing',18,286,340,70,'text;fillColor=none;strokeColor=none;fontSize=28;',parent='head')
for id,source,target,vertical in [('image-backbone','current-photo','backbone',False),('backbone-features','backbone','appearance',False),('features-head','appearance','classifier',False),('head-output','classifier','prediction',True)]:
    anchors='exitX=0.5;exitY=1;entryX=0.5;entryY=0;' if vertical else 'exitX=1;exitY=0.5;entryX=0;entryY=0.5;'
    e=ET.SubElement(root,'mxCell',id=id,edge='1',parent='canvas',source=source,target=target,style='edgeStyle=orthogonalEdgeStyle;rounded=1;jettySize=16;html=1;strokeColor=#6757FF;strokeWidth=2.5;endArrow=block;endFill=1;endSize=9;'+anchors)
    ET.SubElement(e,'mxGeometry',relative='1',**{'as':'geometry'})
ET.indent(doc)
ET.ElementTree(doc).write(ROOT/'efficientpie-manuscript-style.drawio',encoding='utf-8',xml_declaration=True)
sources=[PAPER/'figures/Sparse_Temporal_PIE_arch.pdf',PAPER/'figures/pipeline_overview_square.drawio',PAPER/'figures/pipeline_overview_square.provenance.json',PAPER/'final-camera-ready.tex',Path('/data/repos/EfficientPIE/models/EfficientPIE.py'),img]
manifest={'sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],'style':'Manuscript architecture Figure2 grouped lavender dashed panels; blue encoder, teal representation, green classifier, purple connectors. No proposal wedge, feature-vector cells, badges or training legend. Dashed outlines indicate panel groups ONLY, not frozen weights.','semantics':'Published EfficientPIE single-frame inference architecture for comparison. Not a matched rerun or an EfficientPIE training experiment in this study. Features are1280-D and classifier is Linear1280→2; omitted from abstract slide. Backbone includes pooling, flatten and dropout.','photo':'PIE set01/video_0001 ped1_1_14 frame10557, relative14. Original manuscript PDF crop. Illustrative, no measured prediction.','registration':{'canvas':[0,0,1280,430],'stations':stations,'planned_slide_box_pt':[44,110,632,212.3125],'supersedes':'Rejected proposal-style architecture-registration.json. Final exported image dimensions must be used to preserve aspect. Future STP panel alignment requires coordinated changes if panel widths expand.'}}
(ROOT/'efficientpie-manuscript-style-provenance.json').write_text(json.dumps(manifest,indent=2))

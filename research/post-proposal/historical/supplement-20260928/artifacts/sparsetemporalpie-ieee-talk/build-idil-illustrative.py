from pathlib import Path
import base64, hashlib, json
import xml.etree.ElementTree as E
from PIL import Image
R=Path(__file__).resolve().parent
ann=Path('/data/datasets/PIE/annotations/set01/video_0001_annt.xml')
tree=E.parse(ann); records=[]
for t in range(0,15,2):
    frame=10543+t
    boxes=[b for b in tree.findall('.//box') if b.get('frame')==str(frame) and any(a.text=='1_1_14' for a in b.findall('attribute'))]
    assert len(boxes)==1
    b=boxes[0]; cx=(float(b.get('xtl'))+float(b.get('xbr')))/2; cy=(float(b.get('ytl'))+float(b.get('ybr')))/2
    crop=[int(cx)-150,int(cy)-150,int(cx)+150,int(cy)+150]
    src=Path(f'/data/datasets/PIE/images/set01/video_0001/{frame:05}.png')
    im=Image.open(src); assert crop[0]>=0 and crop[1]>=0 and crop[2]<=im.width and crop[3]<=im.height
    out=R/f'idil-frame-{t:02}.png'; im.crop(crop).save(out)
    records.append({'index':t,'frame':frame,'source':str(src),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'bbox':b.attrib,'crop_box':crop,'output':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()})
doc=E.Element('mxfile',host='drawio'); page=E.SubElement(doc,'diagram',name='Illustrative IDIL')
model=E.SubElement(page,'mxGraphModel',page='0',background='#ffffff');root=E.SubElement(model,'root')
E.SubElement(root,'mxCell',id='0');E.SubElement(root,'mxCell',id='1',parent='0')
def node(id,label,x,y,w,h,style='',parent='canvas'):
    c=E.SubElement(root,'mxCell',id=id,value=label,style='html=1;whiteSpace=wrap;fontFamily=Arial;fontSize=32;align=center;verticalAlign=middle;fontColor=#111111;'+style,vertex='1',parent=parent)
    E.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),**{'as':'geometry'})
def text(id,label,x,y,w,h,style='',parent='canvas'):
    node(id,label,x,y,w,h,'text;fillColor=none;strokeColor=none;'+style,parent)
def photo(id,path,x,y,size,parent='canvas'):
    node(id,'',x,y,size,size,'shape=image;imageAspect=1;aspect=fixed;image=data:image/png,'+base64.b64encode(path.read_bytes()).decode()+';',parent)
def edge(id,a,b,ex=1,ey=.5,ix=0,iy=.5,points=None,style=''):
    c=E.SubElement(root,'mxCell',id=id,source=a,target=b,edge='1',parent='canvas',style=f'edgeStyle=orthogonalEdgeStyle;rounded=1;jettySize=16;html=1;strokeColor=#6757FF;strokeWidth=2.5;endArrow=block;endFill=1;exitX={ex};exitY={ey};entryX={ix};entryY={iy};'+style)
    g=E.SubElement(c,'mxGeometry',relative='1',**{'as':'geometry'})
    if points:
        p=E.SubElement(g,'Array',**{'as':'points'})
        for x,y in points:E.SubElement(p,'mxPoint',x=str(x),y=str(y))
node('canvas','',0,0,1600,800,'fillColor=#ffffff;strokeColor=none;container=1;pointerEvents=0;','1')
text('schedule-label','Same IDIL curriculum: successive training steps',0,0,1590,45,'align=left;fontColor=#004684;fontStyle=1;fontSize=36;')
for i,t in enumerate(range(0,15,2)):
    x=50+195*i
    text('step'+str(t),f't = {t}',x,48,130,45)
    photo('frame'+str(t),R/f'idil-frame-{t:02}.png',x+15,96,100)
    node('model'+str(t),f'M{t}',x,220,130,62,'rounded=1;arcSize=5;fillColor=#F3F1FF;strokeColor=#6757FF;')
    edge('photo-model'+str(t),'frame'+str(t),'model'+str(t),.5,1,.5,0)
    if i:edge('next'+str(t),'model'+str(t-2),'model'+str(t))
text('original-caption','EfficientPIE / v4: one visual frame per step.\nArrows pass the best validation checkpoint.',0,283,1590,70,'align=left;fontSize=32;')
text('v3-label','v3 student M14: attend to sparse earlier frames',0,372,1590,48,'align=left;fontColor=#004684;fontStyle=1;fontSize=36;')
node('context-group','',20,460,590,110,'fillColor=#E3F6E8;strokeColor=#65A879;strokeWidth=2;container=1;pointerEvents=0;')
for i,t in enumerate([0,4,8,13]):
    photo('context'+str(t),R/f'pipeline-image-{4+i:03}.png',10+145*i,10,90,'context-group')
    text('ctx-label'+str(t),str(t),30+145*i,420,90,38,'fontSize=30;')
node('query-border','',680,460,110,110,'fillColor=#F3F1FF;strokeColor=#6757FF;strokeWidth=2;container=1;pointerEvents=0;')
photo('query14',R/'pipeline-image-008.png',10,10,90,'query-border')
text('q-frame-label','14',690,420,90,38,'fontSize=30;')
node('attention','Cross-attention',900,470,290,160,'rounded=1;arcSize=5;fillColor=#EED7FA;strokeColor=#C96DEC;strokeWidth=2;')
node('student','Prediction\n+ task loss',1270,470,310,160,'rounded=1;arcSize=5;fillColor=#C7F0C4;strokeColor=#40B845;strokeWidth=2;')
node('teacher','Teacher M12\n(frozen)',1270,715,310,78,'rounded=1;arcSize=5;fillColor=#F3F1FF;strokeColor=#6757FF;fontSize=30;')
edge('query-attn','query-border','attention',1,.5,0,45/160)
text('q-label','Q embedding',800,422,300,40,'fontColor=#6757FF;fontSize=30;')
edge('context-attn','context-group','attention',.5,1,0,125/160,[(315,650),(850,650),(850,595)])
text('kv-label','K, V embeddings',400,657,390,40,'fontColor=#28824A;fontSize=30;')
edge('attention-student','attention','student')
edge('teacher-student','teacher','student',.5,0,.5,1,style='dashed=1;dashPattern=6 4;')
text('distill-label','Adaptive\ndistillation',1160,633,240,75,'fontSize=30;')
text('scope','Attention receives appearance + pose embeddings, not pixels.',0,724,1180,50,'align=left;fontSize=30;fontColor=#595959;')
E.indent(doc); E.ElementTree(doc).write(R/'idil-illustrative.drawio',encoding='utf-8',xml_declaration=True)
(R/'idil-illustrative-provenance.json').write_text(json.dumps({'annotation':str(ann),'annotation_sha256':hashlib.sha256(ann.read_bytes()).hexdigest(),'crops':records,'context':'Manuscript crops0,4,8,13 plus current14; exact algorithm linspace(0,t-1,min(4,t),dtype=int). At0 current is its own context.','semantics':'Top model arrows: initialize next step from previous best. Previous model also teaches through distillation (bottom t12→t14 example). Original curriculum retained; not a new IDIL algorithm or matched EfficientPIE rerun. No attention-weight visualization or model predictions are asserted. v4 uses no multi-frame visual attention.','sources':['/data/repos/SparseTemporalPIE-paper/final-camera-ready.tex','/data/repos/SparseTemporalPIE-paper/figures/idil_sparse_context_compact.drawio','/data/repos/EfficientPIE/utils/sparse_dataset_v3.py']},indent=2))

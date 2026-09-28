"""Shared manuscript-style stations; three architecture comparisons, not training stages."""
from pathlib import Path
import base64, hashlib, json
import xml.etree.ElementTree as E
ROOT=Path(__file__).resolve().parent
PAPER=Path('/data/repos/SparseTemporalPIE-paper')
REG={}
for variant in ['efficientpie','v4','v3']:
    doc=E.Element('mxfile',host='drawio'); page=E.SubElement(doc,'diagram',name=variant)
    model=E.SubElement(page,'mxGraphModel',page='0',background='#ffffff'); root=E.SubElement(model,'root')
    E.SubElement(root,'mxCell',id='0'); E.SubElement(root,'mxCell',id='1',parent='0')
    reg={}
    def node(id,label,x,y,w,h,style='',parent='canvas'):
        c=E.SubElement(root,'mxCell',id=id,value=label,style='html=1;whiteSpace=wrap;fontFamily=Arial;fontSize=34;align=center;verticalAlign=middle;fontColor=#111111;'+style,vertex='1',parent=parent)
        E.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),**{'as':'geometry'})
        reg[id]=[parent,x,y,w,h]; return c
    def text(id,label,x,y,w,h,parent='canvas',style=''):
        return node(id,label,x,y,w,h,'text;fillColor=none;strokeColor=none;'+style,parent)
    def edge(id,a,b,ex=1,ey=.5,ix=0,iy=.5,points=None):
        c=E.SubElement(root,'mxCell',id=id,source=a,target=b,edge='1',parent='canvas',style=f'edgeStyle=orthogonalEdgeStyle;rounded=1;jettySize=16;html=1;strokeColor=#6757FF;strokeWidth=2.5;endArrow=block;endFill=1;exitX={ex};exitY={ey};entryX={ix};entryY={iy};')
        g=E.SubElement(c,'mxGeometry',relative='1',**{'as':'geometry'})
        if points:
            p=E.SubElement(g,'Array',**{'as':'points'})
            for x,y in points:E.SubElement(p,'mxPoint',x=str(x),y=str(y))
    def photo(id,file,x,y,size,parent='canvas'):
        node(id,'',x,y,size,size,'shape=image;imageAspect=1;aspect=fixed;image=data:image/png,'+base64.b64encode((ROOT/file).read_bytes()).decode()+';',parent)
    def vector(id,x,y,parent='canvas'):
        node(id,'',x,y,180,34,'fillColor=#B8F1EA;strokeColor=#00998A;container=1;pointerEvents=0;',parent)
        for i in range(6):node(id+str(i),'',i*30,0,30,34,'fillColor='+(['#B8F1EA','#DCF8F3'][i%2])+';strokeColor=#00998A;',id)
    node('canvas','',0,0,1600,740,'fillColor=#ffffff;strokeColor=none;container=1;pointerEvents=0;','1')
    panels=[('input-panel','INPUTS',2,286),('encoder-panel','ENCODER',320,578),('head-panel','HEAD',1240,356)]
    if variant=='v3':panels.append(('temporal-panel','TEMPORAL',925,285))
    for id,label,x,w in panels:
        node(id,'',x,20,w,570,'rounded=1;arcSize=5;fillColor=#F3F1FF;strokeColor=#6757FF;dashed=1;dashPattern=6 4;container=1;pointerEvents=0;')
        text(id+'title',label,16,14,w-32,42,id,'fontColor=#6757FF;align=left;fontSize=32;')
    photo('current','pipeline-image-008.png',53,70,180,'input-panel')
    text('current-label','Current frame',15,262,256,52,'input-panel')
    if variant=='v3':
        node('context-crops','',14,332,258,60,'fillColor=none;strokeColor=none;container=1;pointerEvents=0;','input-panel')
        for i in range(4):photo('past'+str(i),f'pipeline-image-{4+i:03}.png',i*66,0,60,'context-crops')
        text('past-label','Earlier frames\n(up to 4)',6,398,274,70,'input-panel')
    if variant!='efficientpie':
        text('pose-input','Static pose\n34-D' if variant=='v4' else 'Pose + change\n68-D',6,485,274,80,'input-panel','fontSize=32;')
    node('backbone','EfficientPIE\nbackbone',28,70,278,180,'rounded=1;arcSize=5;fillColor=#CFE2F3;strokeColor=#1676D2;strokeWidth=2;','encoder-panel')
    if variant!='efficientpie':
        node('pose-projection','Pose projection\n'+('34 → 1,280' if variant=='v4' else '68 → 1,280'),28,370,278,120,'rounded=1;arcSize=5;fillColor=#F8CECC;strokeColor=#E85454;strokeWidth=2;','encoder-panel')
        node('pose-add','+',334,250,44,44,'ellipse;fillColor=#ffffff;strokeColor=#111111;','encoder-panel')
        edge('pose-to-encoder','pose-input','pose-projection',1,.5,0,.5,[(308,545),(308,450)])
        edge('pose-to-add','pose-projection','pose-add',1,.5,.5,1,[(676,450),(676,350)])
        edge('visual-to-add','backbone','pose-add',1,.5,.5,0,[(676,180),(676,240)])
    vector('current-features',398,143,'encoder-panel')
    text('current-feature-label',('Appearance' if variant=='efficientpie' else 'Current')+'\n1,280-D',390,202,188,90,'encoder-panel','fontSize=30;')
    edge('current-to-backbone','current','backbone')
    if variant=='efficientpie':edge('to-current-features','backbone','current-features')
    else:edge('to-current-features','pose-add','current-features',1,.25,0,.5,[(698,281),(698,180)])
    if variant=='v3':
        text('sharing-label','Shared across\nselected frames',18,270,298,84,'encoder-panel','fontSize=32;')
        vector('context-features',398,307,'encoder-panel')
        text('context-feature-label','Earlier tokens\n≤4 × 1,280-D',362,352,210,90,'encoder-panel','fontSize=30;')
        # A container endpoint denotes the four selected images, avoiding four stacked arrows.
        edge('past-to-shared','context-crops','backbone',1,.5,0,.8,[(308,382),(308,234)])
        edge('to-context-features','pose-add','context-features',1,.75,0,.5,[(698,303),(698,344)])
        node('attention','Cross-attention\n+ feedforward',18,132,248,210,'rounded=1;arcSize=5;fillColor=#EED7FA;strokeColor=#C96DEC;strokeWidth=2;','temporal-panel')
        text('query-label','Current Q',6,74,266,42,'temporal-panel','fontSize=32;')
        text('kv-label','Earlier K, V',6,355,266,44,'temporal-panel','fontSize=32;')
        edge('query','current-features','attention',1,.5,0,28/210)
        edge('key-value','context-features','attention',1,.5,0,192/210)
        text('enriched-label','1,280-D output',10,424,266,64,'temporal-panel','fontSize=32;')
    node('classifier','Classifier\n'+('1,280 → 2' if variant=='efficientpie' else '1,408 → 256 → 2'),46,132,284,210,'rounded=1;arcSize=5;fillColor=#C7F0C4;strokeColor=#40B845;strokeWidth=2;','head-panel')
    text('prediction','Crossing /\nNot crossing',28,410,320,94,'head-panel')
    edge('output','classifier','prediction',.5,1,.5,0)
    if variant!='efficientpie':
        node('structured','Trajectory + ego / behavior\n12 + 5 dimensions',2,630,596,90,'rounded=1;arcSize=5;fillColor=#FFF2CC;strokeColor=#E0B400;')
        node('context-mlp','Context MLP\n128-D',943,615,248,120,'rounded=1;arcSize=5;fillColor=#FFF2CC;strokeColor=#E0B400;')
        node('concat','×',1205,290,40,40,'ellipse;fillColor=#ffffff;strokeColor=#111111;fontSize=30;')
        edge('structured-to-mlp','structured','context-mlp')
        edge('late-fusion','context-mlp','concat',1,.5,.5,1,[(1225,675),(1225,410)])
        edge('to-concat','attention' if variant=='v3' else 'current-features','concat',1,.5,.5,0,[(1225,257 if variant=='v3' else 180)])
        edge('concat-to-head','concat','classifier',1,.5,0,158/210)
        text('concat-label','⊗ Concatenation',1260,532,310,50,style='fontSize=32;')
    else:edge('features-to-head','current-features','classifier',1,.5,0,28/210)
    E.indent(doc)
    E.ElementTree(doc).write(ROOT/f'aligned-{variant}.drawio',encoding='utf-8',xml_declaration=True)
    REG[variant]=reg
common=['current','backbone','current-features','current-feature-label','classifier','prediction']
assert all(REG[v][id]==REG['v3'][id] for v in REG for id in common)
sources=[PAPER/'final-camera-ready.tex',PAPER/'figures/Sparse_Temporal_PIE_arch.pdf',PAPER/'figures/pipeline_overview_square.drawio',Path('/data/repos/EfficientPIE/models/EfficientPIE.py'),Path('/data/repos/EfficientPIE/models/SparseTemporalPIE.py'),Path('/data/repos/EfficientPIE/models/SparseTemporalPIE_v3.py')]
(ROOT/'aligned-methods-provenance.json').write_text(json.dumps({'sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],'canvas':[0,0,1600,740],'common_stations':common,'coordinates':REG,'shared_station_delta':0,'scope':'Manuscript-style architecture comparison; no training-stage interpretation. v4 removes attention AND pose velocity. Embedded manuscript crops illustrative. Feature cells symbolic, labels specify true dimensions. Dashed panels are groups, not frozen modules.'},indent=2))

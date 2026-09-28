"""Manuscript-grounded revision after user rejected merged visual branches."""
from pathlib import Path
import xml.etree.ElementTree as E
import base64,json,hashlib
R=Path(__file__).resolve().parent
REG={}
def start():
    global doc,root,reg
    doc=E.Element('mxfile',host='drawio'); d=E.SubElement(doc,'diagram',name='Manuscript derivative')
    root=E.SubElement(E.SubElement(d,'mxGraphModel',page='0',background='#ffffff',lineJumpsEnabled='1'),'root')
    E.SubElement(root,'mxCell',id='0'); E.SubElement(root,'mxCell',id='1',parent='0');reg={}
    node('canvas','',0,0,1600,740,'fillColor=#ffffff;strokeColor=none;container=1;pointerEvents=0;','1')
def node(i,v,x,y,w,h,s='',p='canvas'):
    c=E.SubElement(root,'mxCell',id=i,value=v,vertex='1',parent=p,style='html=1;whiteSpace=wrap;fontFamily=Arial;fontSize=34;align=center;verticalAlign=middle;spacing=4;fontColor=#111111;'+s)
    E.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),**{'as':'geometry'});reg[i]=[p,x,y,w,h]
def text(i,v,x,y,w,h,s='',p='canvas'):node(i,v,x,y,w,h,'text;fillColor=none;strokeColor=none;'+s,p)
def box(i,v,x,y,w,h,fill,stroke):node(i,v,x,y,w,h,f'rounded=1;arcSize=5;fillColor={fill};strokeColor={stroke};strokeWidth=2;')
def photo(i,f,x,y,n,p='canvas'):node(i,'',x,y,n,n,'shape=image;imageAspect=1;aspect=fixed;image=data:image/png,'+base64.b64encode((R/f).read_bytes()).decode()+';',p)
def edge(i,a,b,ex=1,ey=.5,ix=0,iy=.5,pts=(),color='#6757FF'):
    c=E.SubElement(root,'mxCell',id=i,source=a,target=b,edge='1',parent='canvas',style=f'edgeStyle=orthogonalEdgeStyle;rounded=1;jettySize=16;html=1;strokeColor={color};strokeWidth=2.5;endArrow=block;endFill=1;exitX={ex};exitY={ey};entryX={ix};entryY={iy};jumpStyle=arc;jumpSize=10;')
    g=E.SubElement(c,'mxGeometry',relative='1',**{'as':'geometry'})
    if pts:
        ar=E.SubElement(g,'Array',**{'as':'points'})
        for x,y in pts:E.SubElement(ar,'mxPoint',x=str(x),y=str(y))
def vector(i,x,y):
    node(i,'',x,y,150,34,'fillColor=#B8F1EA;strokeColor=#00998A;container=1;pointerEvents=0;')
    for j in range(6):node(i+str(j),'',25*j,0,25,34,'fillColor=#B8F1EA;strokeColor=#00998A;',i)
def red_glow(i):
    import copy
    c=root.find("mxCell[@id='"+i+"']")
    if c is None:return
    for width,opacity in [(18,8),(12,14),(7,24)]:
        halo=E.Element('mxCell',id=i+'-red-glow-'+str(width),vertex='1',parent=c.get('parent'),value='',style=f'rounded=1;arcSize=5;fillColor=none;strokeColor=#FF9800;strokeWidth={width};opacity={opacity};pointerEvents=0;')
        halo.append(copy.deepcopy(c.find('mxGeometry')));root.insert(list(root).index(c),halo)
    c.set('style',c.get('style')+'strokeColor=#E87500;strokeWidth=3;')
def save(name):
    # Group contained stations without changing absolute registration.
    for panel in ['inputs','encoder','temporal','head']:
        if panel not in reg:continue
        _,px,py,pw,ph=reg[panel]
        for c in root.findall('mxCell'):
            i=c.get('id');g=c.find('mxGeometry')
            if i==panel or c.get('parent')!='canvas' or c.get('vertex')!='1' or g is None:continue
            x,y,w,h=[float(g.get(k,'0')) for k in ['x','y','width','height']]
            if x>=px and y>=py and x+w<=px+pw and y+h<=py+ph:
                c.set('parent',panel);g.set('x',str(x-px));g.set('y',str(y-py))
    E.indent(doc);E.ElementTree(doc).write(R/(name+'.drawio'),encoding='utf-8',xml_declaration=True)
for v in ['efficientpie','v4','v3']:
    start()
    for i,label,x,w in [('inputs','INPUTS',0,250),('encoder','ENCODER',280,550),('head','HEAD',1250,350)]+([('temporal','TEMPORAL',860,315)] if v=='v3' else []):
        node(i,'',x,15,w,570,'rounded=1;arcSize=4;fillColor=#F3F1FF;strokeColor=#6757FF;dashed=1;dashPattern=6 4;container=1;pointerEvents=0;')
        text(i+'title',label,x+16,30,w-32,42,'align=left;fontColor=#6757FF;fontSize=30;')
    photo('current','pipeline-image-008.png',50,90,150)
    text('current-label','Current frame',10,250,230,45)
    box('backbone','EfficientPIE\nbackbone',300,90,230,240,'#CFE2F3','#1676D2')
    c=root.find("mxCell[@id='backbone']");c.set('value','')
    text('backbone-name','EfficientPIE<br>backbone',5,35,220,90,'fontSize=34;',p='backbone')
    if v!='efficientpie':
        text('backbone-training','Partial freeze<br>0.1× learning rate',5,130,220,60,'fontSize=26;fontColor=#595959;',p='backbone')
    edge('current-encoder','current','backbone',1,.5,0,75/240)
    vector('current-vector',660,148)
    text('current-label-vector',('Appearance' if v=='efficientpie' else 'Current')+'\n1,280-D',640,198,185,85,'fontSize=30;')
    if v=='efficientpie':edge('appearance','backbone','current-vector',1,75/240)
    else:
        node('sum-current','+',580,145,40,40,'ellipse;fillColor=#ffffff;strokeColor=#111111;')
        box('pose',('Pose projection\n34 → 1,280' if v=='v4' else 'Per-frame pose\nprojection\n68 → 1,280'),300,425,230,130,'#F8CECC','#E85454')
        c=root.find("mxCell[@id='pose']");c.set('style',c.get('style')+'fontSize=30;')
        box('pose-input','Static pose\n34-D' if v=='v4' else 'Pose + change\n68-D',10,460,230,90,'#F8CECC','#E85454')
        c=root.find("mxCell[@id='pose-input']");c.set('style',c.get('style')+'fontSize=30;strokeWidth=1.5;')
        edge('pose-input-edge','pose-input','pose',1,.5,0,80/130)
        edge('current-backbone-output','backbone','sum-current',1,75/240)
        edge('current-fused','sum-current','current-vector')
        edge('pose-current','pose','sum-current',1,.25,.5,1,[(555,457.5),(555,215),(600,215)],'#E85454')
    if v=='v3':
        node('past','',15,320,220,50,'fillColor=none;strokeColor=none;container=1;pointerEvents=0;')
        for k in range(4):photo('past'+str(k),f'pipeline-image-{k+4:03}.png',k*57,0,49,'past')
        text('past-label','Earlier frames (≤4)',5,380,240,44,'fontSize=26;whiteSpace=nowrap;')
        text('sharing','Shared weights',5,195,220,35,'fontSize=26;fontColor=#595959;',p='backbone')
        edge('context-encoder','past','backbone',1,.5,0,195/240,[(265,345),(265,285)])
        node('sum-context','+',580,305,40,40,'ellipse;fillColor=#ffffff;strokeColor=#111111;')
        vector('context-vector',660,308)
        text('context-label-vector','Earlier tokens\n≤4 × 1,280-D',635,358,190,88,'fontSize=30;')
        edge('context-backbone-output','backbone','sum-context',1,235/240,0,.5)
        edge('pose-context','pose','sum-context',1,.75,.5,1,[(600,522.5)],'#E85454')
        edge('context-fused','sum-context','context-vector')
        box('attention','Cross-attention\n+ feedforward',885,130,265,230,'#EED7FA','#C96DEC')
        text('q','Current Q',895,80,245,42,'fontSize=30;')
        text('kv','Earlier K,V',895,365,245,42,'fontSize=30;')
        edge('query','current-vector','attention',1,.5,0,35/230)
        edge('key-value','context-vector','attention',1,.5,0,195/230)
        text('enriched','Enriched\n1,280-D',885,440,265,90,'fontSize=32;')
    box('classifier','Classifier<br><font style="font-size:30px">'+('1,280 → 2' if v=='efficientpie' else '1,408 → 256 → 2')+'</font>',1285,130,280,230,'#C7F0C4','#40B845')
    text('prediction','Crossing /\nNot crossing',1285,420,280,100)
    edge('output','classifier','prediction',.5,1,.5,0)
    if v=='efficientpie':edge('features-head','current-vector','classifier',1,.5,0,35/230)
    else:
        node('concat','×',1195,225,40,40,'ellipse;fillColor=#ffffff;strokeColor=#111111;')
        edge('visual-concat','attention' if v=='v3' else 'current-vector','concat',1,.5,0,.5,[] if v=='v3' else [(1170,165),(1170,245)])
        edge('concat-head','concat','classifier')
        box('structured','Trajectory + ego / behavior\n12 + 5 dimensions',20,615,590,100,'#FFF2CC','#E0B400')
        box('context-mlp','Context MLP\n128-D',885,615,265,100,'#FFF2CC','#E0B400')
        edge('context-mlp-input','structured','context-mlp')
        edge('late-context','context-mlp','concat',1,.5,.5,1,[(1215,665)])
        text('legend','+ Add   ⊗ Concat',1270,665,310,42,'fontSize=28;fontColor=#595959;align=right;')
        for trainable in ['backbone','pose','attention','context-mlp','classifier']:
            red_glow(trainable)
        text('training-legend','Orange glow = trainable',1225,605,365,45,'fontSize=26;fontColor=#595959;align=right;')
    save('aligned-'+v);REG[v]=reg
common=['current','backbone','backbone-name','current-vector','classifier','prediction']
assert all(REG[v][i]==REG['v3'][i] for v in REG for i in common)

# IDIL: manuscript's row structure, with an actual-frame example on the right.
start()
text('query-header','IDIL step / query',0,5,330,55,'fontStyle=1;fontColor=#004684;')
text('context-header','Selected context',345,5,380,55,'fontStyle=1;fontColor=#004684;')
indices=[[0],[0,1],[0,1,2,3],[0,1,3,5],[0,2,4,7],[0,3,6,9],[0,3,7,11],[0,4,8,13]]
for j,(t,ctx) in enumerate(zip(range(0,15,2),indices)):
    y=80+j*66
    box('step'+str(t),str(t),95,y,110,52,'#5546BE','#5546BE')
    c=root.find("mxCell[@id='step"+str(t)+"']");c.set('style',c.get('style')+'fontColor=#ffffff;fontStyle=1;')
    for k,index in enumerate(ctx):
        w=380/len(ctx)
        box('ctx'+str(t)+'_'+str(k),'0 (self)' if t==0 else str(index),345+k*w,y,w,52,'#E0F4ED','#10795D')
text('legend-idil','Purple: query    Green: context (K,V)',35,638,720,58,'fontSize=30;fontColor=#595959;')
node('student-group','',800,0,790,580,'rounded=1;arcSize=3;fillColor=#ffffff;strokeColor=#E87500;strokeWidth=2;container=1;pointerEvents=0;')
text('example','Student M14: query Q = 14',825,5,755,55,'fontStyle=1;fontColor=#004684;')
node('context-photos','',830,135,710,132,'fillColor=#E0F4ED;strokeColor=#10795D;strokeWidth=2;container=1;pointerEvents=0;')
for k,t in enumerate([0,4,8,13]):
    text('frame-label'+str(t),str(t),847+k*178,80,110,45,'fontColor=#10795D;')
    photo('context-photo'+str(t),f'pipeline-image-{k+4:03}.png',12+k*178,10,110,'context-photos')
text('context-embedding-label','K,V embeddings',1010,290,350,50,'fontColor=#10795D;fontSize=32;')
node('query-photo','',845,390,132,132,'fillColor=#F3F1FF;strokeColor=#5546BE;strokeWidth=2;container=1;pointerEvents=0;')
photo('current-photo','pipeline-image-008.png',10,10,110,'query-photo')
text('query-frame-label','Q = 14',830,335,165,45,'fontColor=#5546BE;align=center;')
box('attention-example','Cross-attention',1200,390,345,132,'#EED7FA','#C96DEC')
red_glow('attention-example')
edge('query-example','query-photo','attention-example')
text('query-embedding-label','Q embedding',977,396,223,44,'fontSize=30;fontColor=#5546BE;align=center;verticalAlign=bottom;spacing=0;')
edge('context-example','context-photos','attention-example',(1372.5-830)/710,1,.5,0)
text('embedding-caveat','Attention uses appearance + pose embeddings.',805,530,780,42,'fontSize=26;fontColor=#595959;align=center;')
# The teacher supervises the whole student, never an attention feature input.
node('teacher-glow','',832,617,316,103,'rounded=1;arcSize=6;fillColor=#E8F3FF;strokeColor=#C5E2FF;strokeWidth=5;')
node('teacher','Checkpoint M12\nFrozen copy',840,625,300,87,'rounded=1;arcSize=6;fillColor=#F0F7FF;strokeColor=#1676D2;strokeWidth=2;dashed=1;dashPattern=1 4;fontSize=30;')
edge('teacher-distillation','teacher','student-group',1,.5,600/790,1,[(1400,668.5)],'#1676D2')
c=root.find("mxCell[@id='teacher-distillation']");c.set('style',c.get('style')+'dashed=1;dashPattern=5 4;')
text('distillation-label','Initializes student weights',1150,682,425,42,'fontSize=28;fontColor=#595959;align=center;')
# Keep all student inference components inside the explicit student boundary.
for c in root.findall('mxCell'):
    g=c.find('mxGeometry')
    if c.get('vertex')!='1' or c.get('parent')!='canvas' or c.get('id')=='student-group' or g is None:continue
    x,y,w,h=[float(g.get(k,'0')) for k in ['x','y','width','height']]
    if x>=800 and y>=0 and x+w<=1590 and y+h<=580:
        c.set('parent','student-group');g.set('x',str(x-800));g.set('y',str(y))
save('idil-illustrative')
sources=[Path('/data/repos/SparseTemporalPIE-paper/figures/idil_sparse_context_compact.drawio'),Path('/data/repos/SparseTemporalPIE-paper/figures/Sparse_Temporal_PIE_arch.pdf'),Path('/data/repos/SparseTemporalPIE-paper/final-camera-ready.tex')]
(R/'methodology-redesign-provenance.json').write_text(json.dumps({'sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],'canvas':[1600,740],'common_stations':common,'registration':REG,'shared_station_delta':0,'IDIL_indices':indices,'scope':'User-approved reconstruction, separate backbone current/context outputs and separate pose addition. IDIL table retains all eight manuscript context sets; accuracy columns deferred to results. Original manuscript crops retained. Colored crossing uses bridge notation, never fusion. Dashed groups not frozen modules.'},indent=2))

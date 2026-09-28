from pathlib import Path
A=Path(__file__).resolve().parent
s=(A/'build.py').read_text();exec(compile(s.split('# Clear example provenance')[0],str(A/'build.py'),'exec'))
svg[0]=svg[0].replace('1450','1200');svg[1]=svg[1].replace('1450','1200');gm.set('pageHeight','1200');diag.set('name','Sketch comparison: simplified implementation')
def link(a,b,p):edge(a,b,p,animate=False)
def dot(id,x,y,r,fill,stroke):
 svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>');vert(id,'',x-r,y-r,2*r,2*r,f'ellipse;fillColor={fill};strokeColor={stroke};strokeWidth=2;')
def mlp(id,x,y,w=210):
 vert(id,'',x,y,w,100,'container=1;pointerEvents=0;fillColor=none;strokeColor=none;')
 cols=[x+16,x+w/2,x+w-16]
 for k in range(2):
  for a in [y+15,y+50,y+85]:
   for b in [y+15,y+50,y+85]:
    svg.append(f'<path d="M{cols[k]},{a} L{cols[k+1]},{b}" stroke="#d7a6a1" stroke-width="2"/>')
    c=E.SubElement(xr,'mxCell',id=f'{id}-wire-{k}-{a}-{b}',edge='1',parent='1',style='endArrow=none;strokeColor=#d7a6a1;strokeWidth=2;');g=E.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'});E.SubElement(g,'mxPoint',x=str(cols[k]),y=str(a),attrib={'as':'sourcePoint'});E.SubElement(g,'mxPoint',x=str(cols[k+1]),y=str(b),attrib={'as':'targetPoint'})
 for k,xx in enumerate(cols):
  for j,yy in enumerate([y+15,y+50,y+85]):dot(f'{id}-n-{k}-{j}',xx,yy,13,'#f8cecc','#b85450')
 text('MLP',x+w/2,y-40 if id=='final' else y+143,29,True,'middle');badge('fire-'+id,'trained',x+w-30,y-45)
def feat(id,x,y,label,purple=False):
 vector(id,x,y,240,45,label=label)
 if purple:
  for i in range(len(svg)):
   if i>=len(svg)-7:svg[i]=svg[i].replace('#85b4dc','#b09ccc').replace('#d8e8f8','#e8dff1').replace('#5093c7','#8064a2').replace('#4b88b5','#8064a2')
  for cell in xr:
   if cell.get('parent')==id:cell.set('style',cell.get('style').replace('#85b4dc','#b09ccc').replace('#d8e8f8','#e8dff1').replace('#5093c7','#8064a2').replace('#4b88b5','#8064a2'))
text('Your sketch, mapped to the implementation',65,75,40,True)
text('Two streams • MLPs with skip connections • learned RoI features • final MLP',65,125,27)
box('text-tower',70,290,210,120,['L₀'],kind='frozen',size=38);badge('ice-text','frozen',245,270);text('Text encoder',175,455,26,False,'middle')
box('video-tower',70,790,210,120,['V₀'],kind='frozen',size=38);badge('ice-video','frozen',245,770);text('Video encoder',175,965,26,False,'middle')
text('Label phrases',70,255,25);text('“car turning left”',65,195,23,color='#8064a2');text('Scene + RoI + crop',65,755,25)
for j in range(3):pic(f'clip-preview-{j}',roaduri,75+j*48,620+j*18,105,75,'1')
feat('textfeat',390,327,'Text features',True);feat('videofeat',390,827,'Visual features')
mlp('textmlp',790,300);mlp('videomlp',790,800)
for id,y in [('textplus',325),('videoplus',825)]:box(id,1090,y,50,50,['+'],fill='#fff2cc',stroke='#d6b656',size=31)
vert('fusion','',1260,520,220,160,'fillColor=none;strokeColor=none;');dot('fusion-a',1335,600,55,'#e1d5e7','#8064a2');dot('fusion-b',1395,600,55,'#dae8fc','#4b88b5');text('Attention',1230,735,28,True,'middle');badge('fire-fusion','trained',1435,500)
# Same grid idea as the hand drawing, schematic cells rather than dimensions.
vert('roi','',1590,520,230,160,'container=1;pointerEvents=0;fillColor=none;strokeColor=none;')
for row in range(4):
 for col in range(6):
  x=1590+col*230/6;y=520+row*40;fill=['#85b4dc','#d8e8f8','#5093c7'][(row+col)%3]
  svg.append(f'<rect x="{x}" y="{y}" width="{230/6}" height="40" fill="{fill}" stroke="#4b88b5" stroke-width="1.5"/>');vert(f'roi-{row}-{col}','',x,y,230/6,40,f'fillColor={fill};strokeColor=#4b88b5;',parent='roi')
text('RoI features',1705,485,29,True,'middle')
mlp('final',1920,550,180);vert('out','',2190,550,150,100,'fillColor=none;strokeColor=none;');
for j,h in enumerate([40,75,25,90,60,45]):
 x=2195+j*24;svg.append(f'<rect x="{x}" y="{650-h}" width="17" height="{h}" fill="#55969e"/>');vert(f'bar-{j}','',x,650-h,17,h,'fillColor=#55969e;strokeColor=none;')
text('184',2265,525,29,True,'middle')
text('Predictions',2265,735,25,False,'middle')
vert('contrast','',1260,990,420,95,'fillColor=none;strokeColor=none;');dot('contrast-v',1350,1025,23,'#dae8fc','#4b88b5');dot('contrast-t',1530,1025,23,'#e1d5e7','#8064a2');text('↔',1440,1037,40,True,'middle');text('Contrastive learning',1450,1100,26,True,'middle')
for a,b,pts in [('text-tower','textfeat',[(280,350),(390,350)]),('video-tower','videofeat',[(280,850),(390,850)]),('textfeat','textmlp',[(630,350),(790,350)]),('videofeat','videomlp',[(630,850),(790,850)]),('textmlp','textplus',[(1000,350),(1090,350)]),('videomlp','videoplus',[(1000,850),(1090,850)]),('textfeat','textplus',[(630,350),(710,350),(710,220),(1115,220),(1115,325)]),('videofeat','videoplus',[(630,850),(710,850),(710,960),(1115,960),(1115,875)]),('textplus','fusion',[(1140,350),(1370,350),(1370,520)]),('videoplus','fusion',[(1140,850),(1370,850),(1370,680)]),('fusion','roi',[(1480,600),(1590,600)]),('roi','final',[(1820,600),(1920,600)]),('final','out',[(2100,600),(2190,600)]),('videoplus','final',[(1140,850),(2010,850),(2010,650)]),('videoplus','contrast',[(1115,875),(1115,1037),(1260,1037)])]:link(a,b,pts)
text('Skip',870,201,24,False,'middle');text('Skip',870,1003,24,False,'middle');text('Visual skip',1880,823,24,False,'middle')
text('Visual branch includes context fusion; grid is schematic. Full implementation differences are below.',65,1135,24,color='#65717b')
badge('legend-ice','frozen',65,1150,30);text('Frozen encoders',110,1175,23);badge('legend-fire','trained',430,1150,30);text('Trainable networks',475,1175,23)
raw=''.join(svg)+'</svg>';(A/'sketch-simple.svg').write_text(raw);E.indent(root);E.ElementTree(root).write(A/'sketch-simple.drawio',encoding='utf-8',xml_declaration=True)
page='''<!doctype html><html><head><meta charset="utf-8"><title>Architecture sketch comparison</title><style>body{margin:0;font:20px Arial;color:#26323d;background:#eef1f5}main{max-width:1600px;margin:auto;background:white}svg{width:100%;height:auto}section{padding:10px 45px 40px;line-height:1.5}li{margin:12px 0}h2{font-size:25px}</style></head><body><main>'''+raw+'''<section><h2>How close is it to your drawing?</h2><p>The two streams, residual MLPs, central RoI features, visual skip, and final MLP follow your sketch. It is not an exact implementation of the sketch.</p><ul><li><b>Fusion:</b> the implemented model uses attention to mix the streams, rather than a joint MLP before the RoI features.</li><li><b>Contrastive learning:</b> supervises the visual RoI before text is mixed in. The central grid above is the joint, language-conditioned RoI used for classification.</li><li><b>Final text pathway:</b> the final MLP also receives visual–text cosine scores. It does not receive the raw text feature vector through a direct skip. That auxiliary scoring branch is omitted above to keep the sketch readable.</li><li><b>Video side:</b> the visual MLP block summarizes residual adaptation and context fusion. Scene and crop clips use separate passes through the same frozen encoder; RoIAlign and box position features are added downstream.</li></ul><p>The grid is a schematic RoI representation, not a spatial feature map. This simplified view does not change the experiment.</p></section></main></body></html>'''
(A/'sketch-simple.html').write_text(page)

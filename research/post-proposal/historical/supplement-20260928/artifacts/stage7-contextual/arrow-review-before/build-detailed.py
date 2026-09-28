from pathlib import Path
import hashlib
A=Path(__file__).resolve().parent
source=A.parent/'animated-stage5-composition/build.py'
helpers=source.read_text().split("text('Stage 5: crop features")[0].replace('start,duration=schedule[(a,b)]','start,duration=phase')
exec(compile(helpers,str(source),'exec'))
H=2100
svg[0]=svg[0].replace('1450','2100');svg[1]=svg[1].replace('1450','2100');gm.set('pageHeight','2100');diag.set('name','Stage 7 detailed (proposed)')
def vector(id,x,y,w=240,h=40,label='',purple=False,rows=1):
 colors=['#b09ccc','#e8dff1','#8064a2'] if purple else ['#85b4dc','#d8e8f8','#5093c7'];stroke='#8064a2' if purple else '#4b88b5'
 vert(id,'',x,y,w,h,'container=1;pointerEvents=0;fillColor=none;strokeColor=none;')
 for row in range(rows):
  for j in range(6):
   xx=x+j*w/6;yy=y+row*h/rows;f=colors[(j+row)%3];svg.append(f'<rect x="{xx}" y="{yy}" width="{w/6}" height="{h/rows}" fill="{f}" stroke="{stroke}" stroke-width="1.5"/>');vert(f'{id}-{row}-{j}','',xx,yy,w/6-.01,h/rows-.01,f'fillColor={f};strokeColor={stroke};',id)
 if label:text(label,x+w/2,y-18,23,True,'middle')
def mlp(id,x,y,w=200,label='MLP',frozen=False):
 box(id,x,y,w,100,fill='#fff7f6',stroke='#b85450',kind='frozen' if frozen else 'trained')
 cols=[x+20,x+w/2,x+w-20]
 for k in range(2):
  for a in [y+20,y+50,y+80]:
   for b in [y+20,y+50,y+80]:
    svg.append(f'<path d="M{cols[k]},{a} L{cols[k+1]},{b}" stroke="#d7a6a1" stroke-width="1.5"/>');c=E.SubElement(xr,'mxCell',id=f'{id}-w-{k}-{a}-{b}',edge='1',parent='1',style='endArrow=none;strokeColor=#d7a6a1;');g=E.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'});E.SubElement(g,'mxPoint',x=str(cols[k]),y=str(a),attrib={'as':'sourcePoint'});E.SubElement(g,'mxPoint',x=str(cols[k+1]),y=str(b),attrib={'as':'targetPoint'})
 for k,xx in enumerate(cols):
  for j,yy in enumerate([y+20,y+50,y+80]):
   svg.append(f'<circle cx="{xx}" cy="{yy}" r="11" fill="#f8cecc" stroke="#b85450" stroke-width="2"/>');vert(f'{id}-n{k}-{j}','',xx-11,yy-11,22,22,'ellipse;fillColor=#f8cecc;strokeColor=#b85450;strokeWidth=2;')
 text(label,x+w/2,y-23,25,True,'middle');badge('badge-'+id,'frozen' if frozen else 'trained',x+w-24,y-22)
def flow(a,b,p,t,d=.8):edge(a,b,p,phase=(t,d))
text('Stage 7: Contextual refinement + Stage 5',45,60,41,True)
text('Proposed architecture • one refined crop feature feeds both Stage 5 branches • frozen heads in the first phase',45,106,25)
for id,x,w,title in [('p1',40,720,'1. Candidate detection'),('p2',830,710,'2. Frozen visual encoder'),('p3',1610,750,'3. Score assembly')]:
 box(id,x,165,w,495,fill='white',stroke='#26323d',container=True);text(title,x+18,146,29,True)
pic('frame',roaduri,75,255,240,162,'1');box('yolo',430,260,275,100,['YOLO detector'],fill='#dae8fc',stroke='#6c8ebf',kind='frozen');badge('badge-yolo','frozen',675,240)
box('candidates',425,470,285,95,['Candidate boxes','Confidence q','Agent class'],fill='#e5f0d8',stroke='#7e9f59',size=22)
text('Annotated crop illustration, not a detector prediction',65,623,21)
box('clip-input',865,235,235,115,['Whole-scene clip','+ actor crop clip','8 frames each'],fill='#d5e8d4',stroke='#82b366',size=22)
box('video-tower',1180,250,295,110,['InternVideo2-CLIP-S','V₀'],kind='frozen',size=25);badge('badge-video','frozen',1445,230)
box('cache',870,450,630,100,['Crop features [1024] + context RoI [1024]','Scene tokens [16 × 1024] + box geometry [8]'],fill='#e8f1f9',stroke='#4b88b5',size=23)
text('Shared frozen encoder; separate scene / crop passes',870,593,22);text('RoIAlign on scene tokens; geometry added afterward',870,626,22)
box('detector',1640,210,275,85,['Candidate boxes','Confidence q','Agent class'],fill='#e5f0d8',stroke='#7e9f59',size=22)
box('linear-final',1640,330,275,75,['Linear scores','22 action + 16 location'],fill='#e3f3f5',stroke='#55969e',size=22)
box('comp-final',1640,445,275,85,['Composition scores','49 duplex + 86 triplet'],fill='#eee8f6',stroke='#9478b5',size=22)
box('assembly',2030,210,280,320,['Confidence weighting','q × linear scores','q × composition scores'],fill='#fff5d9',stroke='#c4a756',size=22)
box('output',2000,580,325,62,['Candidate boxes','+ 184 scores'],fill='#e5f0d8',stroke='#7e9f59',size=21)
text('YOLO agentness and agent scores are retained',1640,562,20)
box('context-panel',40,760,2320,650,fill='white',stroke='#26323d',container=True);text('Contextual RoI: visual and text adaptation, then language attention',65,737,30,True)
box('text-tower',90,875,210,100,['L₀'],kind='frozen',size=36);badge('badge-text','frozen',270,852);text('Text encoder',195,1020,23,False,'middle');text('All 184 label phrases',80,832,24)
vector('fixedbank',400,900,220,70,'Phrase matrix P',True,3);text('184 × 512',510,1007,22,False,'middle')
mlp('textmlp',730,885,200);box('textplus',990,910,48,48,['+'],fill='#fff2cc',stroke='#d6b656',size=30);vector('adapted',1120,912,240,70,'Phrase matrix T',True,3)
box('visual-inputs',80,1155,310,115,['Crop + context RoI','Scene tokens + geometry'],fill='#e8f1f9',stroke='#4b88b5',size=24)
mlp('visualmlp',530,1160,235);text('Residual visual adaptation',647,1320,21,False,'middle');text('+ context fusion',647,1350,21,False,'middle')
vector('visual',930,1190,240,45,'Visual RoI v [512]')
box('attn',1450,1025,255,115,['Language attention','+ residual / LN'],fill='#e1d5e7',stroke='#8064a2',kind='trained',size=25);badge('badge-attn','trained',1670,1005)
vector('joint',1850,1040,240,120,'Joint RoI h [512]',rows=3);text('Schematic grid',1970,1200,22,False,'middle')
box('loss',1210,1300,420,65,['Visual contrastive loss','86 triplets against fixed P'],fill='#fff2cc',stroke='#d6b656',size=22)
mlp('bridge',1880,1280,230);text('512 → 512 → 1024',2230,1400,20,False,'middle');text('Zero-initialized',2230,1310,20,False,'middle');text('output layer',2230,1345,20,False,'middle')
box('head-panel',40,1530,2320,470,fill='white',stroke='#26323d',container=True);text('Stage 5 readout: same two heads, same refined crop features',65,1506,30,True)
vector('original',85,1660,230,40,'Crop features x [1024]')
box('sum',405,1655,50,50,['+'],fill='#fff2cc',stroke='#d6b656',size=31)
vector('enriched',550,1660,240,40,'Refined crop features x′')
box('linear',920,1620,280,100,['Linear scoring head','1024 → 184 logits'],fill='#fcebea',stroke='#b85450',kind='frozen',size=24);badge('badge-linear','frozen',1170,1600)
box('linear-scores',1360,1580,350,70,['Linear scores','22 action + 16 location'],fill='#e3f3f5',stroke='#55969e',size=22)
box('primitive',920,1790,280,85,['Primitive scores [49]','sigmoid(first 49 logits)'],fill='#e3f3f5',stroke='#55969e',size=22)
box('concat',1360,1790,280,85,['Concatenate','49 + 1024 = 1073'],fill='#fff2cc',stroke='#d6b656',size=24)
mlp('comp',1780,1780,230,'Composition MLP',True);text('1073 → 512 → 135',1895,1920,22,False,'middle')
box('composition',2100,1770,220,110,['Composition scores','49 duplex','86 triplet'],fill='#eee8f6',stroke='#9478b5',size=23)
text('x′ = x + MLP(h)',320,1840,26,True,'middle');text('One 1-D vector per crop sequence',425,1937,24,False,'middle')
text('Frozen heads still pass gradients into their inputs. Gold dots illustrate forward features, not gradients.',55,2047,24)
badge('legend-ice','frozen',55,2060,30);text('Frozen weights',96,2086,23);badge('legend-fire','trained',355,2060,30);text('Trainable weights',396,2086,23);text('Proposed integration; no Stage 7 result claimed',1050,2086,23)
# Sequential phases. The two branches may travel together, never more than two packets.
for a,b,p,t,d in [
 ('frame','yolo',[(315,320),(430,320)],.3,.8),('yolo','candidates',[(565,360),(565,470)],1.6,.8),('p1','p2',[(760,400),(830,400)],2.7,.8),
 ('clip-input','video-tower',[(1100,300),(1180,300)],3.7,.8),('video-tower','cache',[(1320,360),(1320,450)],5,.8),('p2','context-panel',[(1185,660),(1185,760)],6,.8),
 ('text-tower','fixedbank',[(300,925),(400,925)],7,.8),('fixedbank','textmlp',[(620,925),(730,925)],8,.8),('fixedbank','textplus',[(620,940),(665,940),(665,810),(1014,810),(1014,910)],8,2),
 ('textmlp','textplus',[(930,935),(990,935)],9.2,.8),('textplus','adapted',[(1038,935),(1120,935)],10.2,.8),
 ('visual-inputs','visualmlp',[(390,1210),(530,1210)],11.2,.8),('visualmlp','visual',[(765,1210),(930,1210)],12.6,.8),
 ('visual','loss',[(1050,1235),(1050,1332),(1210,1332)],13.6,1),('fixedbank','loss',[(440,970),(440,1387),(1420,1387),(1420,1365)],13.6,1.5),
 ('visual','attn',[(1170,1212),(1400,1212),(1400,1100),(1450,1100)],15.4,1),('adapted','attn',[(1360,935),(1575,935),(1575,1025)],15.4,1),('attn','joint',[(1705,1080),(1850,1080)],17,.8),
 ('joint','bridge',[(1850,1120),(1780,1120),(1780,1330),(1880,1330)],18.1,.8),('bridge','sum',[(1995,1380),(1995,1520),(430,1520),(430,1655)],19.4,1.8),('original','sum',[(315,1680),(405,1680)],19.4,1.8),
 ('sum','enriched',[(455,1680),(550,1680)],21.5,.8),('enriched','linear',[(790,1675),(920,1675)],22.6,.8),('enriched','concat',[(670,1700),(670,1955),(1500,1955),(1500,1875)],22.6,2.4),
 ('linear','primitive',[(1060,1720),(1060,1790)],25.2,.8),('linear','linear-scores',[(1200,1640),(1280,1640),(1280,1615),(1360,1615)],25.2,.8),
 ('primitive','concat',[(1200,1830),(1360,1830)],26.3,.8),('concat','comp',[(1640,1830),(1780,1830)],27.4,.8),('comp','composition',[(2010,1830),(2100,1830)],28.8,.8),
 ('head-panel','p3',[(2350,1530),(2385,1530),(2385,700),(2250,700),(2250,660)],30,1.4),
 ('detector','assembly',[(1915,252),(2030,252)],31.8,.8),('linear-final','assembly',[(1915,367),(2030,367)],33,.8),('comp-final','assembly',[(1915,487),(2030,487)],33,.8),('assembly','output',[(2170,530),(2170,580)],34.2,.8)
]:flow(a,b,p,t,d)
# Nest network neurons in their editable module, not overlapping sibling shapes.
for c in list(xr):
 for module in ['textmlp','visualmlp','bridge','comp']:
  if c.get('id','').startswith(module+'-n'):
   g=c.find('mxGeometry');mx,my,_,_=positions[module];c.set('parent',module);g.set('x',str(float(g.get('x'))-mx));g.set('y',str(float(g.get('y'))-my))
# Native grouping of entirely contained vertices, preserving tensor children.
for c in list(xr):
 g=c.find('mxGeometry')
 if c.get('vertex')!='1' or c.get('parent')!='1' or g is None or c.get('id') in ['p1','p2','p3','context-panel','head-panel']:continue
 x,y,w,h=[float(g.get(k,0)) for k in ['x','y','width','height']]
 for parent in ['p1','p2','p3','context-panel','head-panel']:
  px,py,pw,ph=positions[parent]
  if x>=px and y>=py and x+w<=px+pw and y+h<=py+ph:c.set('parent',parent);g.set('x',str(x-px));g.set('y',str(y-py));break
static=''.join(svg)+'</svg>';(A/'stage7-detailed-static.svg').write_text(static);E.indent(root);E.ElementTree(root).write(A/'stage7-detailed.drawio',encoding='utf-8',xml_declaration=True)
D=37
svg.append('<defs><filter id="glow" x="-90%" y="-90%" width="280%" height="280%"><feGaussianBlur stdDeviation="4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>')
def pulse(t,e):return f'<animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;{t/D};{(t+.1)/D};{(e-.1)/D};{e/D};1" dur="{D}s" repeatCount="indefinite"/>'
windows={'yolo':(1.1,1.6),'video':(4.5,5),'text':(6.8,7.5),'textmlp':(8.8,9.2),'visualmlp':(12,12.6),'attn':(16.4,17),'bridge':(18.9,19.4),'linear':(23.4,25.2),'comp':(28.2,28.8)}
for b in badges:
 name=b['id'].replace('badge-','')
 if name not in windows:continue
 t,e=windows[name];svg.append(f'<circle cx="{b["x"]+b["size"]/2}" cy="{b["y"]+b["size"]/2}" r="23" fill="none" stroke="'+('#39a8ed' if b['kind']=='frozen' else '#ff981e')+f'" stroke-width="5" filter="url(#glow)" opacity="0">{pulse(t,e)}</circle>')
for b in trained:
 t,e=windows[b['id']];svg.append(f'<rect x="{b["x"]}" y="{b["y"]}" width="{b["w"]}" height="{b["h"]}" rx="15" fill="none" stroke="#ee7855" stroke-width="5" filter="url(#glow)" opacity="0">{pulse(t,e)}</rect>')
for id,t,e in [('p1',.3,2.5),('p2',3.7,5.8),('context-panel',7,19.4),('head-panel',19.4,29.6),('p3',31.8,35)]:
 x,y,w,h=positions[id];svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="15" fill="none" stroke="#e9ad35" stroke-width="4" filter="url(#glow)" opacity="0">{pulse(t,e)}</rect>')
for r in routes:
 t=r['phase']*1.5;e=t+r['duration'];p=rounded_route(r['points'])[0]
 svg.append(f'<path d="{p}" fill="none" stroke="#ff981e" stroke-width="4" opacity="0" filter="url(#glow)">{pulse(t,e)}</path><circle r="8" fill="#FDB927" stroke="#fff2be" stroke-width="2" filter="url(#glow)" opacity="0">{pulse(t,e)}<animateMotion dur="{D}s" repeatCount="indefinite" keyTimes="0;{t/D};{e/D};1" keyPoints="0;0;1;1" calcMode="linear" path="{p}"/></circle>')
(A/'stage7-detailed-animated.svg').write_text(''.join(svg)+'</svg>')
assert max(sum(r['phase']*1.5<=t<r['phase']*1.5+r['duration'] for r in routes) for t in [i/100 for i in range(D*100)])<=2
(A/'animation-check.json').write_text(json.dumps({'duration':D,'max_simultaneous_packets':2,'routes':routes},indent=2))
print('Detailed figure built')

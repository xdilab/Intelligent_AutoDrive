"""Stage 5 composition-head architecture in the animated stage house style."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from diagram_geometry import rounded_route
from PIL import Image,ImageDraw,ImageOps
import cv2,json,base64,io,html,xml.etree.ElementTree as E
from urllib.parse import quote
A=Path(__file__).resolve().parent;W,H=2400,1450
def uri(im):
 b=io.BytesIO();im.save(b,format='PNG');return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
pretraining_images=[uri(Image.open(A/'assets'/name)) for name in ['pretraining-crossing.jpg','pretraining-road.jpg']]
roaduri=uri(Image.open(A/'assets/road-tail-crop.png'))
svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2400 1450" width="2400" height="1450">','<rect width="2400" height="1450" fill="white"/>','<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto"><path d="M0,1 L9,5 L0,9 Z" fill="#26323d"/></marker></defs>']
root=E.Element('mxfile',host='app.diagrams.net');diag=E.SubElement(root,'diagram',name='Stage 5 composition scoring');gm=E.SubElement(diag,'mxGraphModel',page='1',pageWidth=str(W),pageHeight=str(H),background='#ffffff');xr=E.SubElement(gm,'root');E.SubElement(xr,'mxCell',id='0');E.SubElement(xr,'mxCell',id='1',parent='0')
positions={};counter=0;routes=[];badges=[];trained=[]
ENCODERS={'v0','l0','vstar','lstar','vfrozen','lfrozen','video-tower','text-tower','legend-v0','legend-vstar','legend-l'}
def vert(id,value,x,y,w,h,style,parent='1'):
 px,py=(positions[parent][:2] if parent!='1' else (0,0));positions[id]=(x,y,w,h)
 c=E.SubElement(xr,'mxCell',id=id,value=value,style=style,vertex='1',parent=parent);E.SubElement(c,'mxGeometry',x=str(x-px),y=str(y-py),width=str(w),height=str(h),attrib={'as':'geometry'});return id
def text(value,x,y,size=25,bold=False,anchor='start',parent='1',color='#26323d',width=None):
 global counter
 counter+=1;svg.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Arial,Helvetica,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{html.escape(value)}</text>')
 w=width or max(20,len(value)*size*.55);xx=x-w/2 if anchor=='middle' else x
 return vert(f't{counter}',value,xx,y-size,w,size,f'text;html=1;align={"center" if anchor=="middle" else "left"};fontFamily=Helvetica;fontSize={size};fontStyle={1 if bold else 0};fontColor={color};strokeColor=none;fillColor=none;',parent)
def box(id,x,y,w,h,lines=(),fill='#f3f5f7',stroke='#34414d',kind='plain',parent='1',size=25,container=False):
 if id in {'features','x','phrasebank','fixed-p'}:
  phrase=id in {'phrasebank','fixed-p'}
  colors=['#b09ccc','#e8dff1','#8064a2'] if phrase else ['#85b4dc','#d8e8f8','#5093c7']
  line='#8064a2' if phrase else '#4b88b5'
  vert(id,'',x,y,w,h,'container=1;pointerEvents=0;fillColor=none;strokeColor=none;',parent)
  rows=1 if id in {'x','features'} else 3
  for row in range(rows):
   for col in range(6):
    xx=x+col*w/6;yy=y+row*h/rows;fill=colors[(col+row*2)%3]
    svg.append(f'<rect x="{xx}" y="{yy}" width="{w/6}" height="{h/rows}" fill="{fill}" stroke="{line}" stroke-width="1.5"/>')
    vert(f'{id}-tile-{row}-{col}','',xx,yy,w/6-.01,h/rows-.01,f'rounded=0;fillColor={fill};strokeColor={line};strokeWidth=1;',id)
  title=('Fixed phrase matrix P' if id=='fixed-p' else 'Phrase matrix') if phrase else ('Crop features [1024]' if id=='x' else 'Crop features [1024]')
  dims='184 × 512' if phrase else 'One sequence: [1024]'
  text(title,x+w/2,y-15,23,True,'middle',parent)
  if id!='x':text(dims,x+w/2,y+h+31,22,False,'middle',parent)
  return id
 if id in ENCODERS or id in {'v-early','v-last','l-early','l-last'}:
  visual=id.startswith('v') or id in {'legend-v0','legend-vstar'}
  fill,stroke=('#ecf6f4','#0e8a7d') if visual else ('#f1edf7','#8064a2')
 if kind=='trained':trained.append({'id':id,'x':x,'y':y,'w':w,'h':h,'stroke':stroke,'polygon':id in ENCODERS})
 dash=' stroke-dasharray="8 6"' if kind=='frozen' else '';sw=4 if kind=='trained' else 2
 if id in ENCODERS:
  pad=min(25,w*.18)
  svg.append(f'<polygon points="{x+pad},{y} {x+w},{y} {x+w-pad},{y+h} {x},{y+h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash}/>')
 else:svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="15" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash}/>')
 vert(id,'<br>'.join(lines),x,y,w,h,(f'shape=parallelogram;perimeter=parallelogramPerimeter;fixedSize=1;size={min(25,w*.18)};' if id in ENCODERS else 'rounded=1;arcSize=8;')+f'whiteSpace=wrap;html=1;fontFamily=Helvetica;fontSize={size};fontColor=#26323d;fillColor={fill};strokeColor={stroke};strokeWidth={sw};'+('dashed=1;dashPattern=8 6;' if kind=='frozen' else '')+('container=1;pointerEvents=0;' if container else ''),parent)
 for j,line in enumerate(lines):
  yy=y+h/2-(len(lines)-1)*size*.62+j*size*1.24+size*.32
  svg.append(f'<text x="{x+w/2}" y="{yy}" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="{size}" fill="#26323d">{html.escape(line)}</text>')
 return id
def badge(id,kind,x,y,size=40,parent='1'):
 raw=(A/'assets'/('ice.svg' if kind=='frozen' else 'fire.svg')).read_text()
 u='data:image/svg+xml,'+quote(raw,safe='')
 badges.append({'id':id,'index':len(svg),'x':x,'y':y,'size':size,'kind':kind})
 svg.append(f'<image href="{html.escape(u)}" x="{x}" y="{y}" width="{size}" height="{size}"/>')
 vert(id,'',x,y,size,size,'shape=image;imageAspect=0;aspect=fixed;image='+u+';',parent)
def pic(id,u,x,y,w,h,parent):
 svg.append(f'<image href="{u}" x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid slice"/>')
 return vert(id,'<img src="'+u+'" width="'+str(w)+'" height="'+str(h)+'">',x,y,w,h,'html=1;whiteSpace=wrap;fillColor=none;strokeColor=none;spacing=0;',parent)
def edge(a,b,points,phase=0,parent='1',animate=True,dashed=False):
 global counter
 counter+=1;original=[list(p) for p in points];points=[list(p) for p in points]
 for idx,node in [(0,a),(-1,b)]:
  if node in ENCODERS:
   xx,yy,ww,hh=positions[node];shift=min(25,ww*.18);q=(points[idx][1]-yy)/hh
   if abs(points[idx][0]-xx)<.01:points[idx][0]+=shift*(1-q)
   elif abs(points[idx][0]-xx-ww)<.01:points[idx][0]-=shift*q
 path=rounded_route(points)[0]
 svg.append(f'<path d="{path}" fill="none" stroke="#26323d" stroke-width="2.5" stroke-linejoin="round" marker-end="url(#arrow)"'+(' stroke-dasharray="7 5"' if dashed else '')+'/>')
 if animate:
  schedule={
   ('frame','yolo'):(0,.6),('yolo','candidates'):(.7,.6),
   ('p1','p2'):(1.5,.9),('road-crop','vstar'):(2.6,.9),('vstar','features'):(3.7,.9),
   ('p2','detail'):(4.8,1.0),
   ('x','flat'):(6,.9),('x','concat'):(6,2.1),('flat','concat'):(7.2,.9),
   ('flat','flat-scores'):(8.2,.9),('concat','mlp'):(8.3,.9),('mlp','comps'):(9.5,.9),('detail','p3'):(10.8,1.0),
   ('detector','assembly'):(12,.6),('flat-input','assembly'):(12.8,.9),('comp-input','assembly'):(12.8,.9),('assembly','out'):(14,.9),
  }
  start,duration=schedule[(a,b)]
  routes.append({'points':points,'phase':start/1.5,'duration':duration,'panel':parent,'source':a,'target':b})
 ax,ay,aw,ah=positions[a];bx,by,bw,bh=positions[b];px,py=0,0
 ex,ey=(original[0][0]-ax)/aw,(original[0][1]-ay)/ah;ix,iy=(original[-1][0]-bx)/bw,(original[-1][1]-by)/bh
 st=f'edgeStyle=orthogonalEdgeStyle;rounded=1;arcSize=24;jettySize=20;html=1;endArrow=block;strokeColor=#26323d;strokeWidth=2;exitX={ex};exitY={ey};entryX={ix};entryY={iy};'+('dashed=1;' if dashed else '')
 c=E.SubElement(xr,'mxCell',id=f'e{counter}',edge='1',parent='1',source=a,target=b,style=st);geo=E.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'});ar=E.SubElement(geo,'Array',attrib={'as':'points'})
 for x,y in points[1:-1]:E.SubElement(ar,'mxPoint',x=str(x-px),y=str(y-py))
text('Stage 5: crop features with composition learning',45,63,43,True)
text('YOLO candidates → frozen InternVideo2 features → flat predictions + composition learning',45,105,26)
for id,x,w,title in [('p1',40,720,'1. YOLO candidate detection'),('p2',830,710,'2. InternVideo2-CLIP-S'),('p3',1610,750,'3. Score assembly')]:
 box(id,x,165,w,495,fill='#ffffff',stroke='#26323d',container=True)
 text(title,x+20,147,31,True)
# YOLO has its own major panel.
pic('frame',roaduri,75,245,240,162,'p1')
text('ROAD-Waymo example',195,215,24,True,'middle','p1')
box('yolo',410,275,275,110,['YOLO detector','Frozen at head training'],kind='frozen',parent='p1',size=24)
edge('frame','yolo',[(315,325),(410,325)],0,'p1')
box('candidates',410,465,275,90,['Candidate boxes','Confidence q','Agent class'],fill='#e5f0d8',stroke='#7e9f59',parent='p1',size=22)
edge('yolo','candidates',[(547,385),(547,465)],0.47,'p1')
text('Shared candidate boxes for downstream scoring',400,602,24,True,'middle','p1')
text('Image shows an annotated example, not a measured prediction.',400,641,21,False,'middle','p1')
# Frozen video feature extraction; no text tower is used in Stage 5.
pic('road-crop',roaduri,860,290,220,148,'p2')
text('GT crop illustration; YOLO crops at inference',1185,209,24,True,'middle','p2')
box('vstar',1150,320,95,85,['V₀'],kind='frozen',parent='p2',size=34)
box('features',1320,344,180,40,['Crop feature','1,024-D'],parent='p2',size=24)
edge('road-crop','vstar',[(1080,364),(1150,364)],2,'p2')
edge('vstar','features',[(1245,364),(1320,364)],3,'p2')
text('Eight frames • 2× padded spatial region',1185,503,24,True,'middle','p2')
text('224 × 224 crops; keyframe token mean',1185,550,24,False,'middle','p2')
text('Released visual encoder; cached frozen features',1185,597,23,False,'middle','p2')
text('Contextual frame detection: t−3 through t+4',1185,640,23,False,'middle','p2')
edge('p1','p2',[(760,400),(830,400)],1)

# Panel 3: score assembly preserves the detector's outputs.
box('detector',1640,210,275,85,['Candidate boxes','Confidence q','Agent class'],fill='#e5f0d8',stroke='#7e9f59',parent='p3',size=22)
box('flat-input',1640,330,275,75,['Flat scores','22 action + 16 location'],fill='#e3f3f5',stroke='#55969e',parent='p3',size=22)
box('comp-input',1640,445,275,85,['Composition scores','49 duplex scores','86 triplet scores'],fill='#eee8f6',stroke='#9478b5',parent='p3',size=22)
box('assembly',2030,345,280,155,['Confidence weighting','q × flat scores','q × composition scores'],fill='#fff5d9',stroke='#c4a756',parent='p3',size=22)
box('out',2020,550,290,64,['Candidate boxes','+ event scores'],fill='#e5f0d8',stroke='#7e9f59',parent='p3',size=22)
edge('detector','assembly',[(1915,252),(2070,252),(2070,345)],parent='p3')
edge('flat-input','assembly',[(1915,367),(1990,367),(1990,410),(2030,410)],parent='p3')
edge('comp-input','assembly',[(1915,487),(1975,487),(1975,465),(2030,465)],parent='p3')
edge('assembly','out',[(2165,500),(2165,550)],parent='p3')
text('Agentness and agent labels retained from YOLO',1985,640,23,False,'middle','p3')
# Expanded composition model.
box('detail',300,765,2060,535,fill='#ffffff',stroke='#26323d',container=True)
text('Inside Stage 5: learn compositions from primitives and crop features',335,813,32,True,parent='detail')
edge('p2','detail',[(1185,660),(1185,765)])
edge('detail','p3',[(2250,765),(2250,660)])
box('legend-v0',55,850,130,65,['V₀'],size=34)
text('Visual encoder',40,955,23)
box('legend-boxes',45,995,25,25,fill='#e5f0d8',stroke='#7e9f59');text('Candidate boxes',85,1017,22)
box('legend-features',45,1040,25,25,fill='#85b4dc',stroke='#4b88b5');text('Crop features',85,1062,22)
badge('legend-ice','frozen',42,1120);text('Frozen',95,1149,23)
badge('legend-fire','trained',42,1190);text('Trained',95,1219,23)
box('x',335,950,230,40,['Crop feature x','1,024-D'],parent='detail',size=24)
box('flat',650,900,285,140,['Flat linear head','1,024 → 184','Sigmoid flat scores'],fill='#fcebea',stroke='#b85450',kind='trained',parent='detail',size=24)
box('flat-scores',1110,835,285,60,['Flat scores','22 action + 16 location'],fill='#e3f3f5',stroke='#55969e',parent='detail',size=22)
box('concat',1110,920,285,140,['Concatenate','49 primitives + x','1,073-D input'],parent='detail',size=24)
box('mlp',1550,920,310,140,['Composition MLP','1,073 → 512 → 135','ReLU hidden layer'],fill='#fcebea',stroke='#b85450',kind='trained',parent='detail',size=24)
box('comps',2000,920,315,140,['Composition scores','49 duplex scores','86 triplet scores'],fill='#eee8f6',stroke='#9478b5',parent='detail',size=25)
edge('x','flat',[(565,970),(650,970)],2,'detail')
edge('flat','concat',[(935,970),(1110,970)],3,'detail')
edge('flat','flat-scores',[(792,900),(792,865),(1110,865)],parent='detail')
text('First 49',1022,911,23,True,'middle','detail');text('ungated',1022,943,22,False,'middle','detail')
edge('x','concat',[(450,990),(450,1140),(1252,1140),(1252,1060)],3,'detail')
text('Crop features x',810,1120,23,False,'middle','detail')
edge('concat','mlp',[(1395,990),(1550,990)],4,'detail')
edge('mlp','comps',[(1860,990),(2000,990)],5,'detail')
text('Sigmoid',1930,970,21,False,'middle','detail')
text('Training: video-level two-fold out-of-fold primitive scores.',335,1210,25,True,parent='detail')
text('At inference: full-data flat head; composition scores replace its duplex/triplet outputs.',335,1260,24,parent='detail')
text('Action/location: YOLO confidence × flat score. Duplex/triplet: confidence × composition score.',335,1350,24)
text('Both heads use focal classification loss; the encoder remains frozen. No text tower in this stage.',335,1390,24)
text('The 49 primitives are agentness (1), agent (10), action (22), and location (16).',335,1428,23)
for node,kind in [('yolo','frozen'),('vstar','frozen'),('flat','trained'),('mlp','trained')]:
 x,y,w,h=positions[node];badge(node+'-badge',kind,x+w-22,y-20)
static=''.join(svg)+'</svg>';(A/'stage5-static.svg').write_text(static)
# Glow only during the matching flow phase; legend and idle icons stay unlit.
windows={'yolo-badge':(0,1.3),'vstar-badge':(2.6,4.6),'flat-badge':(6,8.1),'mlp-badge':(8.3,10.4)}
for b in badges:
 if b['id'] not in windows:continue
 start,end=windows[b['id']];b['window']=[start,end]
 i=b['index'];x=b['x']+b['size']/2;y=b['y']+b['size']/2
 color='#39a8ed' if b['kind']=='frozen' else '#ff981e'
 times=[0]+([start/18] if start else [])+[(start+.25)/18,(end-.25)/18,end/18,1]
 values=[0]+([0] if start else [])+[.85,.85,0,0]
 keytimes=';'.join(str(t) for t in times);vals=';'.join(str(v) for v in values)
 svg[i]=f'<g class="status-glow" data-badge="{b["id"]}"><circle cx="{x}" cy="{y}" r="{b["size"]*.52}" fill="{color}" filter="url(#badge-glow)" opacity="0"><animate attributeName="opacity" values="{vals}" keyTimes="{keytimes}" dur="18s" repeatCount="indefinite"/></circle><circle class="pulse-ring" cx="{x}" cy="{y}" r="{b["size"]*.5}" fill="none" stroke="{color}" stroke-width="2.5" opacity="0"><animate attributeName="opacity" values="{vals}" keyTimes="{keytimes}" dur="18s" repeatCount="indefinite"/><animate attributeName="r" values="{b["size"]*.48};{b["size"]*.75};{b["size"]*.48}" dur="1s" repeatCount="indefinite"/></circle>'+svg[i]+'</g>'

svg.append('<defs><filter id="badge-glow" x="-80%" y="-80%" width="260%" height="260%"><feGaussianBlur stdDeviation="4"/></filter></defs>')
(A/'animation-badges.json').write_text(json.dumps(badges,indent=2)+'\n')
# Only trainable model boundaries pulse; frozen layers retain their dashed borders.
svg.append('<defs><filter id="border-glow" x="-30%" y="-70%" width="160%" height="240%"><feGaussianBlur stdDeviation="4" result="halo"/><feMerge><feMergeNode in="halo"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>')
for b in trained:
 start,end=windows[b['id']+'-badge'];b['window']=[start,end]
 x,y,w,h=b['x'],b['y'],b['w'],b['h'];color={'#0e8a7d':'#14b8a6','#8064a2':'#a879e0','#b85450':'#e66a61'}[b['stroke']];b['glow_color']=color
 times=[0,start/18]+[(start+(end-start)*f)/18 for f in [.25,.5,.75,1]]+[1]
 animation='<animate attributeName="opacity" values="0;0;1;0.25;1;0;0" keyTimes="'+';'.join(map(str,times))+'" dur="18s" repeatCount="indefinite"/>'
 if b['polygon']:
  pad=min(25,w*.18);shape=f'<polygon points="{x+pad},{y} {x+w},{y} {x+w-pad},{y+h} {x},{y+h}"'
 else:shape=f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="15"'
 svg.append(f'<g class="trainable-border" data-node="{b["id"]}" opacity="0" fill="none" stroke="{color}" stroke-width="5" filter="url(#border-glow)">{animation}'+shape+'/></g>')
(A/'animation-borders.json').write_text(json.dumps(trained,indent=2)+'\n')
# Section emphasis is separate from the frozen/trainable model grammar.
panels=[]
for node,start,end in [('p1',0,1.3),('p2',2.6,4.6),('detail',6,10.4),('p3',12,14.9)]:
 x,y,w,h=positions[node]
 times=[0]+([start/18] if start else [])+[(start+.25)/18,(end-.25)/18,end/18,1]
 values=[0]+([0] if start else [])+[.8,.8,0,0]
 anim='<animate attributeName="opacity" values="'+';'.join(map(str,values))+'" keyTimes="'+';'.join(map(str,times))+'" dur="18s" repeatCount="indefinite"/>'
 svg.append(f'<rect class="section-glow" data-node="{node}" x="{x}" y="{y}" width="{w}" height="{h}" rx="15" fill="none" stroke="#e9ad35" stroke-width="3" opacity="0" filter="url(#border-glow)">{anim}</rect>')
 panels.append({'id':node,'x':x,'y':y,'w':w,'h':h,'window':[start,end]})
(A/'animation-panels.json').write_text(json.dumps(panels,indent=2)+'\n')


# Animated phase routes + positive-cell pulses; timeline is explanatory, not runtime.
svg.append('<defs><filter id="packet-glow" x="-200%" y="-200%" width="500%" height="500%"><feGaussianBlur stdDeviation="5" result="halo"/><feMerge><feMergeNode in="halo"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>')
# Active forward-flow edges heat up only while their packet is moving.
svg.append('<defs><marker id="hot-arrow" markerWidth="25" markerHeight="25" viewBox="0 0 10 10" refX="9" refY="5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,1 L9,5 L0,9 Z" fill="#ffab24"/></marker><filter id="arrow-heat" x="-30" y="-30" width="2460" height="1510" filterUnits="userSpaceOnUse"><feGaussianBlur stdDeviation="3" result="heat"/><feMerge><feMergeNode in="heat"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>')
for n,r in enumerate(routes):
 start=r['phase']*1.5;end=start+r['duration'];path=rounded_route(r['points'])[0]
 times=[0]+([start/18] if start else [])+[(start+.06)/18,(end-.06)/18,end/18,1]
 values=[0]+([0] if start else [])+[.95,.95,0,0]
 anim='<animate attributeName="opacity" values="'+';'.join(map(str,values))+'" keyTimes="'+';'.join(map(str,times))+'" dur="18s" repeatCount="indefinite"/>'
 svg.append(f'<path class="hot-flow-arrow" data-route="{n}" d="{path}" fill="none" stroke="#ff981e" stroke-width="4" stroke-linejoin="round" marker-end="url(#hot-arrow)" filter="url(#arrow-heat)" opacity="0">{anim}</path>')
for r in routes:
 start=r['phase']*1.5;duration=r['duration'];path=rounded_route(r['points'])[0]
 svg.append(f'<circle r="8" fill="#FDB927" stroke="#fff2be" stroke-width="2" filter="url(#packet-glow)"><animateMotion dur="18s" repeatCount="indefinite" keyTimes="0;{start/18:.5f};{(start+duration)/18:.5f};1" keyPoints="0;0;1;1" calcMode="linear" path="{path}"/><animate attributeName="opacity" dur="18s" repeatCount="indefinite" values="0;0;1;1;0;0" keyTimes="0;{start/18:.5f};{(start+.01)/18:.5f};{(start+duration-.05)/18:.5f};{(start+duration)/18:.5f};1"/></circle>')
animated=''.join(svg)+'</svg>';(A/'stage5-animated.svg').write_text(animated)
player='''<!doctype html><html><head><meta charset="utf-8"><title>Stage 5 composition classifier</title><style>body{margin:0;background:#eef1f5;font:16px Arial;color:#26323d}main{max-width:1900px;margin:auto}svg{display:block;width:100%;height:calc(100vh -64px);background:white}nav{display:flex;gap:18px;align-items:center;padding:12px 24px}button,select{padding:8px 14px;border:1px solid #bbc4ce;border-radius:6px;background:white}label{margin-left:auto}</style></head><body><main>'''+animated+'''<nav><button id="pause">Pause</button><button id="restart">Restart</button><span>1. YOLO → 2. InternVideo2 → lower detail → 3. Detection</span><label>Speed <select id="speed"><option value="0.5">0.5×</option><option selected value="1">1×</option><option value="1.5">1.5×</option></select></label></nav></main><script>const svg=document.querySelector('svg');let paused=false,speed=1,t=0,last=performance.now();svg.pauseAnimations();function run(now){if(!paused)t+=(now-last)/1000*speed;svg.setCurrentTime(t%18);last=now;requestAnimationFrame(run)}requestAnimationFrame(run);document.querySelector('#pause').onclick=()=>{paused=!paused;document.querySelector('#pause').textContent=paused?'Play':'Pause'};document.querySelector('#restart').onclick=()=>{t=0;svg.setCurrentTime(0)};document.querySelector('#speed').onchange=e=>speed=+e.target.value;</script></body></html>'''
(A/'stage5-animation.html').write_text(player);(A/'animation-routes.json').write_text(json.dumps(routes));E.indent(root);E.ElementTree(root).write(A/'stage5.drawio',encoding='utf-8',xml_declaration=True)

print(A)

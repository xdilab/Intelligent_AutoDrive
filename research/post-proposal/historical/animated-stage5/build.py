"""Reproducible abstract Stage 5 diagram, editable draw.io and animated SVG/HTML."""
from pathlib import Path
from PIL import Image, ImageDraw
import base64,io,json,xml.etree.ElementTree as E,html
A=Path(__file__).resolve().parent
W,H=1920,1080
P={'input':('#edf5ed','#82b366'),'frozen':('#edf4fc','#6c8ebf'),'op':('#fff7df','#d6b656'),'trained':('#fcebea','#b85450'),'out':('#f3f5f7','#909090')}
source=Path('/data/datasets/road_waymo/rgb-images/train_00407/00001.jpg')
im=Image.open(source).convert('RGB')
# Example crop comes from the documented GT box, for illustration only.
box=(547,755,1085,1214);cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2;bw=box[2]-box[0];bh=box[3]-box[1]
pad=(max(0,int(cx-bw)),max(0,int(cy-bh)),min(im.width,int(cx+bw)),min(im.height,int(cy+bh)))
def uri(img):
 b=io.BytesIO();img.save(b,format='PNG');return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
thumb=im.copy();d=ImageDraw.Draw(thumb);d.rectangle(box,outline='#f3bf36',width=14);thumb.thumbnail((430,280))
imguri=uri(thumb)
crops=[]
for f in [1,2,5]:
 x=Image.open(source.parent/f'{f:05d}.jpg').convert('RGB').crop(pad);x=x.resize((110,90));crops.append(uri(x))
N={
'input':(50,340,230,180,'input','plain',['Driving video']),
'detect':(360,350,200,140,'frozen','frozen',['YOLOv8x','detector']),
'crop':(650,350,230,140,'op','plain',['Fixed-window','video crop']),
'encoder':(980,350,240,140,'frozen','frozen',['InternVideo2','CLIP-S','video encoder']),
'flat':(1340,350,240,140,'trained','trained',['Flat scoring','head']),
'comp':(1340,700,240,140,'trained','trained',['Composition','multilayer','perceptron']),
'output':(1690,605,190,235,'out','plain',['Frame-level','detections'])}
# IDs, explicit paths, progressive animation phase, label placement.
ED=[
('input','detect',[(280,420),(360,420)],0,'Frame',(320,397)),
('detect','crop',[(560,420),(650,420)],1,'Box',(605,397)),
('input','crop',[(165,520),(165,610),(765,610),(765,490)],1,'Eight frames, same crop window',(490,641)),
('crop','encoder',[(880,420),(980,420)],2,'Crop clip',(930,397)),
('encoder','flat',[(1220,420),(1340,420)],3,'Feature',(1280,397)),
('encoder','comp',[(1100,490),(1100,770),(1340,770)],3,'Crop feature',(1200,802)),
('flat','comp',[(1460,490),(1460,700)],4,'Primitive scores',(1460,588)),
('flat','output',[(1580,420),(1630,420),(1630,665),(1690,665)],5,'Action + location',(1630,545)),
('comp','output',[(1580,770),(1690,770)],5,'Compositions',(1635,875)),
('detect','output',[(460,350),(460,210),(1785,210),(1785,605)],5,'YOLO boxes, agent class, and confidence',(1120,180))]
S=['<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 1920 1080" width="1920" height="1080">', '<rect width="1920" height="1080" fill="white"/>','<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto"><path d="M 0 1 L 9 5 L 0 9 Z" fill="#293442"/></marker></defs>']
def text(s,x,y,size=24,bold=False,anchor='middle',fill='#243345'):
 S.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{fill}">{html.escape(s)}</text>')
text('STAGE 5',50,73,22,True,'start','#9a4745');text('Learn compositions from video crops',50,125,44,True,'start')
S.append('<rect x="50" y="149" width="600" height="5" fill="#b85450"/>')
text('1  LOCALIZE',360,310,22,True);text('2  EXTRACT VISUAL EVIDENCE',900,310,22,True);text('3  SCORE AND COMPOSE',1470,310,22,True)
for i,(a,b,pts,phase,label,lp) in enumerate(ED):
 path='M '+' L '.join(f'{x},{y}' for x,y in pts)
 S.append(f'<path id="route{i}" d="{path}" fill="none" stroke="#293442" stroke-width="2.5" stroke-linejoin="round" marker-end="url(#arrow)"/>')
 # White backing prevents labels from obscuring paths.
 if label in ('Primitive scores','Action + location'):S.append(f'<rect x="{lp[0]-110}" y="{lp[1]-26}" width="220" height="36" fill="white"/>')
 text(label,*lp,22)
for k,(x,y,w,h,role,kind,lines) in N.items():
 fill,stroke=P[role];dash=' stroke-dasharray="9 7"' if kind=='frozen' else ''
 S.append(f'<rect id="node-{k}" x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="{4 if kind=="trained" else 2}"{dash}/>')
 if k=='input':
  text(lines[0],x+w/2,y+31,24,True);S.append(f'<image href="{imguri}" x="{x+14}" y="{y+44}" width="202" height="123"/>')
 elif k=='crop':
  text('Fixed-window crop',x+w/2,y+30,24,True)
  for j,u in enumerate(crops):S.append(f'<image href="{u}" x="{x+13+j*70}" y="{y+48}" width="65" height="56"/>')
  text('Across eight frames',x+w/2,y+127,21)
 elif k=='output':
  for j,l in enumerate(lines):text(l,x+w/2,y+36+29*j,25,True)
  for j,l in enumerate(['Agent','Action / location','Duplex / triplet']):text(l,x+w/2,y+114+31*j,21)
 else:
  for j,l in enumerate(lines):text(l,x+w/2,y+h/2-16.5*(len(lines)-1)+8+33*j,25,True)
text('Illustrative ROAD-Waymo box',165,685,19)
text('Retain detector boxes; apply detector confidence to emitted action, location, and composition scores.',960,913,25)
# Two-state border legend.
S.append('<rect x="540" y="958" width="64" height="30" rx="5" fill="#edf4fc" stroke="#6c8ebf" stroke-width="2" stroke-dasharray="9 7"/>')
text('Frozen during head training',620,981,22,False,'start')
S.append('<rect x="1100" y="958" width="64" height="30" rx="5" fill="#fcebea" stroke="#b85450" stroke-width="4"/>')
text('Trained scoring module',1180,981,22,False,'start')
static=''.join(S)+'</svg>';(A/'stage5-static.svg').write_text(static)
# Progressive travelling packets. Paths remain visible throughout.
for i,(_,_,pts,phase,_,_) in enumerate(ED):
 path='M '+' L '.join(f'{x},{y}' for x,y in pts)
 start=phase*1.3
 S.append(f'<circle r="7" fill="#b85450" stroke="white" stroke-width="2"><animateMotion dur="10s" repeatCount="indefinite" keyTimes="0;{start/10:.3f};{(start+1.2)/10:.3f};1" keyPoints="0;0;1;1" calcMode="linear" path="{path}"/><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;{start/10:.3f};{(start+.01)/10:.3f};{(start+1.15)/10:.3f};{(start+1.2)/10:.3f};1" dur="10s" repeatCount="indefinite"/></circle>')
(A/'stage5-animated.svg').write_text(''.join(S)+'</svg>')
(A/'stage5-animation.html').write_text('''<!doctype html><html><head><meta charset="utf-8"><title>Stage 5 animated architecture</title><style>body{margin:0;background:#eef1f5;font:16px Arial;color:#243345}main{max-width:1600px;margin:auto}svg{display:block;width:100%;height:auto;background:white}nav{display:flex;gap:16px;align-items:center;padding:14px 24px}button{padding:9px 18px;cursor:pointer;border:1px solid #b9c2cf;border-radius:6px;background:white}label{margin-left:auto}</style></head><body><main>'''+''.join(S)+'''</svg><nav><button id="pause">Pause</button><button id="restart">Restart</button><span>Follow the moving dots through Stage 5.</span><label>Speed <select id="speed"><option value="0.5">0.5×</option><option selected value="1">1×</option><option value="1.5">1.5×</option></select></label></nav></main><script>const svg=document.querySelector('svg');let paused=false,speed=1,t=0,last=performance.now();svg.pauseAnimations();function render(now){if(!paused)t+=(now-last)/1000*speed;svg.setCurrentTime(t%10);last=now;requestAnimationFrame(render)}requestAnimationFrame(render);document.querySelector('#pause').onclick=()=>{paused=!paused;document.querySelector('#pause').textContent=paused?'Play':'Pause'};document.querySelector('#restart').onclick=()=>{t=0;svg.setCurrentTime(0)};document.querySelector('#speed').onchange=e=>speed=+e.target.value;</script></body></html>''')
# Native editable source with matching positions and explicit waypoints.
f=E.Element('mxfile',host='app.diagrams.net');dg=E.SubElement(f,'diagram',name='Stage 5');g=E.SubElement(dg,'mxGraphModel',page='1',pageWidth=str(W),pageHeight=str(H),background='#ffffff');r=E.SubElement(g,'root');E.SubElement(r,'mxCell',id='0');E.SubElement(r,'mxCell',id='1',parent='0')
def vertex(id,label,x,y,w,h,style):
 c=E.SubElement(r,'mxCell',id=id,value=label,style=style,vertex='1',parent='1');E.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
for k,(x,y,w,h,role,kind,lines) in N.items():
 fill,stroke=P[role];label='<br>'.join(lines)
 if k=='input':label='Driving video<br><img src="'+imguri+'" width="200" height="120">'
 if k=='crop':label='Fixed-window crop<br>'+''.join('<img src="'+u+'" width="62" height="50">' for u in crops)+'<br>Across eight frames'
 if k=='output':label+=' <br><br>Agent<br>Action / location<br>Duplex / triplet'
 vertex(k,label,x,y,w,h,f'rounded=1;arcSize=6;whiteSpace=wrap;html=1;fontFamily=Helvetica;fontSize=24;fillColor={fill};strokeColor={stroke};strokeWidth={4 if kind=="trained" else 2};'+('dashed=1;dashPattern=9 7;' if kind=='frozen' else ''))
for i,(a,b,pts,phase,label,lp) in enumerate(ED):
 x,y,w,h,*_=N[a];ex=(pts[0][0]-x)/w;ey=(pts[0][1]-y)/h
 x,y,w,h,*_=N[b];enx=(pts[-1][0]-x)/w;eny=(pts[-1][1]-y)/h
 c=E.SubElement(r,'mxCell',id=f'e{i}',value=label if i in (6,7) else '',edge='1',parent='1',source=a,target=b,style=f'edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=20;html=1;strokeColor=#293442;strokeWidth=2;endArrow=block;flowAnimation=1;fontSize=22;labelBackgroundColor=#ffffff;exitX={ex};exitY={ey};entryX={enx};entryY={eny};')
 geom=E.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'})
 ar=E.SubElement(geom,'Array',attrib={'as':'points'})
 for xx,yy in pts[1:-1]:E.SubElement(ar,'mxPoint',x=str(xx),y=str(yy))
 lw=min(1250,max(44,len(label)*11));
 if i in (6,7):continue
 vertex(f'label{i}',label,lp[0]-lw/2,lp[1]-24,lw,30,'text;html=1;align=center;fillColor=none;strokeColor=none;fontSize=22;fontFamily=Helvetica;')
for id,l,x,y,w,h,size in [('title','STAGE 5: Learn compositions from video crops',50,50,1800,70,40),('caption','Retain detector boxes; apply detector confidence to emitted action, location, and composition scores.',50,885,1830,45,24),('example','Illustrative ROAD-Waymo box',50,660,230,30,19)]:vertex(id,l,x,y,w,h,f'text;html=1;align=center;strokeColor=none;fillColor=none;fontFamily=Helvetica;fontSize={size};')
vertex('legend-frozen','Frozen during head training',540,960,430,44,'rounded=1;html=1;fillColor=#edf4fc;strokeColor=#6c8ebf;dashed=1;dashPattern=9 7;fontSize=22;')
vertex('legend-trained','Trained scoring module',1100,960,380,44,'rounded=1;html=1;fillColor=#fcebea;strokeColor=#b85450;strokeWidth=4;fontSize=22;')
E.indent(f);E.ElementTree(f).write(A/'stage5.drawio',encoding='utf-8',xml_declaration=True)
(A/'animation-routes.json').write_text(json.dumps([{'points':p,'phase':ph} for _,_,p,ph,_,_ in ED]))
print(A)

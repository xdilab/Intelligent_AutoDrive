"""Reference-layout overview of the implemented BDD-X pilot and Stage 6 transfer."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from diagram_geometry import rounded_route
from PIL import Image,ImageDraw,ImageOps
import cv2,json,base64,io,html,xml.etree.ElementTree as E
from urllib.parse import quote
A=Path(__file__).resolve().parent;W,H=2400,1450
manifest=json.loads((A.parent/'bddx-contrastive-pilot/collected/manifest.json').read_text())
examples=[manifest['train'][i] for i in [287,1,4]]
def uri(im):
 b=io.BytesIO();im.save(b,format='PNG');return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
pretraining_images=[uri(Image.open(A/'assets'/name)) for name in ['pretraining-crossing.jpg','pretraining-road.jpg']]
images=[]
for i,row in enumerate(examples):
 cap=cv2.VideoCapture(str(A/'assets'/Path(row['video']).name));cap.set(cv2.CAP_PROP_POS_FRAMES,16);ok,frame=cap.read();cap.release();assert ok
 im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
 if i==0:im=im.crop((125,100,325,213)) # Display-only crop: actual pilot encoded whole segments.
 im.thumbnail((360,220));im.save(A/'assets'/f'pair-{i+1}.png');images.append(uri(im))
tail=json.loads((A/'tail-transfer-example.json').read_text());rr=tail['road']
road=Image.open(f"/data/datasets/road_waymo/rgb-images/{rr['video']}/{rr['frame']:05d}.jpg").convert('RGB');rw,rh=road.size
x1,y1,x2,y2=[v*d for v,d in zip(rr['box'],[rw,rh,rw,rh])];cx,cy=(x1+x2)/2,(y1+y2)/2;bw,bh=x2-x1,y2-y1
ImageDraw.Draw(road).rectangle((x1,y1,x2,y2),outline='#e7b638',width=10)
road=road.crop((max(0,int(cx-bw)),max(0,int(cy-bh)),min(rw,int(cx+bw)),min(rh,int(cy+bh))))
road=ImageOps.fit(road,(350,236),method=Image.Resampling.LANCZOS);road.save(A/'assets'/'road-tail-crop.png');roaduri=uri(road)

svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2400 1450" width="2400" height="1450">','<rect width="2400" height="1450" fill="white"/>','<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto"><path d="M0,1 L9,5 L0,9 Z" fill="#26323d"/></marker></defs>']
root=E.Element('mxfile',host='app.diagrams.net');diag=E.SubElement(root,'diagram',name='Pilot to Stage 6');gm=E.SubElement(diag,'mxGraphModel',page='1',pageWidth=str(W),pageHeight=str(H),background='#ffffff');xr=E.SubElement(gm,'root');E.SubElement(xr,'mxCell',id='0');E.SubElement(xr,'mxCell',id='1',parent='0')
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
  serial=parent in {'p1','p3'} and b!='out'
  outgoing=serial and a in ENCODERS
  routes.append({'points':points,'phase':phase+(0.7/1.5 if outgoing else 0),'duration':.6 if serial else 1.3,'panel':parent,'source':a,'target':b})
 ax,ay,aw,ah=positions[a];bx,by,bw,bh=positions[b];px,py=0,0
 ex,ey=(original[0][0]-ax)/aw,(original[0][1]-ay)/ah;ix,iy=(original[-1][0]-bx)/bw,(original[-1][1]-by)/bh
 st=f'edgeStyle=orthogonalEdgeStyle;rounded=1;arcSize=24;jettySize=20;html=1;endArrow=block;strokeColor=#26323d;strokeWidth=2;exitX={ex};exitY={ey};entryX={ix};entryY={iy};'+('dashed=1;' if dashed else '')
 c=E.SubElement(xr,'mxCell',id=f'e{counter}',edge='1',parent='1',source=a,target=b,style=st);geo=E.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'});ar=E.SubElement(geo,'Array',attrib={'as':'points'})
 for x,y in points[1:-1]:E.SubElement(ar,'mxPoint',x=str(x-px),y=str(y-py))
text('Driving-specific video–text adaptation for Stage 6',45,63,43,True)
text('Released initialization → implemented BDD-X pilot → downstream road-event detection',45,105,26)
for id,x,w,title in [('p1',40,720,'1. Pretrained initialization'),('p2',830,710,'2. Driving-specific adaptation'),('p3',1610,750,'3. Transfer to Stage 6')]:
 box(id,x,165,w,495,fill='#ffffff',stroke='#26323d',container=True)
 text(title,x+20,147,31,True)
# Panel 1: schematic released alignment, no invented training corpus.
text('InternVideo2-CLIP-S',400,211,28,True,'middle','p1')
box('prior-data',75,240,260,250,fill='#f8fafc',stroke='#a8b2bd',parent='p1',container=True)
text('Illustrative pairs',205,268,22,True,'middle','prior-data')
box('film',95,283,220,90,fill='#fff4cf',stroke='#c7ad4b',parent='prior-data',container=True)
pic('prior-photo-a',pretraining_images[0],100,288,102,80,'film')
pic('prior-photo-b',pretraining_images[1],208,288,102,80,'film')
box('prior-caption',95,398,220,72,['People at a crossing','A car on a road'],fill='#eaf2fc',stroke='#719ac6',parent='prior-data',size=21)
box('v0',425,280,110,75,['V₀'],fill='#fff0b3',stroke='#b29b41',kind='frozen',parent='p1',size=34)
box('l0',425,385,110,75,['L₀'],fill='#dceafd',stroke='#6c8ebf',kind='frozen',parent='p1',size=34)
edge('film','v0',[(315,318),(425,318)],0,'p1');edge('prior-caption','l0',[(315,422),(425,422)],0,'p1')
box('aligned',590,310,125,120,['Aligned','video–text','space'],fill='#eff5ed',stroke='#8cac76',parent='p1',size=23)
edge('v0','aligned',[(535,318),(560,318),(560,340),(590,340)],0,'p1');edge('l0','aligned',[(535,422),(560,422),(560,400),(590,400)],0,'p1')
text('Existing pretrained weights',400,535,27,True,'middle','p1')
text('Initialize both towers from the released pair.',400,578,24,False,'middle','p1')
text('Prior work; pretraining is not rerun in this pilot.',400,620,22,False,'middle','p1')
# Panel 2.
pic('bddx-top',images[0],860,238,230,135,'p2')
box('caption-top',860,404,230,86,['“…driving behind','a school bus…”'],fill='#f4f7fb',stroke='#a7b9ce',parent='p2',size=22)
text('BDD-X: language about a bus ahead',1185,209,26,True,'middle','p2')
box('vstar',1150,264,95,75,['V*'],fill='#d5e8d4',stroke='#6e9c59',kind='trained',parent='p2',size=34)
box('lstar',1150,409,95,75,['L*'],fill='#dceafd',stroke='#6c8ebf',kind='trained',parent='p2',size=34)
edge('bddx-top','vstar',[(1090,302),(1150,302)],2,'p2');edge('caption-top','lstar',[(1090,447),(1150,447)],2,'p2')
box('pilot-align',1320,311,180,128,['Contrastive','alignment'],fill='#fff5d9',stroke='#c4a756',parent='p2',size=26)
edge('vstar','pilot-align',[(1245,302),(1280,302),(1280,340),(1320,340)],3,'p2');edge('lstar','pilot-align',[(1245,447),(1280,447),(1280,405),(1320,405)],3,'p2')
text('Adapt last two blocks + projections',1185,550,25,True,'middle','p2')
text('1,024 train / 128 development videos',1185,590,24,False,'middle','p2')
text('Epoch 2 selected by BDD-X development loss',1185,630,23,False,'middle','p2')
# Top inter-panel transfer arrows.
edge('p1','p2',[(760,400),(830,400)],1);edge('p2','p3',[(1540,400),(1610,400)],7)
# Panel 3: abstract but correct feature distinction and both towers.
pic('road',roaduri,1640,220,175,118,'p3');text('GT training example',1740,370,22,False,'middle','p3');text('1,894 training boxes',1727,402,21,False,'middle','p3')
box('vfrozen',1860,243,95,75,['V*'],fill='#d5e8d4',stroke='#6e9c59',kind='frozen',parent='p3',size=34)
box('class-text',1640,450,175,75,['184 class','phrases'],fill='#f4f7fb',stroke='#a7b9ce',parent='p3',size=24)
box('lfrozen',1860,450,95,75,['L*'],fill='#dceafd',stroke='#6c8ebf',kind='frozen',parent='p3',size=34)
box('stage6',2060,292,255,143,['Stage 6 heads','Flat + phrase scores','Composition fusion'],fill='#fcebea',stroke='#b85450',kind='trained',parent='p3',size=25)
box('out',2060,535,255,66,['YOLO boxes','+ event scores'],fill='#f3f5f7',stroke='#919ba7',parent='p3',size=24)
edge('road','vfrozen',[(1815,280),(1860,280)],8,'p3');edge('class-text','lfrozen',[(1815,487),(1860,487)],8,'p3')
edge('vfrozen','stage6',[(1955,280),(2000,280),(2000,320),(2060,320)],8,'p3');text('1,024-D crop features',2150,258,22,False,'middle','p3')
edge('lfrozen','stage6',[(1955,487),(2000,487),(2000,402),(2060,402)],8,'p3');text('184 × 512 phrase matrix',1830,583,22,False,'middle','p3')
edge('stage6','out',[(2187,435),(2187,535)],9,'p3');text('Inference crops and output boxes come from YOLO',1980,640,22,False,'middle','p3')
# Detector bypass is named in output note, to avoid suggesting the encoder generates boxes.
text('ROAD-Waymo: Bus-MovAway-OutgoLane',1980,195,22,False,'middle','p3')
# Lower expanded panel and legend in left margin.
box('detail',300,765,2060,535,fill='#ffffff',stroke='#26323d',container=True)
text('Inside the implemented contrastive pilot',335,813,33,True,parent='detail')
# Expansion arrow ends on panel 2 lower border.
edge('detail','p2',[(1185,765),(1185,660)],6)
box('legend-v0',55,788,130,55,['V₀'],fill='#fff0b3',stroke='#b29b41',size=34);text('Released visual',45,874,22);text('encoder',45,904,22)
box('legend-vstar',55,923,130,55,['V*'],fill='#d5e8d4',stroke='#6e9c59',size=34);text('Adapted visual',45,1004,22);text('encoder',45,1034,22)
box('legend-l',45,1053,175,55,['L₀ / L*'],fill='#dceafd',stroke='#6c8ebf',size=32);text('Text encoders',45,1134,22)
badge('legend-ice','frozen',42,1181);text('Frozen',90,1210,22);badge('legend-fire','trained',42,1230);text('Trained',90,1259,22)
# Actual paired examples.
box('pairs',335,860,570,350,fill='#f7f9fc',stroke='#c3cbd5',parent='detail',container=True)
text('Pilot clip–caption pairs',360,896,25,True,parent='pairs')
short=[['The car is driving behind','a school bus in the middle','lane of a three lane highway'],['The car brakes slightly'],['The car decelerates','to a complete stop']]
for i in range(3):
 pic(f'pair{i}',images[i],355,918+94*i,140,79,'pairs')
 text(f'{i+1}',510,958+94*i,23,True,parent='pairs')
 for j,line in enumerate(short[i]):text(line,542,(935 if i==0 else 950)+94*i+24*j,21 if i==0 else 23,parent='pairs')
text('Eight sampled frames per clip; action text only',620,1250,23,False,'middle','detail')
# Towers are split visually into frozen body + trained final sections.
box('video-tower',965,873,320,134,fill='#f7fbf5',stroke='#8cac76',parent='detail',container=True)
text('VIDEO TOWER',1127,862,23,True,'middle','detail')
box('v-early',1000,900,240,38,['Earlier blocks'],fill='#edf4e9',stroke='#7d9a69',kind='frozen',parent='video-tower',size=22)
box('v-last',1000,951,240,38,['Last 2 + projections'],fill='#d5e8d4',stroke='#6e9c59',kind='trained',parent='video-tower',size=21)
box('text-tower',965,1090,320,134,fill='#f4f8fd',stroke='#8ba6c6',parent='detail',container=True)
text('TEXT TOWER',1127,1076,23,True,'middle','detail')
box('l-early',1000,1117,240,38,['Earlier blocks'],fill='#edf3fb',stroke='#7997bc',kind='frozen',parent='text-tower',size=22)
box('l-last',1000,1168,240,38,['Last 2 + projection'],fill='#dceafd',stroke='#6c8ebf',kind='trained',parent='text-tower',size=21)
edge('pairs','video-tower',[(905,940),(965,940)],2,'detail');edge('pairs','text-tower',[(905,1157),(965,1157)],2,'detail')
# Shared similarity matrix; graphic values encode positive membership, not measured scores.
box('matrix',1430,940,246,246,fill='#fff',stroke='#8794a2',parent='detail',container=True)
for i in range(3):
 for j in range(3):box(f'cell{i}{j}',1440+78*j,950+78*i,68,68,[str(i+1)+'↔'+str(j+1)] if i==j else [],fill='#d5e8d4' if i==j else '#edf0f4',stroke='#6e9c59' if i==j else '#d9dfe6',parent='matrix',size=20)
text('Clip–caption similarities',1553,862,25,True,'middle','detail');text('Normalized global embeddings',1553,899,22,False,'middle','detail')
edge('video-tower','matrix',[(1285,940),(1340,940),(1340,1000),(1430,1000)],3,'detail')
edge('text-tower','matrix',[(1285,1157),(1340,1157),(1340,1124),(1430,1124)],3,'detail')
text('512-D video',1360,932,21,False,'middle','detail');text('512-D text',1360,1220,21,False,'middle','detail')
text('Sᵢⱼ = cosine(vᵢ, tⱼ) / temperature',1553,1248,23,False,'middle','detail')
box('loss',1810,950,495,225,fill='#fff5d9',stroke='#c4a756',parent='detail',container=True)
text('Symmetric contrastive loss',2057,990,27,True,'middle','loss')
text('Video → text  +  text → video',2057,1034,25,False,'middle','loss')
text('Increase probability of matching pairs',2057,1082,23,False,'middle','loss')
text('Duplicate captions are also positives',2057,1124,23,False,'middle','loss')
edge('matrix','loss',[(1676,1063),(1810,1063)],4,'detail')
text('Batch 16 • 3 epochs • one adaptation seed',2057,1230,23,False,'middle','detail')
text('Temperature is learned along with selected layers.',2057,1266,21,False,'middle','detail')
# Clearly label illustration vs computed evidence.
text('Matrix is schematic (3 of 16 pairs); highlighted cells are positives, not measured similarity values.',320,1342,23)
text('Hypothesis: bus and lane language may help rare triplet recognition; a transfer gain is not yet demonstrated.',320,1385,23)
text('BDD-X names a bus ahead and its lane context; ROAD-Waymo supplies the bus action and location labels.',320,1423,23)
# Badges inherit the stage diagrams' exact portable vector artwork.
for node,kind in [('v0','frozen'),('l0','frozen'),('vstar','trained'),('lstar','trained'),('vfrozen','frozen'),('lfrozen','frozen'),('stage6','trained'),('v-early','frozen'),('l-early','frozen'),('v-last','trained'),('l-last','trained')]:
 x,y,w,h=positions[node];small=h<45;size=28 if small else 40
 badge(node+'-badge',kind,x+w-5 if small else x+w-22,y-4 if small else y-20,size)
static=''.join(svg)+'</svg>';(A/'stage6-static.svg').write_text(static)
# Glow only during the matching flow phase; legend and idle icons stay unlit.
windows={'v0-badge':(0,1.3),'l0-badge':(0,1.3),'vstar-badge':(3,5.8),'lstar-badge':(3,5.8),'vfrozen-badge':(12,13.3),'lfrozen-badge':(12,13.3),'stage6-badge':(13.5,14.8),'v-early-badge':(3,5.8),'l-early-badge':(3,5.8),'v-last-badge':(3,5.8),'l-last-badge':(3,5.8)}
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
for node,start,end in [('p1',0,1.3),('p2',3,10.3),('detail',3,10.3),('p3',12,14.8)]:
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
for i in range(3):svg.append(f'<rect x="{1440+78*i}" y="{950+78*i}" width="68" height="68" rx="10" fill="none" stroke="#4f8847" stroke-width="5"><animate attributeName="opacity" values="0;0;1;0;1;0;0" keyTimes="0;.25;.30;.35;.40;.45;1" dur="18s" repeatCount="indefinite"/></rect>')
animated=''.join(svg)+'</svg>';(A/'stage6-animated.svg').write_text(animated)
player='''<!doctype html><html><head><meta charset="utf-8"><title>BDD-X pilot to Stage 6</title><style>body{margin:0;background:#eef1f5;font:16px Arial;color:#26323d}main{max-width:1900px;margin:auto}svg{display:block;width:100%;height:calc(100vh -64px);background:white}nav{display:flex;gap:18px;align-items:center;padding:12px 24px}button,select{padding:8px 14px;border:1px solid #bbc4ce;border-radius:6px;background:white}label{margin-left:auto}</style></head><body><main>'''+animated+'''<nav><button id="pause">Pause</button><button id="restart">Restart</button><span>Pretrained pair → BDD-X contrastive adaptation → Stage 6</span><label>Speed <select id="speed"><option value="0.5">0.5×</option><option selected value="1">1×</option><option value="1.5">1.5×</option></select></label></nav></main><script>const svg=document.querySelector('svg');let paused=false,speed=1,t=0,last=performance.now();svg.pauseAnimations();function run(now){if(!paused)t+=(now-last)/1000*speed;svg.setCurrentTime(t%18);last=now;requestAnimationFrame(run)}requestAnimationFrame(run);document.querySelector('#pause').onclick=()=>{paused=!paused;document.querySelector('#pause').textContent=paused?'Play':'Pause'};document.querySelector('#restart').onclick=()=>{t=0;svg.setCurrentTime(0)};document.querySelector('#speed').onchange=e=>speed=+e.target.value;</script></body></html>'''
(A/'stage6-animation.html').write_text(player);(A/'animation-routes.json').write_text(json.dumps(routes));E.indent(root);E.ElementTree(root).write(A/'stage6.drawio',encoding='utf-8',xml_declaration=True)
(A/'examples.json').write_text(json.dumps(examples,indent=2)+'\n')
print(A)

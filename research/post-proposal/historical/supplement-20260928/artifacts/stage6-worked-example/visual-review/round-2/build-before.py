"""Source-derived Stage 6 worked example; native draw.io + sequential SVG player."""
from pathlib import Path
import hashlib
A=Path(__file__).resolve().parent
source=A.parent/'animated-stage5-composition/build.py'
helpers=source.read_text().split("text('Stage 5: crop features")[0]
helpers=helpers.replace('start,duration=schedule[(a,b)]','start,duration=phase').replace("if id in {'features','x','phrasebank','fixed-p'}:","if id in {'phrasebank','fixed-p'}:")
exec(compile(helpers,str(source),'exec'))
# Reuse the detailed diagram's shape vocabulary, palette, badges and route geometry.
example=json.loads((A.parent/'animated-stage6-pilot/tail-transfer-example.json').read_text())['road']
frames=Path('/data/datasets/ROAD_plusplus/rgb-images')/example['video']
fids=list(range(55,63));im=Image.open(frames/'00058.jpg');iw,ih=im.size
b=example['box'];cx=(b[0]+b[2])/2*iw;cy=(b[1]+b[3])/2*ih;bw=max((b[2]-b[0])*iw,8)*2;bh=max((b[3]-b[1])*ih,8)*2
crop=[max(cx-bw/2,0),max(cy-bh/2,0),min(cx+bw/2,iw),min(cy+bh/2,ih)]
provenance={'source_example':example,'frame_ids':fids,'frame_sha256':{},'crop_xyxy_px':crop,'helper_source':str(source),'helper_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'scope':'Completed Stage 6 forward path, frozen encoders. GT annotation stands in for candidate; no measured detector confidence or predictions. Frame previews are visual crops, not exact normalized RoIAlign tensors.'}
def crop_uri(fid):
 p=frames/f'{fid:05d}.jpg';provenance['frame_sha256'][str(fid)]=hashlib.sha256(p.read_bytes()).hexdigest();u=uri(Image.open(p));x1,y1,x2,y2=crop
 raw=f'<svg xmlns="http://www.w3.org/2000/svg" width="224" height="224" viewBox="{x1} {y1} {x2-x1} {y2-y1}" preserveAspectRatio="none"><image href="{u}" width="{iw}" height="{ih}"/></svg>'
 return 'data:image/svg+xml,'+quote(raw,safe='')
def vector(id,x,y,w=250,h=35,color='blue',label=''):
 cols=['#85b4dc','#d8e8f8','#5093c7'] if color=='blue' else ['#b09ccc','#e8dff1','#8064a2']
 box(id,x,y,w,h,fill='#ffffff',stroke='#4b88b5')
 for i in range(8):
  svg.append(f'<rect x="{x+i*w/8}" y="{y}" width="{w/8}" height="{h}" fill="{cols[i%3]}" stroke="white"/>')
  vert(id+f'-v{i}','',x+i*w/8,y,w/8,h,f'fillColor={cols[i%3]};strokeColor=#ffffff;')
 if label:text(label,x+w/2,y-17,21,True,'middle')
def line(a,b,pts,start,duration=1):edge(a,b,pts,phase=(start,duration))
text('Stage 6: Semantic Crop Classifier + Phrase Fusion + Comp MLP',45,58,37,True)
text('One rare bus example: candidate box → eight-frame crop → shared evidence → composition scores',45,101,25)
for id,x,w,title in [('p1',40,720,'1. Detection and crop creation'),('p2',830,710,'2. One contextual crop feature'),('p3',1610,750,'3. Assemble the event prediction')]:
 box(id,x,165,w,495,fill='#ffffff',stroke='#26323d',container=True);text(title,x+18,146,29,True)
# Panel 1: full frame, model, box, fixed crop sequence.
pic('frame',uri(im),90,210,270,180,'p1');box('yolo',465,225,235,80,['YOLOv8x','candidate detection'],fill='#dae8fc',stroke='#6c8ebf',kind='frozen',size=24);badge('ice-yolo','frozen',670,203)
box('candidate',445,350,275,90,['Candidate boxes','Confidence q','Agent class'],fill='#e5f0d8',stroke='#7e9f59',size=22)
# Overlay the verified annotation and actual padded window on the full frame.
for bb,col,dash in [(b,'#eaa400',''),([crop[0]/iw,crop[1]/ih,crop[2]/iw,crop[3]/ih],'#0e8a7d','dashed=1;')]:
 x=90+bb[0]*270;y=210+bb[1]*180;ww=(bb[2]-bb[0])*270;hh=(bb[3]-bb[1])*180
 svg.append(f'<rect x="{x}" y="{y}" width="{ww}" height="{hh}" fill="none" stroke="{col}" stroke-width="3"'+(' stroke-dasharray="6 4"' if dash else '')+'/>');vert('bbox'+str(len(positions)),'',x,y,ww,hh,f'fillColor=none;strokeColor={col};strokeWidth=3;{dash}')
text('Gold: object box • teal: 2× padded window',65,417,20)
box('sequence',65,491,665,105,fill='#ffffff',stroke='#82b366')
for i,fid in enumerate(fids):
 x=72+i*82;pic('crop'+str(fid),crop_uri(fid),x,498,75,75,'1');text(('t' if fid==58 else f't{fid-58:+d}'),x+37,592,18,fid==58,'middle')
text('Same window across frames 55–62; keyframe t = 58',70,628,21)
line('frame','yolo',[(360,275),(465,275)],.3,.8);line('yolo','candidate',[(582,305),(582,350)],1.3,.7);line('candidate','sequence',[(582,440),(582,491)],2.2,.8)
# Panel 2.
box('clip-input',865,215,230,74,['8 × 3 × 224 × 224','RGB crop sequence'],fill='#d5e8d4',stroke='#82b366',size=23)
box('video-tower',1155,211,325,125,['InternVideo2-CLIP-S','video encoder'],kind='frozen',size=25);badge('ice-video','frozen',1435,189)
vector('features',995,420,350,45,label='Crop features [1024]')
text('One 1-D vector per sequence',990,500,24)
text('Context from eight frames; keyframe spatial-token mean',860,548,23)
text('Not eight separate classification vectors',860,588,22,color='#65717b')
text('Continue below: two evidence branches',940,630,23,True)
line('p1','p2',[(760,400),(830,400)],3.3,1.4);line('clip-input','video-tower',[(1095,251),(1155,251)],5.0,.8);line('video-tower','features',[(1305,336),(1390,336),(1390,442),(1345,442)],6.1,1.0)
# Lower detail.
box('detail',40,750,2320,600,fill='#ffffff',stroke='#26323d',container=True)
text('Inside Stage 6: primitive scores, phrase evidence, and crop features',65,730,29,True)
vector('x',85,820,230,38,label='Crop features [1024]')
box('flat',390,788,300,94,['Linear classifier','a = Wf x + bf'],fill='#f8cecc',stroke='#b85450',kind='trained',size=24);badge('fire-flat','trained',655,765)
box('primitive',780,790,340,105,['Primitive scores [49]','u = sigmoid(a[0:49])','1 + 10 + 22 + 16'],fill='#e3f3f5',stroke='#55969e',size=22)
box('linear-scores',390,923,300,90,['Linear scores','sigmoid(a[11:49])','22 action + 16 location'],fill='#e3f3f5',stroke='#55969e',size=21)
text('Agentness • agent • action • location',780,925,19)
# Text branch: actual label text, not a claim of exact original template wording.
box('phrases',80,1020,310,150,['Phrase bank (184 entries)','“bus moving away”','“bus moving away in','the outgoing lane”'],fill='#f1edf7',stroke='#8064a2',size=21)
box('text-tower',460,1045,220,100,['Text encoder','computed offline'],kind='frozen',size=22);badge('ice-text','frozen',645,1020)
box('fixed-p',750,1030,180,95,size=20)
text('Unit-norm rows',750,1190,20)
box('projection',390,1220,370,75,['z = normalize(Wp x + ap)','Learned 1024 → 512 projection'],fill='#f8cecc',stroke='#b85450',kind='trained',size=22);badge('fire-proj','trained',725,1195)
box('cosine',1020,1020,370,130,['c = Pz  (cosine similarities)','ℓp = exp(η)c + bp','v = sigmoid(ℓp[49:184])','135 phrase-composition scores'],fill='#f8cecc',stroke='#b85450',kind='trained',size=22);badge('fire-cosine','trained',1355,997)
# MLP at right, separate from score assembly.
box('concat',1485,875,300,150,['Concatenate [u; v; x]','49 + 135 + 1024','= 1208 values'],fill='#fff2cc',stroke='#d6b656',size=25)
box('mlp',1875,865,420,170,['Composition MLP','1208 → 512 → ReLU → 135','m = W₂ ReLU(W₁[u;v;x] + b₁) + b₂','49 duplex + 86 triplet logits'],fill='#f8cecc',stroke='#b85450',kind='trained',size=22);badge('fire-mlp','trained',2260,842)
box('composition-scores',1875,1110,420,100,['Composition scores','sigmoid(m)','49 duplex + 86 triplet'],fill='#eee8f6',stroke='#9478b5',size=22)
line('mlp','composition-scores',[(2085,1035),(2085,1110)],19.9,.5)
text('Example: Bus–MovAway–OutgoLane',1490,1260,25,True)
text('Bus • moving away • outgoing lane',1490,1300,22)

text('Phrase examples shown in plain language; full matrix retains all 184 entries.',80,1324,20,color='#65717b')
# Explicit routing: top → detail → top-right, with no simultaneous top/detail phase.
line('p2','detail',[(1185,660),(1185,750)],7.4,1.4)
line('x','flat',[(315,839),(390,839)],9.0,.7)
line('flat','primitive',[(690,835),(780,835)],10.0,.7)
line('x','projection',[(200,858),(200,973),(340,973),(340,1257),(390,1257)],10.9,1.1)
line('phrases','text-tower',[(390,1090),(460,1090)],12.2,.7)
line('text-tower','fixed-p',[(680,1090),(750,1090)],13.1,.7)
line('fixed-p','cosine',[(930,1080),(1020,1080)],14.0,.7)
line('projection','cosine',[(760,1257),(1205,1257),(1205,1150)],14.0,1.0)
line('primitive','concat',[(1120,835),(1415,835),(1415,913),(1485,913)],15.3,1.0)
line('flat','linear-scores',[(540,882),(540,923)],10.0,.7)
line('cosine','concat',[(1390,1085),(1435,1085),(1435,975),(1485,975)],16.5,.8)
line('x','concat',[(200,820),(200,767),(1600,767),(1600,875)],17.5,1.2)
line('concat','mlp',[(1785,950),(1875,950)],19.0,.8)
# Final panel: the same names, colors and layout as the detailed Stage5 reference.
box('detector',1640,210,275,85,['Candidate boxes','Confidence q','Agent class'],fill='#e5f0d8',stroke='#7e9f59',size=22)
box('primitive-final',1640,330,275,75,['Linear scores','22 action + 16 location'],fill='#e3f3f5',stroke='#55969e',size=22)
box('comp-final',1640,445,275,85,['Composition scores','49 duplex scores','86 triplet scores'],fill='#eee8f6',stroke='#9478b5',size=22)
box('assembly',2030,345,280,155,['Confidence weighting','q × linear scores','q × composition scores'],fill='#fff5d9',stroke='#c4a756',size=22)
box('out',2000,550,325,72,['Candidate boxes','+ 184 scores (11 + 38 + 135)'],fill='#e5f0d8',stroke='#7e9f59',size=21)
text('Bus–MovAway–OutgoLane: q × sigmoid(m[triplet])',1640,642,21)
line('detail','p3',[(2250,750),(2250,660)],20.6,1.0)
line('detector','assembly',[(1915,252),(2170,252),(2170,345)],21.8,.6)
line('primitive-final','assembly',[(1915,367),(1970,367),(1970,402),(2030,402)],22.6,.6)
line('comp-final','assembly',[(1915,487),(1970,487),(1970,455),(2030,455)],22.6,.6)
line('assembly','out',[(2170,500),(2170,550)],23.5,.7)
text('Verified annotation example: train_00552, t = 58 • 1,894 training boxes • triplet frequency z = −0.732',45,1398,22)
text('Illustrative candidate and symbolic scores; not measured YOLO output. Frozen encoders shown. Fire = trained head; ice = frozen.',45,1432,21,color='#65717b')
provenance['visual_review']={'reference':'artifacts/animated-stage5-composition/stage5-static.png','cross_panel_routes':[['p1','p2'],['p2','detail'],['detail','p3']],'data_colors':{'Candidate boxes':'#e5f0d8','Crop features':'#85b4dc','Linear scores':'#e3f3f5','Composition scores':'#eee8f6'},'repeated_labels':'Crop features [1024]; Candidate boxes; Linear scores; Composition scores'}
# Static artifact and editable draw.io are kept independent of animated overlays.
static=''.join(svg)+'</svg>';(A/'stage6-static.svg').write_text(static)
# Native groups reflect visible containment; badges intentionally overlap module corners.
for c in list(xr):
 g=c.find('mxGeometry')
 if c.get('parent')!='1' or c.get('vertex')!='1' or g is None or c.get('id') in ['p1','p2','p3','detail']:continue
 x,y,w,h=[float(g.get(k,0)) for k in ['x','y','width','height']]
 for parent in ['p1','p2','p3','detail']:
  px,py,pw,ph=positions[parent]
  if x>=px and y>=py and x+w<=px+pw and y+h<=py+ph:
   c.set('parent',parent);g.set('x',str(x-px));g.set('y',str(y-py));break
E.indent(root);E.ElementTree(root).write(A/'stage6-worked-example.drawio',encoding='utf-8',xml_declaration=True)
# Active-only gold packets, orange arrows, module/badge and outer-panel glow.
svg.append('<defs><filter id="glow" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>')
D=26
def pulse(start,end):
 return f'<animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;{start/D};{(start+.04)/D};{(end-.04)/D};{end/D};1" dur="{D}s" repeatCount="indefinite"/>'
for r in routes:
 start=r['phase']*1.5;end=start+r['duration'];path=rounded_route(r['points'])[0]
 svg.append(f'<path d="{path}" fill="none" stroke="#ff981e" stroke-width="4" opacity="0" filter="url(#glow)">{pulse(start,end)}</path>')
 svg.append(f'<circle r="8" fill="#FDB927" stroke="#fff2be" stroke-width="2" opacity="0" filter="url(#glow)"><animateMotion dur="{D}s" repeatCount="indefinite" keyTimes="0;{start/D};{end/D};1" keyPoints="0;0;1;1" calcMode="linear" path="{path}"/>{pulse(start,end)}</circle>')
windows={'p1':(.2,4.7),'p2':(5,8.8),'detail':(9,20.4),'p3':(21.8,24.3),'flat':(9.7,10.8),'projection':(11,12),'cosine':(14,15.2),'mlp':(19,20.1),'video-tower':(5.8,7.1),'text-tower':(12.8,13.8)}
for node,(start,end) in windows.items():
 x,y,w,h=positions[node]
 if node in ENCODERS:
  pad=min(25,w*.18);svg.append(f'<polygon points="{x+pad},{y} {x+w},{y} {x+w-pad},{y+h} {x},{y+h}" fill="none" stroke="#e9ad35" stroke-width="4" opacity="0" filter="url(#glow)">{pulse(start,end)}</polygon>')
 else:svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="15" fill="none" stroke="#e9ad35" stroke-width="4" opacity="0" filter="url(#glow)">{pulse(start,end)}</rect>')
for b in badges:
 window={'ice-yolo':(.3,2),'ice-video':(5.8,7.1),'ice-text':(12.8,13.8),'fire-flat':(9.7,10.8),'fire-proj':(11,12),'fire-cosine':(14,15.2),'fire-mlp':(19,20.1)}[b['id']]
 raw=(A/'assets'/('ice.svg' if b['kind']=='frozen' else 'fire.svg')).read_text();u='data:image/svg+xml,'+quote(raw,safe='');svg.append(f'<image href="{html.escape(u)}" x="{b["x"]}" y="{b["y"]}" width="{b["size"]}" height="{b["size"]}" opacity="0" filter="url(#glow)">{pulse(*window)}</image>')
animated=''.join(svg)+'</svg>';(A/'stage6-animated.svg').write_text(animated)
player='''<!doctype html><html><head><meta charset="utf-8"><title>Stage 6: one rare bus, end to end</title><style>body{margin:0;background:#eef1f5;font:16px Arial;color:#26323d}main{max-width:2400px;margin:auto}svg{display:block;width:100%;height:calc(100vh - 68px);background:white}nav{display:flex;gap:16px;align-items:center;padding:12px 22px}button,select{padding:8px 14px;border:1px solid #bbc4ce;border-radius:6px;background:white}label{margin-left:auto}</style></head><body><main>'''+animated+'''<nav><button id="pause">Pause</button><button id="restart">Restart</button><span id="phase">Detection → crop sequence → feature → scoring → assembly</span><label>Speed <select id="speed"><option value="0.5">0.5×</option><option selected value="1">1×</option><option value="1.5">1.5×</option></select></label></nav></main><script>const svg=document.querySelector('svg');let paused=false,speed=1,t=0,last=performance.now();svg.pauseAnimations();function run(now){if(!paused)t+=(now-last)/1000*speed;svg.setCurrentTime(t%26);last=now;requestAnimationFrame(run)}requestAnimationFrame(run);document.querySelector('#pause').onclick=()=>{paused=!paused;document.querySelector('#pause').textContent=paused?'Play':'Pause'};document.querySelector('#restart').onclick=()=>{t=0;svg.setCurrentTime(0)};document.querySelector('#speed').onchange=e=>speed=+e.target.value;</script></body></html>'''
(A/'stage6-animation.html').write_text(player);(A/'provenance.json').write_text(json.dumps(provenance,indent=2));(A/'animation-routes.json').write_text(json.dumps(routes,indent=2));print('Built example:',fids,'crop',crop)

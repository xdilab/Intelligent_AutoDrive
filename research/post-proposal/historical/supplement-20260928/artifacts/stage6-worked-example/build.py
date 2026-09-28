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
def vector(id,x,y,w=240,h=40,color='blue',label=''):
 # Exact one-row/six-cell tensor construction from the Stage4/5 helper.
 cols=['#85b4dc','#d8e8f8','#5093c7']
 vert(id,'',x,y,w,h,'container=1;pointerEvents=0;fillColor=none;strokeColor=none;')
 for col in range(6):
  xx=x+col*w/6;fill=cols[col%3]
  svg.append(f'<rect x="{xx}" y="{y}" width="{w/6}" height="{h}" fill="{fill}" stroke="#4b88b5" stroke-width="1.5"/>')
  vert(f'{id}-tile-0-{col}','',xx,y,w/6-.01,h-.01,f'rounded=0;fillColor={fill};strokeColor=#4b88b5;strokeWidth=1;',id)
 if label:text(label,x+w/2,y-15,23,True,'middle')
# Presentation timeline: inputs arrive, computation pulses, then outputs leave.
TIMELINE={
 ('frame','yolo'):(.3,.8), ('yolo','candidate'):(1.5,.7), ('candidate','sequence'):(2.5,.8),
 ('p1','p2'):(3.6,1), ('clip-input','video-tower'):(4.9,.8), ('video-tower','features'):(6.5,.8), ('p2','detail'):(7.7,1),
 ('x','flat'):(9,.7), ('x','projection'):(9,2),
 ('flat','primitive'):(11.5,.8), ('flat','linear-scores'):(11.5,2.4),
 ('phrases','text-tower'):(14.3,.7), ('text-tower','fixed-p'):(15.6,.7),
 ('fixed-p','cosine'):(17,.8), ('projection','cosine'):(17,1.1),
 ('primitive','concat'):(19.3,1.8), ('x','concat'):(19.3,3.5), ('cosine','concat'):(23.1,.7),
 ('concat','mlp'):(24.5,.7), ('mlp','composition-scores'):(26,.7), ('detail','p3'):(27,1),
 ('detector','assembly'):(28.3,.7), ('primitive-final','assembly'):(29.3,.7), ('comp-final','assembly'):(29.3,.7), ('assembly','out'):(30.7,.7),
}
def line(a,b,pts,start,duration=1):edge(a,b,pts,phase=TIMELINE[(a,b)])
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
vector('features',1140,420,350,45)
text('Crop features [1024]',1315,494,21,True,'middle')
text('One 1-D vector per sequence',1315,530,24,False,'middle')
text('Context from eight frames; keyframe spatial-token mean',860,570,23)
text('Not eight separate classification vectors',860,603,22,color='#65717b')
text('Continue below: two evidence branches',940,630,23,True)
line('p1','p2',[(760,400),(830,400)],3.3,1.4);line('clip-input','video-tower',[(1095,251),(1155,251)],5.0,.8);line('video-tower','features',[(1315,336),(1315,420)],6.1,1.0)
# Lower detail: one shared crop vector, with three explicitly routed consumers.
box('detail',40,750,2320,600,fill='#ffffff',stroke='#26323d',container=True)
text('Inside Stage 6: primitive scores, phrase evidence, and crop features',65,730,29,True)
vector('x',130,820,230,40,label='Crop features [1024]')
box('flat',530,790,290,90,['Linear classifier','a = Wf x + bf'],fill='#f8cecc',stroke='#b85450',kind='trained',size=24);badge('fire-flat','trained',785,790)
box('primitive',970,790,280,100,['Primitive scores [49]','u = sigmoid(a[0:49])','1 + 10 + 22 + 16'],fill='#e3f3f5',stroke='#55969e',size=22)
box('linear-scores',1430,790,350,80,['Linear scores','sigmoid(a[11:49])','22 action + 16 location'],fill='#e3f3f5',stroke='#55969e',size=21)
box('projection',835,950,330,75,['z = normalize(Wp x + ap)','Learned 1024 → 512 projection'],fill='#f8cecc',stroke='#b85450',kind='trained',size=21);badge('fire-proj','trained',1130,950)
box('phrases',80,1090,310,150,['Phrase bank (184 entries)','“bus moving away”','“bus moving away in','the outgoing lane”'],fill='#f1edf7',stroke='#8064a2',size=21)
box('text-tower',460,1115,220,100,['L₀'],kind='frozen',size=34);badge('ice-text','frozen',645,1090)
text('Text encoder',570,1250,23,False,'middle')
text('Computed offline',570,1282,21,False,'middle')
box('fixed-p',770,1120,180,95,size=20)
text('Unit-norm rows',860,1280,20,False,'middle')
box('cosine',1120,1100,380,130,['c = Pz  (cosine similarities)','ℓp = exp(η)c + bp','v = sigmoid(ℓp[49:184])','135 phrase-composition scores'],fill='#f8cecc',stroke='#b85450',kind='trained',size=22);badge('fire-cosine','trained',1465,1077)
box('concat',1600,1060,280,170,['Concatenate [u; v; x]','49 + 135 + 1024','= 1208 values'],fill='#fff2cc',stroke='#d6b656',size=24)
box('mlp',1970,1030,350,170,['Composition MLP','1208 → 512 → ReLU → 135','m = W₂ ReLU(W₁[u;v;x] + b₁) + b₂','49 duplex + 86 triplet logits'],fill='#f8cecc',stroke='#b85450',kind='trained',size=19);badge('fire-mlp','trained',2285,1007)
box('composition-scores',1970,1250,350,75,['Composition scores','sigmoid(m): 49 duplex + 86 triplet'],fill='#eee8f6',stroke='#9478b5',size=20)
text('Bus–MovAway–OutgoLane',1900,830,23,True)
text('Bus • moving away',1900,875,23)
text('• outgoing lane',1900,915,23)
text('One shared crop feature x feeds the linear classifier, learned projection, and composition input.',80,1324,20,color='#65717b')
line('p2','detail',[(1185,660),(1185,750)],7.4,1.4)
line('x','flat',[(360,840),(530,840)],9.0,.7)
line('flat','primitive',[(820,835),(970,835)],10.0,.7)
line('flat','linear-scores',[(675,790),(675,765),(1605,765),(1605,790)],10.0,.7)
line('x','projection',[(245,860),(245,987),(835,987)],10.9,1.1)
line('phrases','text-tower',[(390,1165),(460,1165)],12.2,.7)
line('text-tower','fixed-p',[(680,1165),(770,1165)],13.1,.7)
line('fixed-p','cosine',[(950,1165),(1120,1165)],14.0,.7)
line('projection','cosine',[(1000,1025),(1000,1055),(1310,1055),(1310,1100)],14.0,1.0)
line('primitive','concat',[(1110,890),(1110,905),(1740,905),(1740,1060)],15.3,1.0)
line('cosine','concat',[(1500,1165),(1600,1165)],16.5,.8)
line('x','concat',[(360,840),(450,840),(450,925),(1560,925),(1560,1095),(1600,1095)],17.5,1.2)
line('concat','mlp',[(1880,1145),(1970,1145)],19.0,.8)
line('mlp','composition-scores',[(2145,1200),(2145,1250)],19.9,.5)
# Final panel: the same names, colors and layout as the detailed Stage5 reference.
box('detector',1640,210,275,85,['Candidate boxes','Confidence q','Agent class'],fill='#e5f0d8',stroke='#7e9f59',size=22)
box('primitive-final',1640,330,275,75,['Linear scores','22 action + 16 location'],fill='#e3f3f5',stroke='#55969e',size=22)
box('comp-final',1640,445,275,85,['Composition scores','49 duplex scores','86 triplet scores'],fill='#eee8f6',stroke='#9478b5',size=22)
box('assembly',2030,210,280,320,['Confidence weighting','q × linear scores','q × composition scores'],fill='#fff5d9',stroke='#c4a756',size=22)
box('out',2000,580,325,62,['Candidate boxes','+ 184 scores (11 + 38 + 135)'],fill='#e5f0d8',stroke='#7e9f59',size=21)
text('Triplet score: q × sigmoid(m[triplet])',1640,562,20)
line('detail','p3',[(2250,750),(2250,660)],20.6,1.0)
line('detector','assembly',[(1915,252),(2030,252)],21.8,.6)
line('primitive-final','assembly',[(1915,367),(2030,367)],22.6,.6)
line('comp-final','assembly',[(1915,487),(2030,487)],22.6,.6)
line('assembly','out',[(2170,530),(2170,580)],23.5,.7)
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
# Animation implementation reused from the detailed Stage5 reference.
D=34
windows={'ice-yolo':(1.1,1.5),'ice-video':(5.7,6.5),'ice-text':(15,15.6),'fire-flat':(9.7,11.5),'fire-proj':(11,12),'fire-cosine':(18.1,19.1),'fire-mlp':(25.2,26)}
for b in badges:
 if b['id'] not in windows:continue
 start,end=windows[b['id']];b['window']=[start,end]
 i=b['index'];x=b['x']+b['size']/2;y=b['y']+b['size']/2
 color='#39a8ed' if b['kind']=='frozen' else '#ff981e'
 times=[0]+([start/D] if start else [])+[(start+min(.2,(end-start)/3))/D,(end-min(.2,(end-start)/3))/D,end/D,1]
 values=[0]+([0] if start else [])+[.85,.85,0,0]
 keytimes=';'.join(str(t) for t in times);vals=';'.join(str(v) for v in values)
 svg[i]=f'<g class="status-glow" data-badge="{b["id"]}"><circle cx="{x}" cy="{y}" r="{b["size"]*.52}" fill="{color}" filter="url(#badge-glow)" opacity="0"><animate attributeName="opacity" values="{vals}" keyTimes="{keytimes}" dur="34s" repeatCount="indefinite"/></circle><circle class="pulse-ring" cx="{x}" cy="{y}" r="{b["size"]*.5}" fill="none" stroke="{color}" stroke-width="2.5" opacity="0"><animate attributeName="opacity" values="{vals}" keyTimes="{keytimes}" dur="34s" repeatCount="indefinite"/><animate attributeName="r" values="{b["size"]*.48};{b["size"]*.75};{b["size"]*.48}" dur="1s" repeatCount="indefinite"/></circle>'+svg[i]+'</g>'

svg.append('<defs><filter id="badge-glow" x="-80%" y="-80%" width="260%" height="260%"><feGaussianBlur stdDeviation="4"/></filter></defs>')
(A/'animation-badges.json').write_text(json.dumps(badges,indent=2)+'\n')
# Only trainable model boundaries pulse; frozen layers retain their dashed borders.
svg.append('<defs><filter id="border-glow" x="-30%" y="-70%" width="160%" height="240%"><feGaussianBlur stdDeviation="4" result="halo"/><feMerge><feMergeNode in="halo"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>')
for b in trained:
 start,end=windows[{'flat':'fire-flat','projection':'fire-proj','cosine':'fire-cosine','mlp':'fire-mlp'}[b['id']]];b['window']=[start,end]
 x,y,w,h=b['x'],b['y'],b['w'],b['h'];color={'#0e8a7d':'#14b8a6','#8064a2':'#a879e0','#b85450':'#e66a61'}[b['stroke']];b['glow_color']=color
 times=[0,start/D]+[(start+(end-start)*f)/D for f in [.25,.5,.75,1]]+[1]
 animation='<animate attributeName="opacity" values="0;0;1;0.25;1;0;0" keyTimes="'+';'.join(map(str,times))+'" dur="34s" repeatCount="indefinite"/>'
 if b['polygon']:
  pad=min(25,w*.18);shape=f'<polygon points="{x+pad},{y} {x+w},{y} {x+w-pad},{y+h} {x},{y+h}"'
 else:shape=f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="15"'
 svg.append(f'<g class="trainable-border" data-node="{b["id"]}" opacity="0" fill="none" stroke="{color}" stroke-width="5" filter="url(#border-glow)">{animation}'+shape+'/></g>')
(A/'animation-borders.json').write_text(json.dumps(trained,indent=2)+'\n')
# Section emphasis is separate from the frozen/trainable model grammar.
panels=[]
for node,start,end in [('p1',.2,3.4),('p2',4.6,7.4),('detail',8.7,26.8),('p3',28,32)]:
 x,y,w,h=positions[node]
 times=[0]+([start/D] if start else [])+[(start+min(.2,(end-start)/3))/D,(end-min(.2,(end-start)/3))/D,end/D,1]
 values=[0]+([0] if start else [])+[.8,.8,0,0]
 anim='<animate attributeName="opacity" values="'+';'.join(map(str,values))+'" keyTimes="'+';'.join(map(str,times))+'" dur="34s" repeatCount="indefinite"/>'
 svg.append(f'<rect class="section-glow" data-node="{node}" x="{x}" y="{y}" width="{w}" height="{h}" rx="15" fill="none" stroke="#e9ad35" stroke-width="3" opacity="0" filter="url(#border-glow)">{anim}</rect>')
 panels.append({'id':node,'x':x,'y':y,'w':w,'h':h,'window':[start,end]})
(A/'animation-panels.json').write_text(json.dumps(panels,indent=2)+'\n')


for node,start,end in [('concat',23.8,24.5),('assembly',30,30.7),('out',31.4,33.4)]:
 x,y,w,h=positions[node]
 svg.append(f'<rect class="operation-glow" data-node="{node}" x="{x}" y="{y}" width="{w}" height="{h}" rx="15" fill="none" stroke="#e9ad35" stroke-width="3" opacity="0" filter="url(#border-glow)"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;{start/D};{(start+.1)/D};{(end-.1)/D};{end/D};1" dur="34s" repeatCount="indefinite"/></rect>')

# Animated phase routes + positive-cell pulses; timeline is explanatory, not runtime.
svg.append('<defs><filter id="packet-glow" x="-200%" y="-200%" width="500%" height="500%"><feGaussianBlur stdDeviation="5" result="halo"/><feMerge><feMergeNode in="halo"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>')
# Active forward-flow edges heat up only while their packet is moving.
svg.append('<defs><marker id="hot-arrow" markerWidth="25" markerHeight="25" viewBox="0 0 10 10" refX="9" refY="5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,1 L9,5 L0,9 Z" fill="#ffab24"/></marker><filter id="arrow-heat" x="-30" y="-30" width="2460" height="1510" filterUnits="userSpaceOnUse"><feGaussianBlur stdDeviation="3" result="heat"/><feMerge><feMergeNode in="heat"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>')
for n,r in enumerate(routes):
 start=r['phase']*1.5;end=start+r['duration'];path=rounded_route(r['points'])[0]
 times=[0]+([start/D] if start else [])+[(start+.06)/D,(end-.06)/D,end/D,1]
 values=[0]+([0] if start else [])+[.95,.95,0,0]
 anim='<animate attributeName="opacity" values="'+';'.join(map(str,values))+'" keyTimes="'+';'.join(map(str,times))+'" dur="34s" repeatCount="indefinite"/>'
 svg.append(f'<path class="hot-flow-arrow" data-route="{n}" d="{path}" fill="none" stroke="#ff981e" stroke-width="4" stroke-linejoin="round" marker-end="url(#hot-arrow)" filter="url(#arrow-heat)" opacity="0">{anim}</path>')
for r in routes:
 start=r['phase']*1.5;duration=r['duration'];path=rounded_route(r['points'])[0]
 svg.append(f'<circle r="8" fill="#FDB927" stroke="#fff2be" stroke-width="2" filter="url(#packet-glow)"><animateMotion dur="34s" repeatCount="indefinite" keyTimes="0;{start/D:.5f};{(start+duration)/D:.5f};1" keyPoints="0;0;1;1" calcMode="linear" path="{path}"/><animate attributeName="opacity" dur="34s" repeatCount="indefinite" values="0;0;1;1;0;0" keyTimes="0;{start/D:.5f};{(start+.01)/D:.5f};{(start+duration-.05)/D:.5f};{(start+duration)/D:.5f};1"/></circle>')
animated=''.join(svg)+'</svg>';(A/'stage6-animated.svg').write_text(animated)
player='''<!doctype html><html><head><meta charset="utf-8"><title>Stage 6: one rare bus, end to end</title><style>body{margin:0;background:#eef1f5;font:16px Arial;color:#26323d}main{max-width:2400px;margin:auto}svg{display:block;width:100%;height:calc(100vh - 68px);background:white}nav{display:flex;gap:16px;align-items:center;padding:12px 22px}button,select{padding:8px 14px;border:1px solid #bbc4ce;border-radius:6px;background:white}label{margin-left:auto}</style></head><body><main>'''+animated+'''<nav><button id="pause">Pause</button><button id="restart">Restart</button><span id="phase">Detection → crop sequence → feature → scoring → assembly</span><label>Speed <select id="speed"><option value="0.5">0.5×</option><option selected value="1">1×</option><option value="1.5">1.5×</option></select></label></nav></main><script>const svg=document.querySelector('svg');let paused=false,speed=1,t=0,last=performance.now();const steps=[[0,'1 / 8 · Detect the candidate and build its crop sequence'],[4.6,'2 / 8 · Encode eight frames into one crop feature'],[8.7,'3 / 8 · Shared feature → classifier and projection'],[14.1,'4 / 8 · Read the offline text-encoder / phrase-matrix path'],[17,'5 / 8 · Combine projected crop and phrase matrix'],[19.3,'6 / 8 · Gather primitive scores, crop features, and phrase scores'],[24.5,'7 / 8 · Composition MLP → composition scores'],[28,'8 / 8 · Weight the scores and assemble the event'],[31.4,'Complete · Candidate boxes + 184 scores']];svg.pauseAnimations();function run(now){if(!paused)t+=(now-last)/1000*speed;svg.setCurrentTime(t%34);const q=t%34;document.querySelector('#phase').textContent=steps.slice().reverse().find(s=>q>=s[0])[1];last=now;requestAnimationFrame(run)}requestAnimationFrame(run);document.querySelector('#pause').onclick=()=>{paused=!paused;document.querySelector('#pause').textContent=paused?'Play':'Pause'};document.querySelector('#restart').onclick=()=>{t=0;svg.setCurrentTime(0)};document.querySelector('#speed').onchange=e=>speed=+e.target.value;</script></body></html>'''
(A/'stage6-animation.html').write_text(player);(A/'provenance.json').write_text(json.dumps(provenance,indent=2));(A/'animation-timing.json').write_text(json.dumps({'duration':34,'semantics':'Explanatory sequence, not measured execution time','readiness':{'concat':23.8,'mlp':25.2,'assembly':30}},indent=2));(A/'animation-routes.json').write_text(json.dumps(routes,indent=2));print('Built example:',fids,'crop',crop)

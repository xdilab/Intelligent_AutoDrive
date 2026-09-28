from pathlib import Path
import hashlib
A=Path(__file__).resolve().parent
ref=A.parent/'stage6-worked-example/build.py'
s=ref.read_text();exec(compile(s.split('# Presentation timeline:')[0].replace("example=json.loads", "example=json.loads"),str(ref),'exec'))
# Clear example provenance: this figure describes the new implementation.
svg[0]=svg[0].replace('1450','1700');svg[1]=svg[1].replace('1450','1700');gm.set('pageHeight','1700');diag.set('name','Contextual residual RoI classifier')
def link(a,b,pts,t,d=.8):edge(a,b,pts,phase=(t,d))
def train(id,x,y,w,h,lines):
 box(id,x,y,w,h,lines,fill='#f8cecc',stroke='#b85450',kind='trained',size=22);badge('fire-'+id,'trained',x+w-35,y-18)
text('Contextual residual RoI classifier',45,58,40,True)
text('Implemented study: frozen multiview features + residual visual/text adapters + contrastive RoI learning',45,103,25)
for id,x,w,title in [('p1',40,720,'1. Frozen visual feature extraction'),('p2',820,720,'2. Frozen language feature extraction'),('p3',1600,760,'3. Classification and supervision')]:
 box(id,x,175,w,445,fill='#ffffff',stroke='#26323d',container=True);text(title,x+15,152,28,True)
box('clips',65,235,250,145,['Whole-scene clip','+ actor crop clip','Each: 8 × 3 × 224 × 224'],fill='#d5e8d4',stroke='#82b366',size=22)
box('video-tower',400,250,315,110,['V₀','InternVideo2-CLIP-S'],kind='frozen',size=27);badge('ice-video','frozen',680,228)
box('visual-cache',65,450,650,115,['Crop features [1024] • context RoI [1024]','Scene tokens [16 × 1024] • box geometry [8]','RoIAlign on full-scene tokens; geometry computed separately'],fill='#e8f1f9',stroke='#4b88b5',size=22)
text('Shared encoder weights; separate clip passes; offline cache',72,596,21)
link('clips','video-tower',[(315,305),(400,305)],.3);link('video-tower','visual-cache',[(557,360),(557,450)],1.8)
box('phrases',850,230,265,130,['All 184 label phrases','Agent / action / location','Duplex / triplet'],fill='#f1edf7',stroke='#8064a2',size=23)
box('text-tower',1220,240,265,110,['L₀','Text encoder'],kind='frozen',size=28);badge('ice-text','frozen',1450,218)
box('fixed-p',1000,455,320,75)
text('Complete bank at inference; no target phrase supplied',850,596,22)
link('phrases','text-tower',[(1115,295),(1220,295)],3);link('text-tower','fixed-p',[(1352,350),(1352,490),(1320,490)],4.4,1)
train('classifier',1640,225,675,105,['Final MLP: 1208 → 512 → GELU → 184','Input = joint RoI [512] + c [512] + phrase scores [184]'])
box('scores',1640,385,675,75,['184 logits: 11 agentness/agent + 22 action + 16 location','+ 49 duplex + 86 triplet'],fill='#e3f3f5',stroke='#55969e',size=22)
box('loss',1640,515,675,75,['L = focal(184 labels) + λ Lcontrastive(86 triplets)','λ ∈ {0, 0.001}; contrastive temperature τ = 0.07'],fill='#fff2cc',stroke='#d6b656',size=23)
link('classifier','scores',[(1980,330),(1980,385)],25);link('scores','loss',[(1980,460),(1980,515)],26)
box('detail',40,715,2320,755,fill='#ffffff',stroke='#26323d',container=True)
text('Learned RoI: visual fusion → language conditioning',450,695,29,True)
# Two clean horizontal streams; skip paths occupy dedicated lanes.
vector('crop',80,800,245,40,label='Crop features [1024]')
train('crop-adapter',405,775,320,90,['Residual visual adapter','c = Linear(x + MLP(x))'])
train('evidence',80,985,300,160,['Evidence projections','Context RoI: 1024 → 512','Scene: 16 × (1024 → 512)','Geometry: 8 → 128 → 512'])
train('fusion',485,975,325,170,['Visual fusion MLP','2048 → 512 → 512','MLP: mean scene tokens','Variant: actor-query attention'])
vector('visual-roi',930,815,260,50,label='Visual RoI [512]')
text('Visual RoI: v = LN(c + evidence)',1390,1265,21)
box('contrastive',900,985,320,160,['Visual-only contrastive loss','cos(v, fixed phrases) / τ','Multi-positive triplet targets','Before language conditioning'],fill='#fff2cc',stroke='#d6b656',size=22)
train('language',1325,790,375,125,['Language cross-attention','Query: visual RoI [512]','Keys/values: 184 × 512'])
vector('joint-roi',1880,815,325,50,label='Joint RoI features [512]')
text('Joint RoI: h = LN(v + sigmoid(s) × language)',1390,1230,21)
box('final-inputs',1800,995,480,150,['To final MLP: [h; c; cosine(v, T)]','512 + 512 + 184 = 1208','Preserved visual skip c','Phrase scores use adapted bank T'],fill='#fff2cc',stroke='#d6b656',size=23)
box('text-bank',80,1270,300,110,['Fixed phrase matrix P','184 × 512'],fill='#f1edf7',stroke='#8064a2',size=25)
train('text-adapter',490,1270,340,110,['Residual text adapter','T = P + MLP(P)','512 → 128 → 512'])
box('adapted-bank',960,1270,310,110,['Adapted phrase bank T','184 × 512'],fill='#f1edf7',stroke='#8064a2',size=25)
text('Residual adapters: LayerNorm + GELU; visual bottleneck 128.',1390,1300,21)
text('Visual residual output [1024] projects to c [512].',1390,1335,21)
text('Attention variant: query c + geometry; 16 scene keys/values.',1390,1370,21)
text('Both attention modules use 8 heads.',1390,1405,21)
text('Frozen caches; gradients update the RoI head and adapters.',1390,1440,21)
link('p1','detail',[(150,620),(150,715)],5.8)
link('p2','detail',[(1490,620),(1490,715)],5.8)
link('crop','crop-adapter',[(325,820),(405,820)],7)
link('crop-adapter','fusion',[(640,865),(640,975)],8.5)
link('evidence','fusion',[(380,1060),(485,1060)],8.5)
link('fusion','visual-roi',[(810,1060),(855,1060),(855,840),(930,840)],10,1)
link('crop-adapter','visual-roi',[(725,820),(810,820),(810,745),(1170,745),(1170,815)],10,1)
link('visual-roi','contrastive',[(1060,865),(1060,985)],11.5)
link('text-bank','contrastive',[(230,1270),(230,1170),(1060,1170),(1060,1145)],11.5)
link('text-bank','text-adapter',[(380,1325),(490,1325)],13)
link('text-adapter','adapted-bank',[(830,1325),(960,1325)],14.5)
link('adapted-bank','language',[(1270,1325),(1300,1325),(1300,950),(1510,950),(1510,915)],16,1)
link('visual-roi','language',[(1190,840),(1325,840)],16,1)
link('language','joint-roi',[(1700,840),(1880,840)],18)
link('visual-roi','joint-roi',[(1150,815),(1150,750),(2190,750),(2190,815)],18,1)
link('joint-roi','final-inputs',[(2040,865),(2040,995)],20)
# Collector names all retained inputs; dedicated skip lane stays above the text stream.
link('crop-adapter','final-inputs',[(725,850),(835,850),(835,1200),(1760,1200),(1760,1070),(1800,1070)],20,2)
link('detail','p3',[(2300,715),(2300,620)],23,1)
text('Training: GT-positive crops plus sampled negatives. Evaluation: fixed YOLO candidate boxes; preserve detector agent outputs.',50,1520,23)
text('Study compares MLP vs actor-query attention, with/without contrastive loss, across three seeds. No new accuracy result is implied.',50,1560,23)
badge('legend-ice','frozen',50,1590);text('Dashed + ice: frozen encoder',105,1620,23)
badge('legend-fire','trained',550,1590);text('Thick + fire: trained module',605,1620,23)
text('Blue: visual features    Purple: phrase features    Gold: assembly / losses',1100,1620,23)
text('Gold moving dots show feature forward flow, not gradient propagation. Timing is illustrative.',50,1670,21,color='#65717b')
static=''.join(svg)+'</svg>';(A/'architecture-static.svg').write_text(static)
E.indent(root);E.ElementTree(root).write(A/'architecture.drawio',encoding='utf-8',xml_declaration=True)
# Same sequential gold packets, warm edges, active-only badges and borders as the stage figures.
D=30
svg.append('<defs><filter id="glow"><feGaussianBlur stdDeviation="4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>')
def pulse(start,end):return f'<animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;{start/D};{(start+.1)/D};{(end-.1)/D};{end/D};1" dur="30s" repeatCount="indefinite"/>'
windows={'ice-video':(1.1,1.8),'ice-text':(3.8,4.4),'fire-crop-adapter':(7.8,8.5),'fire-evidence':(7.8,8.5),'fire-fusion':(9.3,10),'fire-text-adapter':(13.8,14.5),'fire-language':(17,18),'fire-classifier':(24,25)}
for b in badges:
 if b['id'] in windows:
  t,e=windows[b['id']];svg.append(f'<circle cx="{b["x"]+20}" cy="{b["y"]+20}" r="24" fill="none" stroke="'+('#39a8ed' if b['kind']=='frozen' else '#ff981e')+f'" stroke-width="5" filter="url(#glow)" opacity="0">{pulse(t,e)}</circle>')
for b in trained:
 t,e=windows['fire-'+b['id']];svg.append(f'<rect x="{b["x"]}" y="{b["y"]}" width="{b["w"]}" height="{b["h"]}" rx="15" fill="none" stroke="#e66a61" stroke-width="5" opacity="0" filter="url(#glow)">{pulse(t,e)}</rect>')
for node,t,e in [('p1',.2,2.8),('p2',3,5.5),('detail',6.8,23),('p3',24,27.5)]:
 x,y,w,h=positions[node];svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="15" fill="none" stroke="#e9ad35" stroke-width="4" opacity="0" filter="url(#glow)">{pulse(t,e)}</rect>')
for r in routes:
 t=r['phase']*1.5;e=t+r['duration'];p=rounded_route(r['points'])[0]
 svg.append(f'<path d="{p}" fill="none" stroke="#ff981e" stroke-width="4" opacity="0" filter="url(#glow)">{pulse(t,e)}</path>')
 svg.append(f'<circle r="8" fill="#FDB927" stroke="#fff2be" stroke-width="2" filter="url(#glow)" opacity="0">{pulse(t,e)}<animateMotion dur="30s" repeatCount="indefinite" keyTimes="0;{t/D};{e/D};1" keyPoints="0;0;1;1" calcMode="linear" path="{p}"/></circle>')
animated=''.join(svg)+'</svg>';(A/'architecture-animated.svg').write_text(animated)
player=s[s.index("player='''"):s.index("(A/'stage6-animation.html')")]
# Retain reference player's tested browser controls, replace its narrative.
player=player.replace('Stage 6: one rare bus, end to end','Contextual residual RoI classifier').replace('34','30')
a=player.index('const steps=');b=player.index(';svg.pauseAnimations()',a)
player=player[:a]+"const steps=[[0,'1 · Extract frozen visual features'],[3,'2 · Encode the complete phrase bank'],[5.8,'3 · Load cached evidence'],[7,'4 · Residual visual adaptation and context fusion'],[11.5,'5 · Contrastive supervision on visual RoI'],[13,'6 · Residual text adaptation'],[16,'7 · Language attention and residual joint RoI'],[20,'8 · Collect joint, visual and phrase evidence'],[23,'9 · Final classification and training objective'],[27.5,'Complete · Frozen encoders; trainable RoI head']]"+player[b:]
exec(player);(A/'architecture-animation.html').write_text(player)
model=Path('/data/repos/ROAD_Reason/research/post-proposal/contextual-roi/model.py')
(A/'provenance.json').write_text(json.dumps({'model':str(model),'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),'style_reference':str(ref),'scope':'Implemented frozen-cache contextual residual head; no performance claim','created':'2026-09-17','routes':routes},indent=2))
print('Built contextual architecture')

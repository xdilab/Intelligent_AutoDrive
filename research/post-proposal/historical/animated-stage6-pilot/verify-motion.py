from pathlib import Path
from playwright.sync_api import sync_playwright
import json
A=Path(__file__).resolve().parent
with sync_playwright() as p:
 b=p.chromium.launch(headless=True,args=['--no-sandbox']);page=b.new_page(viewport={'width':2400,'height':1450});page.goto((A/'stage6-animated.svg').as_uri())
 pos=[]
 for t in [.1,.4,5.5]:
  v=page.evaluate('''t=>{const s=document.querySelector('svg');s.pauseAnimations();s.setCurrentTime(t);return [...document.querySelectorAll('svg > circle')].map(c=>({x:c.getCTM().e,y:c.getCTM().f,opacity:getComputedStyle(c).opacity}));}''',t);pos.append(v)
 assert abs(pos[0][0]['x']-pos[1][0]['x'])>20, pos
 routes=json.loads((A/'animation-routes.json').read_text())
 maxima={'p1':0,'p3':0}
 for step in range(360):
  t=step/20
  visible=page.evaluate("t=>{document.querySelector('svg').setCurrentTime(t);return [...document.querySelectorAll('svg > circle')].map(c=>+getComputedStyle(c).opacity>0)}",t)
  for panel in maxima:
   count=sum(on and r['panel']==panel for on,r in zip(visible,routes))
   maxima[panel]=max(maxima[panel],count)
   assert count<=2,(t,panel,count)
 assert maxima=={'p1':2,'p3':2},maxima
 (A/'packet-count-check.json').write_text(json.dumps({'sample_interval_seconds':.05,'full_loop_seconds':18,'maximum_visible_dots_per_panel':maxima},indent=2)+'\n')
 print('Full-loop packet count verified:',maxima)
 glow=[]
 for t in [0,.65,3]:
  glow.append(page.evaluate("t=>{const s=document.querySelector('svg');s.setCurrentTime(t);return getComputedStyle(document.querySelector('.status-glow circle')).opacity}",t))
 assert float(glow[0])==0 and float(glow[1])>.4 and float(glow[2])==0,glow
 page.evaluate("document.querySelector('svg').setCurrentTime(.65)")
 page.screenshot(path=str(A/'glow-preview.png'))
 page.evaluate("document.querySelector('svg').setCurrentTime(5.5)")
 page.screenshot(path=str(A/'browser-motion-preview.png'))
 for panel,t in [('block1',1.0),('block3',12.95)]:
  page.evaluate("t=>document.querySelector('svg').setCurrentTime(t)",t)
  page.screenshot(path=str(A/f'{panel}-packet-preview.png'))
 states=[]
 for t in [0,3.7,6,13.825,15]:
  states.append(page.evaluate("t=>{document.querySelector('svg').setCurrentTime(t);return Object.fromEntries([...document.querySelectorAll('.trainable-border')].map(g=>[g.dataset.node,+getComputedStyle(g).opacity]))}",t))
 assert set(states[0])=={'vstar','lstar','v-last','l-last','stage6'},states
 assert not any(states[0].values()) and not any(states[2].values()) and not any(states[4].values()),states
 assert all(states[1][k]>.9 for k in ['vstar','lstar','v-last','l-last']) and states[1]['stage6']==0,states
 assert states[3]['stage6']>.9 and states[3]['vstar']==0,states
 page.evaluate("document.querySelector('svg').setCurrentTime(3.7)")
 page.screenshot(path=str(A/'trainable-border-preview.png'))
 (A/'border-glow-check.json').write_text(json.dumps({'times':[0,3.7,6,13.825,15],'opacity':states},indent=2)+'\n')
 print('Trainable border off/on/off verified; frozen borders excluded')
 heat=[]
 for t in [.3,.65,1.0,2.95]:
  heat.append(page.evaluate("t=>{document.querySelector('svg').setCurrentTime(t);return [...document.querySelectorAll('.hot-flow-arrow')].map(p=>+getComputedStyle(p).opacity)}",t))
 assert heat[0][0]>.9 and heat[0][2]==0 and not any(heat[1]) and heat[2][2]>.9 and heat[2][0]==0 and not any(heat[3]),heat
 page.evaluate("document.querySelector('svg').setCurrentTime(1.0)")
 page.screenshot(path=str(A/'hot-arrow-preview.png'))
 (A/'arrow-heat-check.json').write_text(json.dumps({'times':[.3,.65,1.,2.95],'opacity':heat,'meaning':'active forward signal flow, not backpropagation'},indent=2)+'\n')
 print('Arrow heat follows sequential packet phases and cools between them')
 sections=[]
 for t in [0,.65,4,11,13,16]:
  sections.append(page.evaluate("t=>{document.querySelector('svg').setCurrentTime(t);return Object.fromEntries([...document.querySelectorAll('.section-glow')].map(g=>[g.dataset.node,+getComputedStyle(g).opacity]))}",t))
 assert not any(sections[0].values()) and sections[1]['p1']>.7 and sections[1]['p2']==0,sections
 assert sections[2]['p2']>.7 and sections[2]['detail']>.7 and sections[2]['p3']==0,sections
 assert not any(sections[3].values()) and sections[4]['p3']>.7 and not any(sections[5].values()),sections
 page.evaluate("document.querySelector('svg').setCurrentTime(4)")
 page.screenshot(path=str(A/'section-glow-preview.png'))
 (A/'section-glow-check.json').write_text(json.dumps({'times':[0,.65,4,11,13,16],'opacity':sections},indent=2)+'\n')
 print('Major panel glow follows active phases')
 b.close()
 print('Icon glow verified:',glow)
 (A/'motion-check.json').write_text(json.dumps({'sample_times':[.1,.4,5.5],'positions':pos,'verified':'First packet advances along its SVG path; playback controls checked separately.'},indent=2)+'\n')
 print('SVG motion verified:',pos[0][0],pos[1][0])

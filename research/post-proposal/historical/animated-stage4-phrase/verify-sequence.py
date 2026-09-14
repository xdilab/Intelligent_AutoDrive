from pathlib import Path
import json
from playwright.sync_api import sync_playwright
A=Path(__file__).resolve().parent
routes=json.loads((A/'animation-routes.json').read_text())
by_pair={(r['source'],r['target']):r for r in routes}
assert ('p2','p3') not in by_pair
assert ('p1','p2') not in by_pair
assert ('p1','p3') in by_pair
assert ('p2','detail') in by_pair and ('detail','p3') in by_pair
order=[('frame','yolo'),('yolo','candidates'),('p1','p3'),('road-crop','vstar'),('vstar','features'),('p2','detail'),('x','projection'),('projection','matrix'),('matrix','scale'),('scale','phrase-scores'),('detail','p3'),('phrase-input','assembly'),('assembly','out')]
for prev,nxt in zip(order,order[1:]):
 a,b=by_pair[prev],by_pair[nxt];assert a['phase']*1.5+a['duration']<=b['phase']*1.5+1e-8,(prev,nxt)
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page(viewport={'width':2400,'height':1550})
 page.goto((A/'stage4-animated.svg').as_uri());page.evaluate('document.querySelector("svg").pauseAnimations()')
 maximum=0;observed=[]
 for step in range(360):
  t=step/20
  state=page.evaluate('''t=>{document.querySelector('svg').setCurrentTime(t);return {panels:[...document.querySelectorAll('.section-glow')].filter(e=>+getComputedStyle(e).opacity>.01).map(e=>e.dataset.node), packets:[...document.querySelectorAll('circle')].filter(e=>e.querySelector('animateMotion')).map(e=>+getComputedStyle(e).opacity>.01)}}''',t)
  assert len(state['panels'])<=1,(t,state)
  active={r['panel'] for r,on in zip(routes,state['packets']) if on and r['panel']!='1'}
  assert len(active)<=1,(t,active)
  assert sum(state['packets'])<=2,(t,state)
  maximum=max(maximum,sum(state['packets']))
  if state['panels'] and (not observed or observed[-1]!=state['panels'][0]):observed.append(state['panels'][0])
 assert observed==['p1','p2','detail','p3'],observed
 for label,t in [('part2',3.3),('detail',6.4),('return',11.3),('part3',12.4),('active-4',4),('active-12.3',12.3)]:
  page.evaluate('t=>document.querySelector("svg").setCurrentTime(t)',t);page.screenshot(path=str(A/(f'{label}.png' if label.startswith('active-') else f'sequence-{label}.png')))
 browser.close()
report={'sequence':observed,'maximum_simultaneous_packets':maximum,'sample_interval_seconds':.05,'loop_seconds':18,'dependent_routes_sequential':True,'part2_and_detail_overlap':False,'downward_then_upward_route':True}
(A/'motion-check.json').write_text(json.dumps(report,indent=2)+'\n');print(report)

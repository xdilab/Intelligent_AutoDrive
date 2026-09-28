from pathlib import Path
import json
from playwright.sync_api import sync_playwright
A=Path(__file__).resolve().parent.parent
with sync_playwright() as p:
 b=p.chromium.launch(headless=True,args=['--no-sandbox']);page=b.new_page(viewport={'width':2400,'height':1450})
 page.goto((A/'stage6-static.svg').as_uri());page.screenshot(path=str(A/'stage6-static.png'))
 errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto((A/'stage6-animation.html').as_uri());page.wait_for_timeout(350);assert page.evaluate('t')>0;assert not errors,errors
 page.locator('#pause').click()
 for sec in [0,5.9,10.2,11.8,17.4,20,24,25.5,29.6,31.8]:
  page.evaluate('(value)=>{t=value;svg.setCurrentTime(t)}',sec);page.wait_for_timeout(50);page.screenshot(path=str(A/f'visual-review/phase-{sec}.png'))
 page.evaluate('(value)=>{t=value;svg.setCurrentTime(t)}',17.4)
 page.wait_for_timeout(50)
 visible=page.locator('svg > circle').evaluate_all('(nodes)=>nodes.filter(n=>+getComputedStyle(n).opacity>0.5).length');assert visible==2,visible
 page.evaluate('(value)=>{t=value;svg.setCurrentTime(t)}',0)
 page.wait_for_timeout(50)
 assert page.locator('svg > circle').evaluate_all('(nodes)=>nodes.filter(n=>+getComputedStyle(n).opacity>0.5).length')==0
 page.locator('#speed').select_option('1.5');page.locator('#restart').click();page.locator('#pause').click();b.close()
r=json.loads((A/'animation-routes.json').read_text());major=[(x['source'],x['target']) for x in r if x['source'] in ['p1','p2','detail']]
assert major==[('p1','p2'),('p2','detail'),('detail','p3')]
assert all(len(x['points'])==2 for x in r if (x['source'],x['target']) in major)
maxpack=max(sum(x['phase']*1.5<=t<x['phase']*1.5+x['duration'] for x in r) for t in [i/100 for i in range(3400)]);assert maxpack<=2
(A/'visual-review/checks.json').write_text(json.dumps({'major_panel_arrows':major,'all_major_arrows_straight':True,'max_simultaneous_packets':maxpack,'player_controls':'passed'},indent=2));print('Three straight panel connectors; max two packets; player controls passed')

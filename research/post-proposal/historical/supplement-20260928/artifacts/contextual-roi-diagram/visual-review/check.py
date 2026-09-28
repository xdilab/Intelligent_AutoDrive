from pathlib import Path
from playwright.sync_api import sync_playwright
import json
A=Path(__file__).resolve().parent.parent
with sync_playwright() as p:
 b=p.chromium.launch(headless=True,args=['--no-sandbox']);page=b.new_page(viewport={'width':2400,'height':1700});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto((A/'architecture-static.svg').as_uri());page.screenshot(path=str(A/'architecture-static.png'))
 page.goto((A/'architecture-animation.html').as_uri());page.wait_for_timeout(500);assert page.evaluate('t')>0;page.locator('#pause').click()
 for sec in [8.8,10.4,16.4,20.5,24.5]:
  page.evaluate('(v)=>{t=v;svg.setCurrentTime(v)}',sec);page.wait_for_timeout(100);page.screenshot(path=str(A/f'visual-review/phase-{sec}.png'))
 page.locator('#speed').select_option('1.5');page.locator('#restart').click();page.locator('#pause').click();assert not errors,errors;b.close()
r=json.loads((A/'provenance.json').read_text())['routes'];n=max(sum(x['phase']*1.5<=t<x['phase']*1.5+x['duration'] for x in r) for t in [i/100 for i in range(3000)]);assert n<=2
(A/'visual-review/checks.json').write_text(json.dumps({'player_errors':errors,'max_packets':n,'controls':'passed'},indent=2));print('Animation passed, max packets',n)

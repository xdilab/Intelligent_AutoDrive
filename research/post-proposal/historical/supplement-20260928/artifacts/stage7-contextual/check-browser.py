from pathlib import Path
import json
from playwright.sync_api import sync_playwright
A=Path(__file__).resolve().parent
with sync_playwright() as p:
 b=p.chromium.launch(headless=True,args=['--no-sandbox']);page=b.new_page(viewport={'width':1600,'height':1100});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto('http://127.0.0.1:8887/stage7-contextual/index.html');page.wait_for_load_state('networkidle');page.screenshot(path=str(A/'browser-aligned.png'))
 page.click('[data-tab="detailed"]');page.evaluate("document.querySelector('#animation svg').pauseAnimations();document.querySelector('#animation svg').setCurrentTime(16)");page.screenshot(path=str(A/'browser-detailed.png'),full_page=True)
 for t in [0,8.5,13.9,20,23,33.4,36]:
  page.evaluate('(t)=>document.querySelector("#animation svg").setCurrentTime(t)',t)
 page.click('#restart');page.click('#play');assert page.evaluate('document.querySelector("#animation svg").animationsPaused()')
 page.click('[data-tab="compare"]');assert page.locator('#compare').is_visible()
 (A/'browser-check.json').write_text(json.dumps({'page_errors':errors,'tabs':'aligned/detailed/compare passed','controls':'restart/pause passed','http':200},indent=2));assert not errors
 b.close()

from pathlib import Path
from playwright.sync_api import sync_playwright
import json
A=Path(__file__).resolve().parent
with sync_playwright() as p:
 b=p.chromium.launch(headless=True,args=['--no-sandbox']);page=b.new_page(viewport={'width':1920,'height':1080});page.goto((A/'stage5-animated.svg').as_uri())
 pos=[]
 for t in [.2,.9,5.5]:
  v=page.evaluate('''t=>{const s=document.querySelector('svg');s.pauseAnimations();s.setCurrentTime(t);return [...document.querySelectorAll('circle')].map(c=>({x:c.getCTM().e,y:c.getCTM().f,opacity:getComputedStyle(c).opacity}));}''',t);pos.append(v)
 assert abs(pos[0][0]['x']-pos[1][0]['x'])>20, pos
 page.screenshot(path=str(A/'browser-motion-preview.png'));b.close()
 (A/'motion-check.json').write_text(json.dumps({'sample_times':[.2,.9,5.5],'positions':pos,'verified':'First packet advances along its SVG path; playback controls checked separately.'},indent=2)+'\n')
 print('SVG motion verified:',pos[0][0],pos[1][0])

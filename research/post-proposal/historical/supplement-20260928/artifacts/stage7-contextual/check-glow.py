from pathlib import Path
import json
from PIL import Image
from playwright.sync_api import sync_playwright
A=Path(__file__).resolve().parent
result={}
with sync_playwright() as p:
 b=p.chromium.launch(headless=True,args=['--no-sandbox']);page=b.new_page(viewport={'width':1600,'height':1100},device_scale_factor=1)
 for label,url in [('before','arrow-review-before/index.html'),('after','index.html')]:
  page.goto('http://127.0.0.1:8887/stage7-contextual/'+url);page.click('[data-tab="detailed"]')
  s=page.locator('#animation svg');page.evaluate('document.querySelector("#animation svg").pauseAnimations()')
  vb=page.evaluate('document.querySelector("#animation svg").viewBox.baseVal.width')
  for orientation,t,region in [('horizontal',.9,(330,308,355,332)),('vertical',2.25,(553,390,577,410))]:
   page.evaluate('(t)=>document.querySelector("#animation svg").setCurrentTime(t)',t);page.wait_for_timeout(100)
   path=A/f'glow-{label}-{orientation}.png';s.screenshot(path=str(path));im=Image.open(path).convert('RGB');k=im.width/vb
   crop=im.crop(tuple(round(v*k) for v in region));crop.save(A/f'glow-{label}-{orientation}-crop.png')
   hot=sum(r>180 and 65<g<215 and bb<110 and r>g+25 for r,g,bb in crop.getdata());result[f'{label}-{orientation}']={'orange_wire_pixels_away_from_dot':hot}
 assert result['after-horizontal']['orange_wire_pixels_away_from_dot']>result['before-horizontal']['orange_wire_pixels_away_from_dot']+10
 assert result['after-vertical']['orange_wire_pixels_away_from_dot']>result['before-vertical']['orange_wire_pixels_away_from_dot']+10
 b.close()
(A/'glow-check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

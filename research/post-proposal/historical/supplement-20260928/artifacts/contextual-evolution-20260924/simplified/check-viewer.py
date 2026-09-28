from pathlib import Path
import json
from playwright.sync_api import sync_playwright

A=Path(__file__).resolve().parent
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':1600,'height':1100},device_scale_factor=1)
 errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto((A/'index.html').as_uri());page.wait_for_function('document.getElementById("diagram").naturalWidth > 0')
 assert page.locator('#tabs button').count()==6
 checks=[]
 for n,num in enumerate([88,89,90,92,94,104]):
  page.locator('#tabs button').nth(n).click();page.wait_for_function('document.getElementById("diagram").complete')
  assert 'model '+str(num) in page.locator('#number').inner_text().lower(), (num,page.locator('#number').inner_text(),errors)
  assert page.locator('#comparison thead th').count()==10
  assert page.locator('#comparison tbody tr').count()==(1 if num==88 else 3)
  assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
  assert page.evaluate('''() => {
   const f=F[current],p=f.positions;
   if(!p.decoder) return true;
   const cy=b=>b[1]+b[3]/2;
   return cy(p.decoder)===(cy(p.visual)+cy(p.adapted))/2 &&
    p.fusion[0]===p['text-adapter'][0] &&
    p.fusion[2]===p['text-adapter'][2] && p.fusion[3]===p['text-adapter'][3] &&
    f.schedule.findIndex(r=>r.id==='scene-summary-out')<f.schedule.findIndex(r=>r.id==='features-head');
  }''')
  page.locator('#diagram-panel').screenshot(path=str(A/f'html-model-{num}.png'))
  checks.append({'model':num,'image_loaded':True,'all_nine_metrics':True,'no_page_overflow':True})
 page.locator('#all-models').click();assert page.locator('#all-table tbody tr').count()==13
 assert '13.428' in page.locator('#comparison').inner_text()
 page.locator('#animate').click();page.wait_for_timeout(550)
 visible=page.locator('#packet').get_attribute('opacity')
 assert visible=='1',visible
 page.locator('#diagram-panel').screenshot(path=str(A/'html-animation.png'))
 page.locator('#animate').click();assert page.locator('#packet').get_attribute('opacity')=='0'
 page.locator('#expand').click();assert page.locator('body.full').count()==1
 page.keyboard.press('Escape');assert page.locator('body.full').count()==0
 page.evaluate("show(3);playing=true;epoch=performance.now()-(F[3].schedule.findIndex(r=>r.target==='decoder')*1.35+1.15)*1000;tick(performance.now())")
 assert page.locator('#attention-glow').get_attribute('opacity')=='0.7'
 assert page.locator('#active').get_attribute('opacity')=='0'
 page.evaluate('playing=false;clearFlow()')
 assert page.locator('#attention-glow').get_attribute('opacity')=='0'
 page.set_viewport_size({'width':390,'height':844});page.reload();page.wait_for_function('document.getElementById("diagram").naturalWidth > 0')
 assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
 assert not errors,errors
 browser.close()
(A/'browser-check.json').write_text(json.dumps({'passed':True,'models':checks,'all_audited_models':13,'animation_one_packet':True,'pause_clears_glow':True,'enlarge_escape':True,'mobile_no_overflow':True,'javascript_errors':errors},indent=2))
print('Browser checks passed for all six diagrams, nine metrics, 13-condition table, animation, pause, enlarged view and mobile layout.')

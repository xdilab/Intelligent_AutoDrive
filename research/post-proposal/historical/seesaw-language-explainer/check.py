import json
from pathlib import Path
from playwright.sync_api import sync_playwright
A=Path(__file__).resolve().parent;errors=[]
with sync_playwright() as p:
 b=p.chromium.launch(headless=True,args=['--no-sandbox']);page=b.new_page(viewport={'width':1440,'height':1000});page.on('pageerror',lambda e:errors.append(str(e)));page.goto((A/'seesaw-language.html').as_uri());page.wait_for_function('window.REPORT_READY===true')
 assert page.locator('section').count()==6
 page.locator('[data-weight="0"]').first.click();assert page.locator('#toyap').inner_text()=='50%';assert page.locator('#pct5').text_content()=='100.00%';zero=float(page.locator('#toybce').inner_text())
 page.locator('[data-weight="75"]').first.click();assert page.locator('#toyap').inner_text()=='100%';assert float(page.locator('#toybce').inner_text())>zero
 page.locator('[data-weight="0.15"]').click();assert page.locator('#pctL').text_content()=='0.15%'
 page.locator('#play').click();page.wait_for_timeout(1700);assert page.locator('#play').inner_text()=='Pause animation';page.locator('#play').click();assert page.locator('#play').inner_text()=='Animate balance'
 page.select_option('#seed','0');page.select_option('#expert','phrase');page.locator('#gridweight').fill('11');page.locator('#gridweight').dispatch_event('input');assert '15.869' in page.locator('#realvalues').inner_text()
 page.select_option('#expert','shuffled');assert '15.869' not in page.locator('#realvalues').inner_text();page.select_option('#expert','phrase');page.select_option('#seed','mean')
 with page.expect_download() as dl:page.locator('#download').click()
 assert dl.value.suggested_filename=='seesaw-evidence.json'
 page.locator('[data-weight="75"]').first.click();page.locator('#balance').scroll_into_view_if_needed();page.screenshot(path=str(A/'desktop-seesaw.png'));page.locator('#evidence').scroll_into_view_if_needed();page.screenshot(path=str(A/'desktop-evidence.png'));assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
 page.set_viewport_size({'width':390,'height':844});page.goto((A/'seesaw-language.html').as_uri());page.wait_for_function('window.REPORT_READY===true');assert page.evaluate('document.documentElement.scrollWidth<=innerWidth');page.locator('#balance').scroll_into_view_if_needed();page.screenshot(path=str(A/'mobile-seesaw.png'));page.locator('#evidence').scroll_into_view_if_needed();page.screenshot(path=str(A/'mobile-evidence.png'));assert page.evaluate('document.documentElement.scrollWidth<=innerWidth');assert not errors,errors;b.close()
(A/'browser-check.json').write_text(json.dumps({'passed':True,'javascript_errors':errors,'checks':['offline self-contained loading','six sections from big picture to math','seesaw weights and toy AP/BCE behavior','animation play/pause','real seed/expert/grid controls match measured audit values','JSON evidence download','desktop/mobile no horizontal page overflow'],'viewports':[1440,390]},indent=2));print('PASS: seesaw interactions, evidence values and responsive layout')

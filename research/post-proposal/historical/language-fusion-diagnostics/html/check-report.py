import json
from pathlib import Path
from playwright.sync_api import sync_playwright
A=Path(__file__).resolve().parent.parent;H=A/'html';errors=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
 page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto((A/'language-fusion-analysis.html').as_uri(),wait_until='load');page.wait_for_function('window.REPORT_READY === true')
 assert page.locator('#class-table tbody tr').count()==86
 assert page.locator('#trace-table tbody tr').count()==6
 assert page.locator('#parity-table tbody tr').count()==72
 assert page.locator('#case-view img').evaluate('(i)=>i.complete && i.naturalWidth>0')
 page.screenshot(path=str(H/'desktop-overview.png'),full_page=False)
 page.locator('#diagnostics').scroll_into_view_if_needed();page.screenshot(path=str(H/'desktop-traces.png'))
 page.select_option('#seed',value='0');page.select_option('#budget','1');assert '204' in page.locator('#trace-table').inner_text()
 page.select_option('#seed','2');page.select_option('#budget','0.5');assert '106' in page.locator('#trace-table').inner_text()
 page.select_option('#seed','mean');page.select_option('#budget','1')
 page.fill('#search','LarVeh-Stop-Jun');assert page.locator('#class-table tbody tr').count()==1
 page.select_option('#comparator','phrase_minus_flat');page.fill('#search','')
 page.select_option('#case-class','Bus-Stop-VehLane');page.select_option('#case-category','phrase_help_preserved');assert 'Yes' in page.locator('#case-view').inner_text()
 assert page.locator('#case-view img').evaluate('(i)=>i.complete && i.naturalWidth>0')
 page.locator('#case-view').scroll_into_view_if_needed();page.screenshot(path=str(H/'desktop-case.png'))
 page.locator('[data-zoom="lost"]').click();assert page.locator('#zoom-dialog').is_visible();page.locator('#close-dialog').click();assert not page.locator('#zoom-dialog').is_visible()
 with page.expect_download() as dl:page.locator('[data-download="budget-summary.csv"]').click()
 assert dl.value.suggested_filename=='budget-summary.csv'
 assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth'), 'Desktop horizontal overflow'
 page.set_viewport_size({'width':390,'height':844});page.goto((A/'language-fusion-analysis.html').as_uri(),wait_until='load');page.wait_for_function('window.REPORT_READY === true');page.screenshot(path=str(H/'mobile-overview.png'));assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth'),'Mobile horizontal overflow'
 page.screenshot(path=str(H/'mobile-overview.png'));page.locator('#case-view').scroll_into_view_if_needed();page.screenshot(path=str(H/'mobile-case.png'))
 assert not errors,errors
 browser.close()
(H/'browser-check.json').write_text(json.dumps({'passed':True,'javascript_errors':errors,'desktop_width':1440,'mobile_width':390,'all_class_rows':86,'trace_rows':6,'ap_checks':72,'verified':['seed/budget controls','class search/comparator','case image switching','figure enlargement','embedded CSV download','no horizontal page overflow','offline file load']},indent=2)+'\n')
print('PASS: offline desktop/mobile rendering and interactive evidence checks')

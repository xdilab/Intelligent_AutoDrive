from pathlib import Path
from playwright.sync_api import sync_playwright
A=Path(__file__).resolve().parent
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':2400,'height':1450},device_scale_factor=1)
 page.goto((A/'stage6-static.svg').as_uri());page.screenshot(path=str(A/'stage6-static.png'))
 page.goto((A/'stage6-animation.html').as_uri());page.wait_for_timeout(250)
 page.locator('#pause').click();page.locator('#speed').select_option('1.5');page.locator('#restart').click()
 print('HTML controls: pause, speed, restart passed; SVG:',page.locator('svg').count())
 page.screenshot(path=str(A/'player-preview.png'))
 browser.close()

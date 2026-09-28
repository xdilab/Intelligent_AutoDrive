"""Render native draw.io SVGs at 4x; native CLI 4x PNG hit a tile limit."""
from pathlib import Path
import json,hashlib
from urllib.parse import quote
from PIL import Image,ImageChops,ImageStat,PngImagePlugin
from playwright.sync_api import sync_playwright

A=Path(__file__).resolve().parent
reports=[]
with sync_playwright() as p:
 b=p.chromium.launch(headless=True,args=['--no-sandbox'])
 page=b.new_page(viewport={'width':2001,'height':1081},device_scale_factor=4)
 for svg in sorted(A.glob('0*.svg')):
  page.goto(svg.as_uri());page.wait_for_load_state('load');page.wait_for_timeout(150)
  png=A/'exports'/(svg.stem+'.png')
  page.screenshot(path=str(png),full_page=True)
  with Image.open(png) as im:
   im.load();info=PngImagePlugin.PngInfo();info.add_text('mxGraphModel',quote((A/(svg.stem+'.drawio')).read_text(),safe="~()*!.'-"))
   info.add_text('Source','Native draw.io SVG rasterized at4x after draw.io PNG tile clipping.')
   im.save(png,pnginfo=info)
   rgb=im.convert('RGB');ref=Image.open(A/(svg.stem+'.png')).convert('RGB');small=rgb.resize(ref.size)
   diff=ImageStat.Stat(ImageChops.difference(small,ref)).mean
   assert max(diff)<5,(svg.name,diff)
   small.save(A/('export-review-'+svg.stem+'.png'))
   reports.append({'file':str(png),'size':im.size,'method':'native draw.io SVG; browser4x; embedded mxfile','mean_rgb_difference_to_reviewed_preview':diff,'sha256':hashlib.sha256(png.read_bytes()).hexdigest()})
 b.close()
assert len(reports)==6
(A/'export-check.json').write_text(json.dumps({'passed':True,'native_cli_failure':'4x PNG tile limit clipped bottom; replaced every affected export','exports':reports},indent=2))
print(json.dumps(reports,indent=2))

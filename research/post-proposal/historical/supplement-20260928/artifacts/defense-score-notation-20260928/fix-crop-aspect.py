from pathlib import Path
import xml.etree.ElementTree as E,re,shutil,json
p=Path(__file__).parent;b=p/'composition-aspect-backup';b.mkdir(exist_ok=True)
for ext in ['drawio','png']:
 if not (b/f'composition.{ext}').exists():shutil.copy2(p/f'composition.{ext}',b/f'composition.{ext}')
t=E.parse(p/'composition.drawio');c={x.get('id'):x for x in t.findall('.//mxCell')};ref={x.get('id'):x for x in E.parse(p/'stage5.drawio').findall('.//mxCell')};g=c['crops'].find('mxGeometry');r=ref['clips'].find('mxGeometry');scale=float(g.get('width'))/float(r.get('width'));h=float(r.get('height'))*scale;g.set('height',str(h));g.set('y',str(375-h/2))
for suffix in ['55','58','62']:
 dst=c['crops_clips-'+suffix].find('mxGeometry');src=ref['clips-'+suffix].find('mxGeometry')
 for k in ['x','y','width','height']:dst.set(k,str(float(src.get(k))*scale))
e=c['clip-enc'];e.set('style',re.sub(r'exitY=[^;]+;', 'exitY=0.5;',e.get('style')))
cg=c['clip-label'].find('mxGeometry');cg.set('y','555');cg.set('height','60')
E.indent(t);t.write(p/'composition.drawio',encoding='utf-8',xml_declaration=True)
(p/'composition-aspect-validation.json').write_text(json.dumps({'uniform_scale':scale,'group_width':280,'group_height':h,'source':'stage5.drawio clips group','source_image_ratio':float(ref['clips-55'].find('mxGeometry').get('width'))/float(ref['clips-55'].find('mxGeometry').get('height')),'images_unchanged':True,'arrow_y':375},indent=2))

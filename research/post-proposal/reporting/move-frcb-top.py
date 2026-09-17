from pathlib import Path
import zipfile,openpyxl,datetime,json,hashlib,os
from lxml import etree as E
p=Path('/home/brandon/Downloads/ROAD-Waymo Results.xlsx');w=openpyxl.load_workbook(p);s=w.worksheets[0];N='http://schemas.openxmlformats.org/spreadsheetml/2006/main';q=lambda x:'{'+N+'}'+x
assert s['B120'].value=='FRCB (MViT)' and s['B121'].value=='FRCB (MViT)' and s['A119'].value is None
raw=p.read_bytes();stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S');out=Path('/data/repos/wiki/artifacts/results-master');backup=out/f'ROAD-Waymo-Results-before-frcb-move-{stamp}.xlsx';backup.write_bytes(raw)
with zipfile.ZipFile(p) as z:parts={n:z.read(n) for n in z.namelist()}
r=E.fromstring(parts['xl/worksheets/sheet1.xml']);sd=r.find(q('sheetData'));byid={int(row.get('r')):row for row in sd};order=list(range(1,7))+[120,121]+list(range(7,119));sd[:]=[]
for new,old in enumerate(order,1):
 row=byid[old];row.set('r',str(new))
 for c in row:c.set('r',''.join(filter(str.isalpha,c.get('r')))+str(new))
 if old in [120,121]:
  c=next(c for c in row if c.get('r')==f'C{new}');v=s.cell(old,3).value+'\nCurrent SOTA reference for this comparison; external evaluation protocol.'
  for child in list(c):c.remove(child)
  c.set('t','inlineStr');E.SubElement(E.SubElement(c,q('is')),q('t')).text=v;row.set('ht','130')
 sd.append(row)
r.find(q('dimension')).set('ref','A1:T120');t=E.fromstring(parts['xl/tables/table1.xml']);t.set('ref','A1:T120');t.find(q('autoFilter')).set('ref','A1:T120');updates={'xl/worksheets/sheet1.xml':E.tostring(r),'xl/tables/table1.xml':E.tostring(t)}
tmp=p.with_name('ROAD-Waymo Results.updating.xlsx')
with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
 for n,b in {**parts,**updates}.items():z.writestr(n,b)
a=openpyxl.load_workbook(tmp);ss=a.worksheets[0]
for new,old in enumerate(order,1):
 for col in range(1,21):
  if old in [120,121] and col==3:continue
  assert ss.cell(new,col).value==s.cell(old,col).value,(old,col)
assert ss['B7'].value==ss['B8'].value=='FRCB (MViT)' and ss.tables['Table5'].ref=='A1:T120'
with zipfile.ZipFile(tmp) as z:
 for n,b in parts.items():
  if n not in updates:assert z.read(n)==b
assert p.read_bytes()==raw;os.replace(tmp,p);(out/f'frcb-move-{stamp}.json').write_text(json.dumps({'file':str(p),'backup':str(backup),'frcb_rows':[7,8],'removed':'empty trailing FRCB-only section heading','sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'validation':'Every moved cell matches original except explicit SOTA-reference note; IDs, metrics, images preserved'},indent=2));print('FRCB moved to rows7–8 in published references; original metrics and IDs preserved.')

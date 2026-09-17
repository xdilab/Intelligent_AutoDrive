from pathlib import Path
from copy import deepcopy
from lxml import etree as E
import zipfile,openpyxl,datetime,hashlib,json,os,re
p=Path('/home/brandon/Downloads/ROAD-Waymo Results.xlsx');w=openpyxl.load_workbook(p);s=w.worksheets[0];N='http://schemas.openxmlformats.org/spreadsheetml/2006/main';q=lambda t:'{'+N+'}'+t
variants={}
for ri in range(2,s.max_row+1):
 if s.cell(ri,1).value is None:continue
 desc=str(s.cell(ri,3).value or '');short=desc.split(' | ')[0]
 if short.startswith(('ROAD full adaptation;','ROAD small pilot;')):
  v='Frozen encoder + focal heads' if 'Frozen visual encoder' in short else ('ROAD focal + contrastive' if 'contrastive' in short else 'ROAD focal only')
  if '; baseline;' in short:v='Pre-adaptation baseline ('+v+')'
  if short.startswith('ROAD small pilot;'):v+=' [small pilot]'
 elif short.startswith('BDD-X paired;'):v='BDD-X contrastive adaptation' if 'BDD-X-adapted' in short else 'Original encoder control'
 elif short.startswith('Original controlled;'):
  v='Original encoder'
  if 'shuffled' in short:v+=' + shuffled phrases'
  if 'matched-width' in short:v+=' + flat-evidence control'
 elif short.startswith(('BCE gates;','AP-selected blends;')):
  v='BCE-fitted blend/gate' if 'BCE fitting' in short else ('Development-AP-selected blend' if 'development AP selection' in short else 'Original encoder control')
  if 'shuffled' in short:v+=' + shuffled phrases'
 elif short.startswith('BDD-X contrastive subset pilot;'):v='Original dual encoder' if short.endswith('baseline') else 'BDD-X contrastive adaptation'
 elif short.startswith('Published Table5;'):v='Published five-run mean'
 elif short.startswith('Published Table3;'):v='Published best-result row'
 else:v=short
 variants[ri]=v
raw=p.read_bytes();stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S');out=Path('/data/repos/wiki/artifacts/results-master');backup=out/f'ROAD-Waymo-Results-before-variant-{stamp}.xlsx';backup.write_bytes(raw)
with zipfile.ZipFile(p) as z:parts={n:z.read(n) for n in z.namelist()}
r=E.fromstring(parts['xl/worksheets/sheet1.xml']);sd=r.find(q('sheetData'))
for row in sd:
 ri=int(row.get('r'))
 for c in row:
  ref=c.get('r');letters=''.join(filter(str.isalpha,ref));ci=openpyxl.utils.column_index_from_string(letters)
  if ci>=3:c.set('r',f'{openpyxl.utils.get_column_letter(ci+1)}{ri}')
 val='Variant' if ri==1 else variants.get(ri)
 if val is not None:
  c=E.Element(q('c'),r=f'C{ri}',s=str(s.cell(ri,3).style_id),t='inlineStr');E.SubElement(E.SubElement(c,q('is')),q('t')).text=val;row.insert(2,c)
 if ri==1:
  c=next(c for c in row if c.get('r')=='D1')
  for x in list(c):c.remove(x)
  c.set('t','inlineStr');E.SubElement(E.SubElement(c,q('is')),q('t')).text='study / configuration details'
cols=r.find(q('cols'))
for col in cols:
 if int(col.get('min'))>=3:col.set('min',str(int(col.get('min'))+1));col.set('max',str(int(col.get('max'))+1))
cols.insert(2,E.Element(q('col'),min='3',max='3',width='34',customWidth='1'));r.find(q('dimension')).set('ref',f'A1:T{s.max_row}')
table=E.fromstring(parts['xl/tables/table1.xml']);table.set('ref',f'A1:T{s.max_row}');table.find(q('autoFilter')).set('ref',f'A1:T{s.max_row}');tc=table.find(q('tableColumns'));nc=deepcopy(tc[2]);nc.set('id',str(max(int(x.get('id')) for x in tc)+1));nc.set('name','Variant')
for attr in list(nc.attrib):
 if attr.endswith('uid'):del nc.attrib[attr]
tc.insert(2,nc);tc[3].set('name','study / configuration details');tc.set('count','20')
# Update architecture key's reference to study column and dedicated variant.
k=E.fromstring(parts['xl/worksheets/architecture-key.xml'])
for t in k.iter(q('t')):
 if t.text:t.text=t.text.replace('Keep baseline/epoch and Original controlled/BDD-X paired/BCE gates/AP blend/ROAD pilot/ROAD full in columnC.','Variant is columnC; study/configuration details are columnD; checkpoint remains in the epoch column and details.')
updates={'xl/worksheets/sheet1.xml':E.tostring(r),'xl/tables/table1.xml':E.tostring(table),'xl/worksheets/architecture-key.xml':E.tostring(k)}
tmp=p.with_name('ROAD-Waymo Results.updating.xlsx')
with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
 for n,b in {**parts,**updates}.items():z.writestr(n,b)
a=openpyxl.load_workbook(tmp);ss=a.worksheets[0]
for row in s:
 for c in row:
  if c.coordinate=='C1':continue
  assert ss.cell(c.row,c.column+(c.column>=3)).value==c.value,c.coordinate
for ri,v in variants.items():assert ss.cell(ri,3).value==v
assert ss.tables['Table5'].ref=='A1:T121' and len(ss.tables['Table5'].tableColumns)==20
with zipfile.ZipFile(tmp) as z:
 for n,b in parts.items():
  if n not in updates:assert z.read(n)==b
assert p.read_bytes()==raw;os.replace(tmp,p)
(out/f'variants-{stamp}.json').write_text(json.dumps({'file':str(p),'backup':str(backup),'populated_variants':len(variants),'columns':'B architecture; C Variant; D details; S tail; T notes','sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'validation':'All previous cells preserved at shifted positions except renamed description header; other ZIP parts unchanged'},indent=2)+'\n');print('Added Variant columnC for',len(variants),'rows; all results and diagrams preserved.')

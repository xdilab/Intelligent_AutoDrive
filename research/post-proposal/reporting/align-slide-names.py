from pathlib import Path
import zipfile,openpyxl,json,hashlib,datetime,os
from lxml import etree as E
p=Path('/home/brandon/Downloads/ROAD-Waymo Results.xlsx');w=openpyxl.load_workbook(p);N='http://schemas.openxmlformats.org/spreadsheetml/2006/main';q=lambda x:'{'+N+'}'+x
mapping={'Stage 1 — YOLO + I3D Score Transfer':'Stage 1: Score Transfer (Exp 11)','Stage 2 — YOLO + I3D RoI Classifier':'Stage 2: RoI Linear Scoring Head (Exp 11)','Stage 3 — I3D Composition MLP':'Stage 3: RoI Linear Scoring Head + Composition MLP (Exp 11)','Stage 4 — Crop Phrase Head':'Stage 4: Semantic Crop Classifier (Exp 13)','Stage 5 — Visual Composition Expert':'Stage 5: Semantic Crop Classifier + Comp MLP (Exp 13)','Stage 6 — Phrase-Fusion MLP':'Stage 6: Semantic Crop Classifier + Phrase Fusion + Comp MLP (Exp 14)'}
raw=p.read_bytes();stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S');out=Path('/data/repos/wiki/artifacts/results-master');backup=out/f'ROAD-Waymo-Results-before-slide-names-{stamp}.xlsx';backup.write_bytes(raw)
with zipfile.ZipFile(p) as z:parts={n:z.read(n) for n in z.namelist()}
r=E.fromstring(parts['xl/worksheets/sheet1.xml']);changes=[]
def put(c,v):
 for x in list(c):c.remove(x)
 c.set('t','inlineStr');E.SubElement(E.SubElement(c,q('is')),q('t')).text=v
for row in r.find(q('sheetData')):
 ri=int(row.get('r'));old=w.worksheets[0].cell(ri,2).value
 if old not in mapping:continue
 cells={c.get('r'):c for c in row};new=mapping[old];put(cells[f'B{ri}'],new);changes.append({'row':ri,'old':old,'new':new})
 detail=w.worksheets[0].cell(ri,4).value or ''
 if old.startswith('Stage 4'):detail+=' | Phrase classifier on InternVideo2 crop features; no composition MLP.'
 if old.startswith('Stage 5'):detail+=' | Flat primitive scores + crop features feed Comp MLP; no explicit phrase-score branch.'
 if old.startswith('Stage 6'):detail+=' | Flat primitive + phrase composition scores + crop features feed Comp MLP.'
 put(cells[f'D{ri}'],detail)
key=E.fromstring(parts['xl/worksheets/architecture-key.xml'])
for t in key.iter(q('t')):
 if t.text in mapping:t.text=mapping[t.text]
sd=key.find(q('sheetData'));ri=len(sd)+1;row=E.SubElement(sd,q('row'),r=str(ri))
for col,v in [('A','Slide naming and experiment lineage'),('B','Exp11: Stages1–3; Exp13 crop study: Stages4–5; Exp14: Stage6. These identify architecture origins, not new adaptation run IDs.'),('C','Semantic Crop Classifier is the crop-model family name: Stage4 uses a phrase head, Stage5 flat primitives plus Comp MLP, Stage6 adds phrase-composition scores. Study and adaptation remain in Variant/details.')]:
 c=E.SubElement(row,q('c'),r=f'{col}{ri}');put(c,v)
updates={'xl/worksheets/sheet1.xml':E.tostring(r),'xl/worksheets/architecture-key.xml':E.tostring(key)};tmp=p.with_name('ROAD-Waymo Results.updating.xlsx')
with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
 for n,b in {**parts,**updates}.items():z.writestr(n,b)
a=openpyxl.load_workbook(tmp)
for row in w.worksheets[0]:
 for c in row:
  if c.column not in [2,4]:assert a.worksheets[0].cell(c.row,c.column).value==c.value
for c in changes:assert a.worksheets[0].cell(c['row'],2).value==c['new']
with zipfile.ZipFile(tmp) as z:
 for n,b in parts.items():
  if n not in updates:assert z.read(n)==b
assert p.read_bytes()==raw;os.replace(tmp,p);(out/f'slide-names-{stamp}.json').write_text(json.dumps({'file':str(p),'backup':str(backup),'mapping':mapping,'changed_rows':changes,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'validation':'All variants, numeric cells, tables, styles and diagram bytes unchanged'},indent=2));print('Aligned',len(changes),'rows to presentation names; preserved variants and metrics.')

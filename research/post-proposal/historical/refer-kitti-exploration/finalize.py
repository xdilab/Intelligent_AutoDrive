from pathlib import Path
import json,hashlib,collections
from PIL import Image
root=Path('/data/datasets/Refer-KITTI');out=Path(__file__).resolve().parent
source=root/'downloads/sequence-verification.json';d=json.loads(source.read_text());assert d['image_files']==6650 and d['verified_crc_all'];dimensions=collections.Counter()
for f in d['files']:
 p=root/'KITTI'/f['path'];assert p.stat().st_size==f['bytes']
 with Image.open(p) as im:dimensions[str(im.size)]+=1
missing=[]
for p in (root/'expression').rglob('*.json'):
 for frame in json.loads(p.read_text())['label']:
  q=root/'KITTI/training/image_02'/p.parent.name/f'{int(frame):06d}.png'
  if not q.exists():missing.append(str(q))
assert not missing
summary={k:v for k,v in d.items() if k!='files'};summary.update({'manifest_path':str(source),'manifest_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'all_image_headers_readable':True,'image_dimensions':dict(dimensions),'missing_expression_images':len(missing)})
(out/'download-verification.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
p=Path('datasets/refer-kitti.md');s=p.read_text().replace('status: draft','status: complete');a=s.index('**Image download status:**');b=s.index('\n\n',a);s=s[:a]+'**Image download status:** complete. All 6,650 selected images passed ZIP CRC verification and have SHA-256 hashes in the dataset manifest. Every image header is readable; all expression-referenced images exist. Selected image payload is 5,405,631,238 bytes (5.41 GB / 5.03 GiB). Summary: `artifacts/refer-kitti-exploration/download-verification.json`.'+s[b:];p.write_text(s)

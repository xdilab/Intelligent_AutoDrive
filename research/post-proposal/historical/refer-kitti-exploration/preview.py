import io,zipfile,json,requests,html,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
out=Path(__file__).resolve().parent
from remote_archive import Remote
samples=json.loads((out/'samples.json').read_text());(out/'previews').mkdir(exist_ok=True)
with zipfile.ZipFile(Remote()) as z:
 infos=[i for i in z.infolist() if i.filename.startswith('training/image_02/') and not i.is_dir()]
 (out/'archive-directory.json').write_text(json.dumps({'training_files':len(infos),'training_bytes':sum(i.file_size for i in infos),'all_files':len(z.infolist()),'all_uncompressed_bytes':sum(i.file_size for i in z.infolist())},indent=2))
 for i,s in enumerate(samples):
  p=out/'previews'/f'{i:02d}.png'
  if p.exists():continue
  raw=z.read(s['image_member']);im=Image.open(io.BytesIO(raw)).convert('RGB');w,h=im.size;_,oid,cx,cy,bw,bh=s['box'];box=(cx*w,cy*h,(cx+bw)*w,(cy+bh)*h)
  ImageDraw.Draw(im).rectangle(box,outline='#ffcc00',width=4);im.save(p)
  print(i,s['keyword'],s['video'],s['frame'],flush=True)
rows=[]
for i,s in enumerate(samples):rows.append(f'<figure><img src="previews/{i:02d}.png"><figcaption>{html.escape(s["sentence"])}<br>Sequence {s["video"]}, frame {s["frame"]}, actor {s["object_id"]}</figcaption></figure>')
(out/'exploration.html').write_text('''<!doctype html><meta charset="utf-8"><title>Refer-KITTI exploration</title><style>body{font:17px system-ui;background:#101820;color:#eef3f5;margin:32px}main{max-width:1500px;margin:auto}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}figure{margin:0;background:#24313c;padding:12px;border-radius:8px}img{width:100%}figcaption{padding:10px 0}p{max-width:1000px;line-height:1.6}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style><main><h1>Refer-KITTI: what does the language actually supervise?</h1><p>Downloaded original annotations: 818 expressions across 18 sequences. All 356,515 expression/frame/actor references resolve to supplied boxes. Yellow boxes show the same selected actor at three labeled times per row. Times need not be consecutive. These are exploratory samples, not verified ROAD label mappings.</p><p>Turning: 24 expressions / 9 videos. Braking: 6 / 3. Parking: 60 / 11. Walking: 20 / 3. Standing: 4 / 1. Selection favors a large target box for visibility. "Left" can mean image position, "turning" lacks turn direction, and "walking" does not establish crossing.</p><div class="grid">'''+''.join(rows)+'</div><p>Sources: <a href="https://github.com/wudongming97/RMOT">Refer-KITTI authors</a> and <a href="https://www.cvlibs.net/datasets/kitti/eval_tracking.php">KITTI tracking</a>. Ground-truth boxes from the authors’ labels_with_ids archive; images retrieved from the official KITTI archive. September 14, 2026.</p></main>')

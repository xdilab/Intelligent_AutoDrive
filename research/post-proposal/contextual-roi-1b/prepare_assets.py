"""Download pinned assets from short-lived signed URLs; never receive an account token."""
from pathlib import Path
import argparse,hashlib,json,time,urllib.request

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
 private=r/'.asset-downloads.json';assets=json.loads(private.read_text());records=[]
 try:
  for entry in assets:
   kind=entry['kind'];dest=r/'assets'/entry['filename'];part=dest.with_suffix(dest.suffix+'.partial')
   if not dest.exists():
    offset=part.stat().st_size if part.exists() else 0
    headers={'User-Agent':'Mozilla/5.0'}
    if offset:headers['Range']=f'bytes={offset}-'
    req=urllib.request.Request(entry['download_url'],headers=headers)
    with urllib.request.urlopen(req,timeout=120) as response:
     append=offset>0 and response.status==206
     if not append:offset=0
     if append:assert response.headers.get('Content-Range','').startswith(f'bytes {offset}-')
     total=offset;reported=time.monotonic()
     with part.open('ab' if append else 'wb') as out:
      while True:
       b=response.read(8<<20)
       if not b:break
       out.write(b);total+=len(b)
       if time.monotonic()-reported>30:
        print('DOWNLOAD_PROGRESS',kind,total,entry['bytes'],flush=True);reported=time.monotonic()
    assert part.stat().st_size==entry['bytes'],(kind,'size mismatch')
    assert digest(part)==entry['sha256'],(kind,'SHA256 mismatch')
    part.replace(dest)
   assert dest.stat().st_size==entry['bytes'] and digest(dest)==entry['sha256']
   rec={k:v for k,v in entry.items() if k!='download_url'};rec['path']=str(dest);records.append(rec)
   print('VERIFIED',kind,rec['bytes'],rec['sha256'],flush=True)
   (r/'results/assets-progress.json').write_text(json.dumps({'files':records,'time':time.time()},indent=2))
  (r/'results/assets.json').write_text(json.dumps({'passed':True,'files':records,'time':time.time()},indent=2));print('ASSETS_COMPLETE',flush=True)
 finally:private.unlink(missing_ok=True)
if __name__=='__main__':main()

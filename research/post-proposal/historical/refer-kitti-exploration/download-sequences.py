import io,json,zipfile,requests,struct,zlib,concurrent.futures,time,hashlib
from pathlib import Path
out=Path(__file__).resolve().parent;root=Path('/data/datasets/Refer-KITTI')
from remote_archive import Remote
vids={p.name for p in (root/'expression').iterdir() if p.is_dir()}
with zipfile.ZipFile(Remote()) as z:
 allinfo=z.infolist(); infos=sorted([i for i in allinfo if i.filename.startswith('training/image_02/') and i.filename.split('/')[2] in vids and not i.is_dir()],key=lambda i:i.header_offset)
 # Each member fits before the next local header; use its exact compressed payload size after parsing the local header.
 groups=[];group=[]
 for i in infos:
  if group and (i.header_offset-group[0].header_offset>32*1024*1024):groups.append(group);group=[]
  group.append(i)
 if group:groups.append(group)
print('Requested',len(vids),'sequences',len(infos),'images',sum(i.file_size for i in infos),'bytes',len(groups),'chunks',flush=True)
def fetch(group):
 if all((root/'KITTI'/i.filename).exists() and (root/'KITTI'/i.filename).stat().st_size==i.file_size for i in group):return len(group)
 a=group[0].header_offset;last=group[-1];b=last.header_offset+30+len(last.filename.encode())+65535+last.compress_size-1
 for attempt in range(5):
  try:
   r=requests.get('https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip',headers={'Range':f'bytes={a}-{b}'},timeout=(20,90));r.raise_for_status();assert r.status_code==206;buf=r.content
   for i in group:
    off=i.header_offset-a;h=struct.unpack_from('<4s5H3I2H',buf,off);assert h[0]==b'PK\x03\x04';start=off+30+h[-2]+h[-1];raw=buf[start:start+i.compress_size];data=zlib.decompress(raw,-15) if i.compress_type==8 else raw
    assert len(data)==i.file_size and zlib.crc32(data)==i.CRC
    p=root/'KITTI'/i.filename;p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():p.write_bytes(data)
   return len(group)
  except Exception:
   if attempt==4:raise
   time.sleep(2)
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 done=0
 for f in concurrent.futures.as_completed([pool.submit(fetch,g) for g in groups]):done+=f.result();print('verified',done,'/',len(infos),flush=True)
manifest=[]
for i in infos:
 p=root/'KITTI'/i.filename;data=p.read_bytes();assert zlib.crc32(data)==i.CRC;manifest.append({'path':i.filename,'bytes':len(data),'crc32':i.CRC,'sha256':hashlib.sha256(data).hexdigest()})
report={'videos':sorted(vids),'image_files':len(infos),'image_bytes':sum(i.file_size for i in infos),'source':'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip','verified_crc_all':True,'files':manifest}
(root/'downloads/sequence-verification.json').write_text(json.dumps(report,indent=2));print('COMPLETE',flush=True)

"""Build a small video-disjoint manifest from released clips and original BDD-X text."""
import argparse,csv,hashlib,json,random,sys
from pathlib import Path

def norm(s):return ' '.join(s.lower().split()).rstrip('.')
def main():
 p=argparse.ArgumentParser();p.add_argument('--videos',type=Path,required=True);p.add_argument('--conversations',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True);p.add_argument('--train-split',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 allowed={Path(x.strip()).stem.split('_',1)[-1] for x in a.train_split.read_text().splitlines()}
 originals={}
 with a.annotations.open() as f:
  for row in csv.DictReader(f):
   vid=Path(row['Input.Video']).stem
   for i in range(1,16):
    text=row.get(f'Answer.{i}action','').strip()
    if text: originals.setdefault(vid,set()).add(norm(text))
 groups={};skips={'not_original_train':0,'unmatched_action':0,'missing_video':0}
 for r in json.loads(a.conversations.read_text()):
  name=Path(r['video'][0]).name;vid=Path(name).stem.rsplit('_',1)[0]
  if vid not in allowed:skips['not_original_train']+=1;continue
  txt=r['conversations'][1]['value'].strip()
  if norm(txt) not in originals.get(vid,set()):skips['unmatched_action']+=1;continue
  path=a.videos/name
  if not path.is_file():skips['missing_video']+=1;continue
  groups.setdefault(vid,[]).append({'source_video':vid,'video':str(path.resolve()),'text':norm(txt),'annotation_action':txt})
 rng=random.Random(20260911);vids=sorted(groups);rng.shuffle(vids)
 assert len(vids)>=1152, f'Need 1152 verified source videos, got {len(vids)}; {skips}'
 rows=[rng.choice(sorted(groups[v],key=lambda r:r['video'])) for v in vids[:1152]]
 sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'packages'))
 import av
 for r in rows:
  with av.open(r['video']) as c:
   stream=c.streams.video[0];n=stream.frames
   if not n:n=sum(1 for _ in c.decode(video=0))
   assert n>=8, f"Insufficient real frames: {r['video']}"
   r['decoded_stream_frames']=n
 data={'train':rows[128:],'dev':rows[:128],'seed':20260911,'source':'SafeAuto released processed clips; original CSV action text verified','caption':'action only; no justification, controls, generated answers, or RAG','split':'original train videos only; 128 internal-dev videos, 1024 train; one segment per video','skips':skips,'eligible_videos':len(vids),'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.annotations,a.conversations,a.train_split]}}
 a.out.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps({k:v for k,v in data.items() if k not in ['train','dev']}))
if __name__=='__main__':main()

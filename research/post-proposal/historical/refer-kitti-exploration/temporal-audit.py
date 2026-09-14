from pathlib import Path
import json,collections
r=Path('/data/datasets/Refer-KITTI/expression');out=Path(__file__).resolve().parent;ds=[(p.parent.name,json.loads(p.read_text())) for p in r.rglob('*.json')];report={}
for word in ['turning','braking','parking','moving','walking','standing']:
 tracks=collections.defaultdict(set)
 for v,d in ds:
  if word not in d['sentence'].split():continue
  for f,ids in d['label'].items():
   for oid in ids:tracks[(v,int(oid))].add(int(f))
 lengths=[]
 for frames in tracks.values():
  fs=sorted(frames);n=1
  for a,b in zip(fs,fs[1:]):
   if b==a+1:n+=1
   else:lengths.append(n);n=1
  lengths.append(n)
 report[word]={'distinct_video_actor_pairs':len(tracks),'unique_actor_frames':sum(map(len,tracks.values())),'contiguous_runs':len(lengths),'runs_at_least_8_frames':sum(n>=8 for n in lengths),'runs_at_least_32_frames':sum(n>=32 for n in lengths),'max_run_frames':max(lengths)}
(out/'temporal-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

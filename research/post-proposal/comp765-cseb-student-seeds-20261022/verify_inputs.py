"""Check every evaluation image against the original study's recorded bytes."""
from common import *
def main():
 verify();splits=read(ROOT/'split-manifest.json');count=0;started=time.perf_counter()
 for number,video in enumerate(splits['groups']['final'],1):
  old=read(ORIGINAL_ROOT/'evaluation-inputs'/f'{video}.json');assert old['reset_at']==1 and old['processed_frames']==splits['final_last_frame'][video]
  for fid,h in old['images_sha256'].items():assert sha(image_path(video,int(fid)))==h,(video,fid);count+=1
  if number%20==0:print(json.dumps(dict(verified_videos=number,verified_images=count,seconds=time.perf_counter()-started)),flush=True)
 assert count==38663
 bundle=read(ORIGINAL_ROOT/'bundle-seed0/bundle.json');assert sha(p.WEIGHTS)==bundle['files_sha256']['backbone.pth'];assert sha(p.YOLO)==bundle['files_sha256']['yolo.pt'];assert sha(p.TEACHER)==bundle['teacher_checkpoint_sha256']
 save(ROOT/'input-parity.json',dict(time=time.time(),passed=True,videos=198,images=count,seconds=time.perf_counter()-started,original_input_records=str(ORIGINAL_ROOT/'evaluation-inputs'),backbone_sha256=sha(p.WEIGHTS),detector_sha256=sha(p.YOLO),teacher_sha256=sha(p.TEACHER)));print('ALL INPUT HASHES MATCH ORIGINAL',flush=True)
if __name__=='__main__':main()

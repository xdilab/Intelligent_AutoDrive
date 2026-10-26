"""Reverify raw input hashes against the original study's recorded provenance."""
from assoc_common import *
import importlib.metadata

def main():
 original_record=read(BASE_ROOT/'provenance.json');verified={}
 for key in ['frames','counts','yolo','gt','backbone','detector','annotation','teacher_partition']:
  entry=original_record['inputs'][key];assert sha(entry['path'])==entry['sha256'],key;verified[key]=entry
 versions={name:importlib.metadata.version(name) for name in ['torch','torchvision','numpy','opencv-python','ultralytics','Pillow','matplotlib']}
 save(ROOT/'asset-audit.json',dict(passed=True,time=time.time(),verified=verified,versions=versions,note='Teacher model is not needed or opened by this pilot; original teacher partition is read only to preserve split provenance.'))
 print('Verified',len(verified),'raw asset hashes',flush=True)
if __name__=='__main__':main()

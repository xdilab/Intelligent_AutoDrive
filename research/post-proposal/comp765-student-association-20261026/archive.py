"""Archive compact, verified evidence in the repository; retain full live artifacts."""
from assoc_common import *
import shutil

def main():
 assert read(ROOT/'final-audit.json')['passed'];dest=HERE/'results';assert not dest.exists(),'Refuse to overwrite an archived study';dest.mkdir();files=[]
 def copy(src,relative):
  out=dest/relative;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,out);assert sha(src)==sha(out);files.append(dict(path=str(relative),sha256=sha(out),bytes=out.stat().st_size))
 for src in sorted(ROOT.glob('*.json')):copy(src,Path(src.name))
 copy(ROOT/'rules.csv',Path('rules.csv'))
 for src in sorted((ROOT/'analysis').iterdir()):
  if src.is_file():copy(src,Path(src.name))
 for directory in ['runs','evaluation-inputs','monitor','logs']:
  for src in sorted((ROOT/directory).rglob('*')):
   if src.is_file() and src.suffix in ['.json','.jsonl','.log']:copy(src,src.relative_to(ROOT))
 for src in sorted((ROOT/'runtime').glob('*/report.json')):copy(src,src.relative_to(ROOT))
 save(dest/'archive-manifest.json',dict(time=time.time(),source_root=str(ROOT),files=files,total_bytes=sum(r['bytes'] for r in files),large_artifacts_retained_in_live_root=['development-seed*.npz','runtime/*/predictions.jsonl.gz'],original_checkpoints_unchanged=True))
 print('Archived',len(files),'files;',sum(r['bytes'] for r in files),'bytes')
if __name__=='__main__':main()
